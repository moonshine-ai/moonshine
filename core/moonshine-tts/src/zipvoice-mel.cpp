#include "zipvoice-mel.h"

#include <cmath>
#include <cstddef>

namespace moonshine_tts {

namespace {

constexpr double kPi = 3.14159265358979323846;

int hz_to_bin_count(int n_fft) { return n_fft / 2 + 1; }

double hz_to_mel_htk(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double mel_to_hz_htk(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// Iterative radix-2 Cooley-Tukey FFT for power-of-two ``n`` (in-place, natural -> natural order).
void fft_radix2(std::vector<double>& re, std::vector<double>& im) {
  const size_t n = re.size();
  // Bit-reversal permutation.
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * kPi / static_cast<double>(len);
    const double wlen_re = std::cos(ang);
    const double wlen_im = std::sin(ang);
    for (size_t i = 0; i < n; i += len) {
      double w_re = 1.0;
      double w_im = 0.0;
      for (size_t k = 0; k < len / 2; ++k) {
        const double u_re = re[i + k];
        const double u_im = im[i + k];
        const double v_re = re[i + k + len / 2] * w_re - im[i + k + len / 2] * w_im;
        const double v_im = re[i + k + len / 2] * w_im + im[i + k + len / 2] * w_re;
        re[i + k] = u_re + v_re;
        im[i + k] = u_im + v_im;
        re[i + k + len / 2] = u_re - v_re;
        im[i + k + len / 2] = u_im - v_im;
        const double nw_re = w_re * wlen_re - w_im * wlen_im;
        const double nw_im = w_re * wlen_im + w_im * wlen_re;
        w_re = nw_re;
        w_im = nw_im;
      }
    }
  }
}

// Reflect index into [0, len) the same way torch ``pad(mode="reflect")`` mirrors without repeating
// the edge sample.
size_t reflect_index(long idx, long len) {
  if (len <= 1) {
    return 0;
  }
  const long period = 2 * (len - 1);
  long m = idx % period;
  if (m < 0) {
    m += period;
  }
  if (m >= len) {
    m = period - m;
  }
  return static_cast<size_t>(m);
}

}  // namespace

VocosFbank::VocosFbank() {
  hann_.resize(static_cast<size_t>(kNFft));
  for (int k = 0; k < kNFft; ++k) {
    // Periodic Hann window (torch.hann_window(win_length, periodic=True)).
    hann_[static_cast<size_t>(k)] =
        static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * k / static_cast<double>(kNFft)));
  }

  // torchaudio.functional.melscale_fbanks with htk scale and norm=None.
  const int n_freqs = hz_to_bin_count(kNFft);
  const double f_min = 0.0;
  const double f_max = static_cast<double>(kSampleRate) / 2.0;
  std::vector<double> all_freqs(static_cast<size_t>(n_freqs));
  for (int i = 0; i < n_freqs; ++i) {
    all_freqs[static_cast<size_t>(i)] =
        static_cast<double>(i) * f_max / static_cast<double>(n_freqs - 1);
  }
  const double m_min = hz_to_mel_htk(f_min);
  const double m_max = hz_to_mel_htk(f_max);
  std::vector<double> f_pts(static_cast<size_t>(kNMels + 2));
  for (int i = 0; i < kNMels + 2; ++i) {
    const double m = m_min + (m_max - m_min) * static_cast<double>(i) / static_cast<double>(kNMels + 1);
    f_pts[static_cast<size_t>(i)] = mel_to_hz_htk(m);
  }
  std::vector<double> f_diff(static_cast<size_t>(kNMels + 1));
  for (int i = 0; i < kNMels + 1; ++i) {
    f_diff[static_cast<size_t>(i)] = f_pts[static_cast<size_t>(i + 1)] - f_pts[static_cast<size_t>(i)];
  }

  fb_.assign(static_cast<size_t>(n_freqs) * static_cast<size_t>(kNMels), 0.F);
  for (int f = 0; f < n_freqs; ++f) {
    const double freq = all_freqs[static_cast<size_t>(f)];
    for (int m = 0; m < kNMels; ++m) {
      // slopes to the (m) and (m+2) mel points; triangular filter peaks at (m+1).
      const double down = -(f_pts[static_cast<size_t>(m)] - freq) / f_diff[static_cast<size_t>(m)];
      const double up = (f_pts[static_cast<size_t>(m + 2)] - freq) / f_diff[static_cast<size_t>(m + 1)];
      double v = std::min(down, up);
      if (v < 0.0) {
        v = 0.0;
      }
      fb_[static_cast<size_t>(f) * static_cast<size_t>(kNMels) + static_cast<size_t>(m)] =
          static_cast<float>(v);
    }
  }
}

int VocosFbank::num_frames_for(size_t num_samples) {
  return 1 + static_cast<int>(num_samples / static_cast<size_t>(kHop));
}

std::vector<float> VocosFbank::extract(const std::vector<float>& samples, int* out_frames) const {
  const long L = static_cast<long>(samples.size());
  const int frames = num_frames_for(samples.size());
  if (out_frames != nullptr) {
    *out_frames = frames;
  }
  const int n_freqs = hz_to_bin_count(kNFft);
  const long pad = kNFft / 2;  // center padding
  std::vector<float> out(static_cast<size_t>(frames) * static_cast<size_t>(kNMels), 0.F);
  if (L <= 0 || frames <= 0) {
    return out;
  }

  std::vector<double> re(static_cast<size_t>(kNFft));
  std::vector<double> im(static_cast<size_t>(kNFft));
  std::vector<double> mag(static_cast<size_t>(n_freqs));

  for (int t = 0; t < frames; ++t) {
    const long start = static_cast<long>(t) * kHop - pad;  // index into original signal
    for (int k = 0; k < kNFft; ++k) {
      const long idx = start + k;
      float sample;
      if (idx >= 0 && idx < L) {
        sample = samples[static_cast<size_t>(idx)];
      } else {
        sample = samples[reflect_index(idx, L)];
      }
      re[static_cast<size_t>(k)] = static_cast<double>(sample) * static_cast<double>(hann_[static_cast<size_t>(k)]);
      im[static_cast<size_t>(k)] = 0.0;
    }
    fft_radix2(re, im);
    for (int f = 0; f < n_freqs; ++f) {
      const double r = re[static_cast<size_t>(f)];
      const double i = im[static_cast<size_t>(f)];
      mag[static_cast<size_t>(f)] = std::sqrt(r * r + i * i);  // power=1 (magnitude)
    }
    float* row = out.data() + static_cast<size_t>(t) * static_cast<size_t>(kNMels);
    for (int m = 0; m < kNMels; ++m) {
      double acc = 0.0;
      for (int f = 0; f < n_freqs; ++f) {
        acc += static_cast<double>(fb_[static_cast<size_t>(f) * static_cast<size_t>(kNMels) +
                                       static_cast<size_t>(m)]) *
               mag[static_cast<size_t>(f)];
      }
      if (acc < 1e-7) {
        acc = 1e-7;
      }
      row[m] = static_cast<float>(std::log(acc));
    }
  }
  return out;
}

}  // namespace moonshine_tts
