// SPDX-License-Identifier: MIT
// Minimal RIFF WAVE reader: PCM 16-bit LE → float32 mono [-1, 1], plus linear
// resampling.

#ifndef WAV_PCM_FLOAT32_H_
#define WAV_PCM_FLOAT32_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wav_pcm {

inline std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("wav: failed to open: " + path);
  }
  f.seekg(0, std::ios::end);
  const std::streamoff sz = f.tellg();
  if (sz <= 0) {
    throw std::runtime_error("wav: empty file");
  }
  f.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> buf(static_cast<size_t>(sz));
  f.read(reinterpret_cast<char*>(buf.data()), sz);
  if (!f) {
    throw std::runtime_error("wav: read failed");
  }
  return buf;
}

inline std::uint32_t u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint16_t u16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         (static_cast<std::uint16_t>(p[1]) << 8);
}

// PCM 16 LE mono or stereo (mean to mono) → float32 mono, sample_rate_out set
// from header.
inline std::vector<float> load_wav_pcm16_mono_float32(const std::string& path,
                                                      int& sample_rate_out) {
  const std::vector<std::uint8_t> raw = read_file_bytes(path);
  if (raw.size() < 44) {
    throw std::runtime_error("wav: file too small");
  }
  if (raw[0] != 'R' || raw[1] != 'I' || raw[2] != 'F' || raw[3] != 'F') {
    throw std::runtime_error("wav: not RIFF");
  }
  size_t off = 12;
  int num_channels = 0;
  int bits_per_sample = 0;
  int sample_rate = 0;
  size_t data_off = 0;
  size_t data_size = 0;
  while (off + 8 <= raw.size()) {
    const std::uint32_t id = u32(raw.data() + off);
    const std::uint32_t sz = u32(raw.data() + off + 4);
    const size_t chunk_data = off + 8;
    const size_t chunk_end = chunk_data + sz;
    if (chunk_end > raw.size()) {
      break;
    }
    if (id == 0x20746d66) {  // 'fmt '
      if (sz < 16) {
        throw std::runtime_error("wav: fmt too small");
      }
      const std::uint16_t audio_format = u16(raw.data() + chunk_data);
      num_channels = static_cast<int>(u16(raw.data() + chunk_data + 2));
      sample_rate = static_cast<int>(u32(raw.data() + chunk_data + 4));
      bits_per_sample = static_cast<int>(u16(raw.data() + chunk_data + 14));
      if (audio_format != 1) {
        throw std::runtime_error("wav: only PCM format 1 supported");
      }
      if (bits_per_sample != 16) {
        throw std::runtime_error("wav: only 16-bit PCM supported");
      }
      if (num_channels < 1 || num_channels > 8) {
        throw std::runtime_error("wav: unsupported channel count");
      }
    } else if (id == 0x61746164) {  // 'data'
      data_off = chunk_data;
      data_size = sz;
    }
    off = chunk_end + (sz & 1u);  // word align
  }
  if (data_off == 0 || data_size == 0 || sample_rate <= 0) {
    throw std::runtime_error("wav: missing fmt or data chunk");
  }
  if (data_size % static_cast<size_t>(2 * num_channels) != 0) {
    throw std::runtime_error("wav: data size not multiple of frame");
  }
  const size_t num_frames = data_size / static_cast<size_t>(2 * num_channels);
  std::vector<float> mono(num_frames);
  const std::uint8_t* d = raw.data() + data_off;
  for (size_t i = 0; i < num_frames; ++i) {
    float sum = 0.f;
    for (int ch = 0; ch < num_channels; ++ch) {
      const size_t o = i * static_cast<size_t>(num_channels) * 2u +
                       static_cast<size_t>(ch) * 2u;
      const int16_t s = static_cast<int16_t>(u16(d + o));
      sum += static_cast<float>(s) / 32768.f;
    }
    mono[i] = sum / static_cast<float>(num_channels);
  }
  sample_rate_out = sample_rate;
  return mono;
}

inline std::vector<float> linear_resample(const std::vector<float>& x,
                                          int sr_in, int sr_out) {
  if (sr_in == sr_out || x.empty()) {
    return x;
  }
  if (sr_in <= 0 || sr_out <= 0) {
    throw std::runtime_error("resample: bad sample rate");
  }
  const double ratio = static_cast<double>(sr_out) / static_cast<double>(sr_in);
  const size_t n_out = static_cast<size_t>(std::max<std::int64_t>(
      1, static_cast<std::int64_t>(
             std::llround(static_cast<double>(x.size()) * ratio))));
  std::vector<float> y(n_out);
  for (size_t j = 0; j < n_out; ++j) {
    const double src_pos = (static_cast<double>(j) + 0.5) /
                               static_cast<double>(n_out) *
                               static_cast<double>(x.size()) -
                           0.5;
    const double src_f =
        std::max(0., std::min(static_cast<double>(x.size() - 1), src_pos));
    const size_t i0 = static_cast<size_t>(std::floor(src_f));
    const size_t i1 = std::min(i0 + 1, x.size() - 1);
    const double t = src_f - static_cast<double>(i0);
    y[j] = static_cast<float>((1. - t) * static_cast<double>(x[i0]) +
                              t * static_cast<double>(x[i1]));
  }
  return y;
}

}  // namespace wav_pcm

#endif  // WAV_PCM_FLOAT32_H_
