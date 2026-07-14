#include "synth_stream.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "g2p/g2p.h"
#include "phonemes.h"

namespace tts {

namespace {

// Soft limiter: perfectly linear up to a knee (so the RMS body of the signal is
// untouched), then a saturating soft knee that rounds peaks toward +/-1 instead
// of hard-clipping. Only the rare overshoot the fixed output_gain doesn't
// account for is shaped, avoiding the hard-clip buzz without dulling loudness.
inline float SoftClip(float x) {
  constexpr float kKnee = 0.8f;
  constexpr float kRange =
      1.0f - kKnee;  // headroom above the knee, ceiling 1.0
  const float a = std::fabs(x);
  if (a <= kKnee) return x;
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  return s * (kKnee + kRange * std::tanh((a - kKnee) / kRange));
}

}  // namespace

StreamSynth::StreamSynth(const VoiceParams& vp, uint8_t* arena,
                         size_t arena_size)
    : vp_(vp), arena_(arena), arena_size_(arena_size) {}

uint8_t* StreamSynth::ArenaBytes(size_t count, size_t align) {
  const size_t base = (arena_used_ + (align - 1)) & ~(align - 1);
  if (arena_ == nullptr || base + count > arena_size_) return nullptr;
  arena_used_ = base + count;
  return arena_ + base;
}

float* StreamSynth::ArenaFloats(size_t count) {
  return reinterpret_cast<float*>(
      ArenaBytes(count * sizeof(float), alignof(float)));
}

int StreamSynth::BeginText(const char* text, const StreamOptions& opts,
                           const g2p::Lexicon* overrides) {
  if (text == nullptr) return kStreamErrBadArg;
  const std::vector<std::string> phones =
      g2p::TextToPhones(std::string(text), overrides);
  return BeginPhones(phones, opts);
}

int StreamSynth::BeginIpa(const char* ipa, const StreamOptions& opts) {
  if (ipa == nullptr) return kStreamErrBadArg;
  const std::vector<std::string> phones = g2p::TokenizeIpa(std::string(ipa));
  return BeginPhones(phones, opts);
}

int StreamSynth::BeginPhones(const std::vector<std::string>& phones,
                             const StreamOptions& opts) {
  if (opts.sample_rate < 1.0f) return kStreamErrBadArg;

  // 1) Phones -> segments (shared with the batch path).
  std::vector<synth_detail::Segment> segs =
      synth_detail::BuildSegments(phones, vp_);
  if (segs.empty()) return kStreamErrNoPhones;

  const float dur_scale =
      vp_.duration_scale * ((opts.speed > 0.01f) ? (1.0f / opts.speed) : 1.0f);
  nframes_ = synth_detail::CountFrames(segs, dur_scale);
  if (nframes_ == 0) return kStreamErrNoPhones;

  spf_ = std::max(1, static_cast<int>(std::lround(
                         opts.sample_rate * synth_detail::kFrameMs / 1000.0f)));
  sr_hz_ = static_cast<int>(std::lround(opts.sample_rate));

  // 2) Carve the parameter tracks + a one-frame PCM scratch out of the arena.
  ArenaReset();
  float* fptrs[15];
  for (int i = 0; i < 15; ++i) {
    fptrs[i] = ArenaFloats(nframes_);
    if (fptrs[i] == nullptr) return kStreamErrArenaFull;
  }
  uint8_t* major = ArenaBytes(nframes_, alignof(uint8_t));
  frame_buf_ = ArenaFloats(static_cast<size_t>(spf_));
  if (major == nullptr || frame_buf_ == nullptr) return kStreamErrArenaFull;

  tracks_.f1 = fptrs[0];
  tracks_.f2 = fptrs[1];
  tracks_.f3 = fptrs[2];
  tracks_.b1 = fptrs[3];
  tracks_.b2 = fptrs[4];
  tracks_.b3 = fptrs[5];
  tracks_.av = fptrs[6];
  tracks_.af = fptrs[7];
  tracks_.ah = fptrs[8];
  tracks_.nasal = fptrs[9];
  tracks_.fnp = fptrs[10];
  tracks_.fnz = fptrs[11];
  tracks_.fric_cf = fptrs[12];
  tracks_.accent = fptrs[13];
  tracks_.f0 = fptrs[14];
  tracks_.major = major;
  tracks_.n = nframes_;

  // 3-4b) Rasterize + smooth + F0 + voice-identity scaling (shared).
  synth_detail::FillParamTracks(segs, vp_, dur_scale, opts.question, tracks_);

  // 5) Fresh Klatt core for this utterance.
  synth_.emplace(opts.sample_rate, synth_detail::MakeKlattParams(vp_));

  // 6) Reset streaming cursors + loudness stage.
  frame_idx_ = 0;
  frame_buf_pos_ = 0;
  frame_buf_len_ = 0;
  out_idx_ = 0;
  total_samples_ = nframes_ * static_cast<size_t>(spf_);
  fade_ = std::min<int>(static_cast<int>(total_samples_ / 2),
                        static_cast<int>(opts.sample_rate * 0.005f));
  gain_ = vp_.output_gain;
  return kStreamOk;
}

void StreamSynth::RenderNextFrame() {
  if (frame_idx_ >= nframes_ || !synth_) {
    frame_buf_pos_ = 0;
    frame_buf_len_ = 0;
    return;
  }
  const SynthFrame cur = synth_detail::FrameAt(tracks_, frame_idx_);
  const SynthFrame nxt =
      synth_detail::FrameAt(tracks_, std::min(frame_idx_ + 1, nframes_ - 1));
  synth_->RenderFrame(cur, nxt, spf_, frame_buf_);
  frame_buf_pos_ = 0;
  frame_buf_len_ = spf_;
  ++frame_idx_;
}

int StreamSynth::Read(float* out, int max_samples) {
  if (out == nullptr || max_samples <= 0) return 0;
  int written = 0;
  while (written < max_samples) {
    if (frame_buf_pos_ >= frame_buf_len_) {
      if (frame_idx_ >= nframes_) break;  // utterance fully rendered
      RenderNextFrame();
      if (frame_buf_len_ == 0) break;
    }
    float s = SoftClip(frame_buf_[frame_buf_pos_] * gain_);
    // Short edge fades to suppress boundary clicks (same 5 ms as batch).
    if (fade_ > 0) {
      const size_t gi = out_idx_;
      if (gi < static_cast<size_t>(fade_)) {
        s *= static_cast<float>(gi) / static_cast<float>(fade_);
      } else if (gi + static_cast<size_t>(fade_) >= total_samples_) {
        s *= static_cast<float>(total_samples_ - 1 - gi) /
             static_cast<float>(fade_);
      }
    }
    out[written++] = s;
    ++frame_buf_pos_;
    ++out_idx_;
  }
  return written;
}

}  // namespace tts
