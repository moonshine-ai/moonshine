#include "neural_tts/pb_decoder.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if defined(PICO_BUILD)
#include "pico/time.h"
// Progress hook (defined by the app): records where the pipeline is for
// post-reboot hang reports AND feeds the hardware watchdog. Called before
// every TFLM op and every synth pulse, so it doubles as the watchdog
// heartbeat -- IRQ-timer feeding proved unreliable (the repeating timer
// died after one beat while Invoke was running).
extern "C" void tts_checkpoint(uint32_t v);
extern "C" void tts_trace(uint32_t tag, uint32_t val);
#define PB_CHECKPOINT(v) tts_checkpoint(v)
#define PB_TRACE(tag, val) tts_trace((tag), (val))
#else
#define PB_CHECKPOINT(v) \
  do {                   \
  } while (0)
#define PB_TRACE(tag, val) \
  do {                     \
  } while (0)
#endif

namespace neural_tts {
namespace {

uint64_t NowUs() {
#if defined(PICO_BUILD)
  return time_us_64();
#else
  return 0;
#endif
}


// exp10f(x) = 10^x for the benv dB/20 -> amplitude conversion
inline float Exp10(float x) { return expf(x * 2.302585093f); }

constexpr int kNumOps = 6;
using Resolver = tflite::MicroMutableOpResolver<kNumOps>;

// Bring-up: checkpoint 100+N = executing the N-th op of the current
// Invoke(), so the post-reboot hang report identifies the guilty kernel
// (map N to an op via the .tflite execution order).
class CkptProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char*) override {
    PB_CHECKPOINT(100 + op_index_);
    ++op_index_;
    return 0;
  }
  void EndEvent(uint32_t) override {}
  void Reset() { op_index_ = 0; }

 private:
  uint32_t op_index_ = 0;
};

CkptProfiler g_ckpt_profiler;

}  // namespace

PbDecoder::PbDecoder(const Config& config, uint8_t* arena,
                     size_t arena_bytes)
    : cfg_(config) {
  model_ = tflite::GetModel(cfg_.model_data);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("pb_decoder: model schema %d != %d",
                static_cast<int>(model_->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Place the resolver + interpreter at the head of the arena
  // (classifier.cc pattern: one contiguous block, no malloc).
  constexpr size_t kStatics = 1024;
  if (arena_bytes < kStatics + (1u << 16)) {
    MicroPrintf("pb_decoder: arena too small");
    return;
  }
  size_t off = 0;
  auto place = [&](size_t size) -> void* {
    off = (off + 15u) & ~static_cast<size_t>(15u);
    void* p = arena + off;
    off += size;
    return p;
  };
  auto* resolver = new (place(sizeof(Resolver))) Resolver();
  if (resolver->AddTranspose() != kTfLiteOk ||
      resolver->AddReshape() != kTfLiteOk ||
      resolver->AddTransposeConv() != kTfLiteOk ||
      resolver->AddAdd() != kTfLiteOk ||
      resolver->AddGelu() != kTfLiteOk ||
      resolver->AddConv2D() != kTfLiteOk) {
    MicroPrintf("pb_decoder: op registration failed");
    return;
  }
  auto* interp_mem = place(sizeof(tflite::MicroInterpreter));
  if (off > kStatics) {
    MicroPrintf("pb_decoder: statics overflow");
    return;
  }
  interp_ = new (interp_mem) tflite::MicroInterpreter(
      model_, *resolver, arena + kStatics, arena_bytes - kStatics,
      /*resource_variables=*/nullptr, &g_ckpt_profiler);
  if (interp_->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("pb_decoder: AllocateTensors failed");
    return;
  }
  TfLiteTensor* in = interp_->input(0);
  TfLiteTensor* out = interp_->output(0);
  if (in->type != kTfLiteInt16 || out->type != kTfLiteInt16) {
    MicroPrintf("pb_decoder: expected int16 I/O, got %d/%d", in->type,
                out->type);
    return;
  }
  in_data_ = in->data.i16;
  out_data_ = out->data.i16;
  ok_ = true;
}

size_t PbDecoder::arena_used_bytes() const {
  return interp_ ? interp_->arena_used_bytes() : 0;
}

void PbDecoder::BeginUtterance(const PbCodedUtterance* utt) {
  utt_ = utt;
  win_frame0_ = 0;
  win_frames_ = 0;
  next_latent_ = 0;
  decode_us_ = 0;
  tiles_decoded_ = 0;
}

bool PbDecoder::DecodeTileAt(int latent_start) {
  PB_CHECKPOINT(20);
  PB_TRACE(5, static_cast<uint32_t>(latent_start));
  const uint64_t t0 = NowUs();
  const int margin = (cfg_.tile_latents - cfg_.tile_hop) / 2;
  int lo = latent_start - margin;
  if (lo < 0) lo = 0;

  // build int16 latents: sum of int8 codebook rows, requantized.
  // q_ is a member: this runs below the vocoder's render frame now that
  // decode is lazy, and the stack is a fixed 4 KB scratch bank.
  const int dim = cfg_.latent_dim;
  float* q = q_;
  for (int j = 0; j < cfg_.tile_latents; ++j) {
    const int lj = lo + j;
    int16_t* dst = in_data_ + j * dim;
    if (lj >= utt_->n_latents) {
      memset(dst, 0, sizeof(int16_t) * dim);
      continue;
    }
    for (int d = 0; d < dim; ++d) q[d] = 0.0f;
    for (int s = 0; s < cfg_.n_stages; ++s) {
      const int code = utt_->codes[lj * cfg_.n_stages + s];
      const int8_t* row = cfg_.codebooks[s] + code * dim;
      const float* sc = cfg_.codebook_scales[s];
      for (int d = 0; d < dim; ++d) q[d] += row[d] * sc[d];
    }
    const float inv_scale = 1.0f / cfg_.input_scale;
    for (int d = 0; d < dim; ++d) {
      float v = q[d] * inv_scale;
      v = v < 0.0f ? v - 0.5f : v + 0.5f;  // round half away from zero
      int iv = static_cast<int>(v);
      if (iv > 32767) iv = 32767;
      if (iv < -32768) iv = -32768;
      dst[d] = static_cast<int16_t>(iv);
    }
  }

  PB_CHECKPOINT(21);
  g_ckpt_profiler.Reset();
  if (interp_->Invoke() != kTfLiteOk) {
    MicroPrintf("pb_decoder: Invoke failed");
    return false;
  }
  PB_CHECKPOINT(22);
  memcpy(win_buf_, out_data_,
         sizeof(int16_t) * cfg_.tile_latents * 4 * 60);
  win_frame0_ = lo * 4;
  win_frames_ = cfg_.tile_latents * 4;
  next_latent_ = latent_start + cfg_.tile_hop;
  ++tiles_decoded_;
  decode_us_ += NowUs() - t0;
  PB_TRACE(6, static_cast<uint32_t>(tiles_decoded_));
  PB_CHECKPOINT(23);
  return true;
}

void PbDecoder::GetFrame(int t, WorldFrame* frame) {
  PB_CHECKPOINT(10);
  // f0 comes from the side stream, not the net
  frame->f0 = utt_->f0q[t] * (1.0f / 16.0f);

  const int lt = t / 4;
  if (win_frames_ == 0 && !DecodeTileAt(0)) return;
  // advance while the frame is past the current tile's kept region
  // (kept latents are [next_latent_ - hop, next_latent_))
  while (lt >= next_latent_) {
    if (!DecodeTileAt(next_latent_)) return;  // don't retry forever
  }
  if (t < win_frame0_) {
    // backward request beyond the tile overlap: re-decode a tile whose
    // kept region starts at lt (breaks the regular tile grid; only a
    // fallback -- the synth access pattern is monotonic within +/-1 frame,
    // well inside the (tile - hop) / 2 latent margin)
    DecodeTileAt(lt);
  }

  PB_CHECKPOINT(11);
  const int16_t* row = win_buf_ + (t - win_frame0_) * 60;
  const float os = cfg_.output_scale;
  for (int i = 0; i < kWorldNumBenv; ++i)
    frame->benv[i] = Exp10(row[i] * os);
  for (int i = 0; i < kWorldNumBap; ++i) {
    float b = row[kWorldNumBenv + i] * os;
    if (b < 0.0f) b = 0.0f;
    if (b > 1.0f) b = 1.0f;
    frame->bap[i] = b;
  }
  PB_CHECKPOINT(12);
}

void PbDecoder::GetFrameThunk(void* user, int t, WorldFrame* frame) {
  static_cast<PbDecoder*>(user)->GetFrame(t, frame);
}

void PbDecoder::ReadRows(int t0, int n, int16_t* out) {
  int t = t0;
  const int end = t0 + n;
  while (t < end) {
    const int lt = t / 4;
    bool decode_ok = true;
    if (win_frames_ == 0 || lt >= next_latent_) {
      // A jump of a full hop or more (skipped span) restarts on the tile
      // grid at lt instead of decoding every intermediate tile.
      const bool jump =
          win_frames_ == 0 || lt >= next_latent_ + cfg_.tile_hop;
      decode_ok = DecodeTileAt(jump ? (lt / cfg_.tile_hop) * cfg_.tile_hop
                                    : next_latent_);
    } else if (t < win_frame0_) {
      // backward seek beyond the tile overlap (shouldn't happen with the
      // monotonic access pattern); re-decode off-grid as a fallback
      decode_ok = DecodeTileAt(lt);
    }
    if (!decode_ok) {
      // Invoke failed; window state didn't advance. Zero-fill and bail
      // instead of retrying forever.
      MicroPrintf("pb_decoder: ReadRows decode failed at t=%d", t);
      memset(out + (t - t0) * 60, 0, sizeof(int16_t) * (end - t) * 60);
      return;
    }
    // copy every decoded frame in [t, kept-window end) that we still need
    const int win_end = next_latent_ * 4;
    int m = win_end - t;
    if (t + m > end) m = end - t;
    if (m <= 0 || t < win_frame0_ ||
        (t - win_frame0_) + m > win_frames_) {
      // decode failed to advance (Invoke error) or window inconsistent:
      // zero-fill the remainder instead of looping / a wild memcpy
      MicroPrintf("pb_decoder: ReadRows stalled at t=%d", t);
      memset(out + (t - t0) * 60, 0, sizeof(int16_t) * (end - t) * 60);
      return;
    }
    memcpy(out + (t - t0) * 60, win_buf_ + (t - win_frame0_) * 60,
           sizeof(int16_t) * m * 60);
    t += m;
  }
}

}  // namespace neural_tts
