// Black-box neural TTS pipeline (see neural_tts.h). Port of the host
// reference scripts/synth_diphone_world.py synthesize() with the shipping
// flag set; every stage is annotated with its Python counterpart.
//
// Arena life cycle per chunk (all caller arena; the G2P front end's small
// std::string/vector transients are the only malloc heap use):
//
//   [plan: runs, D, parts, candidate sets, track, f0]   -- whole chunk
//   [decode: PbDecoder (+TFLM arena), one-unit cache]   -- rolled back
//   [render: WorldLiteSynth + kissfft plans in arena]    -- reuses decode's
//
// The control track is materialized in the decoder's *quantized log
// domain* (int16, output_scale, benv as log10-amplitude): all assembly
// math (crossfades, gain EQ, timbre/energy offsets, declination) is
// additive there, and frames convert to linear WorldFrames only at the
// vocoder boundary.

#include "neural_tts/neural_tts.h"

#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "g2p/g2p.h"
#if defined(PICO_BUILD)
#include "g2p/g2p_phones.h"
#endif
#include "neural_tts/pb_decoder.h"
#include "neural_tts/worldlite_synth.h"
#include "tts/config.h"
#include "tts/synth_internal.h"

#if defined(PICO_BUILD)
#include "pico/time.h"
#endif

#if defined(PICO_BUILD)
// Secondary progress hook (see hooks.cc / the app's watchdog handler):
// records pipeline position in a watchdog scratch register without feeding
// the watchdog, so a post-watchdog-reboot STATUS localizes hangs between
// the decoder's tts_checkpoint calls.
extern "C" void tts_checkpoint2(uint32_t v);
#define NT_CHECKPOINT2(v) tts_checkpoint2(v)
#else
#define NT_CHECKPOINT2(v) \
  do {                    \
  } while (0)
#endif

namespace neural_tts {
namespace {

inline uint64_t NowUs() {
#if defined(PICO_BUILD)
  return time_us_64();
#else
  return 0;
#endif
}

// Host reference flags (scripts/export_pb_demo_data.py eval configuration).
constexpr float kWJoin = 5.0f;
constexpr float kWDur = 0.3f;
constexpr float kWCtx = 0.3f;
constexpr int kXfadeHw = 3;
constexpr float kTimbreNorm = 0.3f;
constexpr float kWWordDur = 2.0f;
constexpr float kWWordF0 = 1.0f;
constexpr float kWordReuse = 0.5f;

constexpr int kCap = 8;              // candidate pool cap (pack stores <= 6)
// ~2 s per synthesis chunk. Bounded by arena: the track (60 int16/frame)
// coexists with the TFLM decoder arena (~160 KB) AND the ~57 KB vocoder
// now that decode is lazy (interleaved with render), so longer chunks
// starve AllocateTensors in the 340 KiB arena. Chunk length no longer
// affects time-to-first-audio; splits still prefer silence boundaries.
#ifndef NEURAL_TTS_MAX_CHUNK_FRAMES
#define NEURAL_TTS_MAX_CHUNK_FRAMES 400
#endif
constexpr int kMaxChunkFrames = NEURAL_TTS_MAX_CHUNK_FRAMES;
constexpr int kMaxRuns = 200;        // phone segments per chunk
constexpr int kMaxParts = 220;
constexpr int kMaxUnitFrames = 272;  // largest stored unit is 262 frames
constexpr int kMaxWordKey = 24;
constexpr int kMaxUsedWords = 48;

constexpr float kLn10 = 2.302585093f;

inline float Exp10(float x) { return expf(x * kLn10); }

// --- Planning-time loudness contour ------------------------------------------
// Concatenative TTS from a known unit inventory: the whole unit sequence -- and
// each unit's baked loudness *shape* (kPackLoudKnots log-summed-amplitude knots)
// -- is known before a single sample is rendered, so loudness leveling is a
// planning problem, not a runtime one. PlanLoudness() reconstructs the
// utterance's loudness contour from the baked knots plus the exact per-part
// offsets the synth will apply (gain_q makeup + prosody energy), finds the loud
// reference with full lookahead, and lifts every too-quiet region up to a floor
// below it -- so the onset of "wifi"/"hello" is raised to match the vowels that
// FOLLOW it, which a causal runtime AGC can never see. The result is a smooth
// per-frame log-amplitude offset applied at render (RenderGetFrame), composing
// with the boundary gain-EQ and joins rather than layering a filter on top.
// Boost-only; everything is expressed relative to the utterance's own reference
// (pack-agnostic). LSA / boosts are natural-log amplitude; +1 nat ~= +8.686 dB.
//
// Tune by ear. kLoudLevelEnabled=false bypasses the pass entirely.
constexpr bool kLoudLevelEnabled = true;
constexpr float kLoudFloorDrop = 0.6f;     // floor = ref - this (~ -5 dB)
constexpr float kLoudGateDrop = 3.5f;      // below ref-this: consonant/quiet, skip
constexpr float kLoudMaxBoost = 2.0f;      // cap the lift (~+17 dB)
constexpr int kLoudSmoothHw = 12;          // boxcar half-width (frames) on boost
constexpr int kLoudRefHw = 2;              // half-width for the reference smoother
constexpr float kLoudSilentLsa = -1000.0f; // sentinel for silence-part frames

// ---------------------------------------------------------------------------
// Small readers for the pack's packed streams.

class BitReader {
 public:
  explicit BitReader(const uint8_t* p) : p_(p) {}
  uint32_t get(int bits) {
    while (nbits_ < bits) {
      acc_ |= static_cast<uint64_t>(*p_++) << nbits_;
      nbits_ += 8;
    }
    const uint32_t v = static_cast<uint32_t>(acc_ & ((1u << bits) - 1));
    acc_ >>= bits;
    nbits_ -= bits;
    return v;
  }

 private:
  const uint8_t* p_;
  uint64_t acc_ = 0;
  int nbits_ = 0;
};

int ReadVarU8(const uint8_t*& p) {
  int v = 0;
  while (*p == 255) {
    v += 255;
    ++p;
  }
  v += *p++;
  return v;
}

// Voiced-run table for DecodeF0Stream. The stream's run count is a byte,
// so 255 is the hard upper bound; the caller passes arena scratch for it
// (2 KB is too much for the 4 KB scratch-bank stack).
struct F0RunSpan {
  int start, len;
};
constexpr int kMaxF0Runs = 255;

// SynthesizeChunk returns this when planning exceeds kMaxChunkFrames; RunAfterBuild
// retries with a smaller run range split at the last word gap.
constexpr int kChunkTooLong = -4;

// Decode a unit's f0 side stream into per-frame Hz (0 = unvoiced).
void DecodeF0Stream(const uint8_t* p, int n_frames, float* out,
                    F0RunSpan* runs) {
  memset(out, 0, sizeof(float) * n_frames);
  const int n_runs = *p++;
  int pos = 0;
  for (int r = 0; r < n_runs; ++r) {
    pos += ReadVarU8(p);
    const int len = ReadVarU8(p) + 1;
    runs[r] = {pos, len};
    pos += len;
  }
  for (int r = 0; r < n_runs; ++r) {
    const int s = runs[r].start, len = runs[r].len;
    // knot count: grid every 4 frames + off-grid last frame
    int n_knots = (len - 1) / 4 + 1;
    if ((len - 1) % 4 != 0) ++n_knots;
    float codes[80];
    if (n_knots > 80) n_knots = 80;  // bounded by kMaxUnitFrames / 4 + 1
    int cur = *p++;
    codes[0] = static_cast<float>(cur);
    const uint8_t* nib = p;
    for (int k = 1; k < n_knots; ++k) {
      int d = (nib[(k - 1) >> 1] >> (((k - 1) & 1) * 4)) & 0xF;
      if (d >= 8) d -= 16;
      cur += d;
      codes[k] = static_cast<float>(cur);
    }
    p += (n_knots - 1 + 1) / 2;  // byte-padded per run
    // piecewise linear between knots (knot k at frame min(4k, len-1))
    for (int k = 0; k + 1 < n_knots || (n_knots == 1 && k == 0); ++k) {
      const int f0i = 4 * k;
      const int f1i = (k + 1 < n_knots)
                          ? ((4 * (k + 1) < len - 1) ? 4 * (k + 1) : len - 1)
                          : f0i;
      const float c0 = codes[k];
      const float c1 = codes[(k + 1 < n_knots) ? k + 1 : k];
      const int span = f1i - f0i;
      for (int f = f0i; f <= f1i && f < len; ++f) {
        const float a = span > 0 ? static_cast<float>(f - f0i) / span : 0.0f;
        const float code = c0 + (c1 - c0) * a;
        if (s + f < n_frames) {
          out[s + f] =
              kPackF0BaseHz * exp2f(code / kPackF0StepsPerOctave);
        }
      }
      if (k + 1 >= n_knots) break;
    }
  }
}

inline float F0FromCode(uint8_t q) {
  return q == 0 ? 0.0f
               : kPackF0BaseHz * exp2f(q / kPackF0StepsPerOctave);
}

// ---------------------------------------------------------------------------
// Plan data structures.

struct Part {
  enum Kind : uint8_t { kSilence, kDiphone, kWord };
  Kind kind;
  int32_t unit;          // diphone unit index / word unit index / -1
  uint16_t h2, h1;       // half lengths (kSilence / kDiphone)
  uint16_t out_frames;   // total output frames (h2 + h1 for halves)
  uint16_t cut;          // diphone source cut frame
  float f0_scale_a, f0_scale_b;  // per-half f0 multipliers
  float e_off_a, e_off_b;        // per-half natural-log energy offsets
  int16_t centroid_type;         // diphone type index for timbre norm, -1
};

// Bump allocator over the caller arena.
class Bump {
 public:
  Bump(uint8_t* base, size_t size) : base_(base), size_(size) {}
  void* Alloc(size_t bytes, size_t align = 4) {
    size_t off = (used_ + align - 1) & ~(align - 1);
    if (off + bytes > size_) return nullptr;
    used_ = off + bytes;
    return base_ + off;
  }
  template <typename T>
  T* AllocArray(size_t n, size_t align = 4) {
    return static_cast<T*>(Alloc(n * sizeof(T), align));
  }
  size_t Mark() const { return used_; }
  void Reset(size_t mark) { used_ = mark; }
  size_t remaining() const { return size_ - used_; }

 private:
  uint8_t* base_;
  size_t size_;
  size_t used_ = 0;
};

// ---------------------------------------------------------------------------
// The per-Synthesize() engine.

class Engine {
 public:
  Engine(const Pack& pk, uint8_t* arena, size_t arena_size,
         NeuralTts::Stats* stats)
      : pk_(pk), bump_(arena, arena_size), stats_(stats) {}

  // `tokens` is cleared after the front end so its heap is free again
  // before the kissfft render plans are allocated. plan_only runs the
  // deterministic planning passes and returns the exact sample count
  // without decoding or rendering.
  int Run(std::vector<std::string>* tokens, NeuralTts::EmitFn emit,
          void* user, bool plan_only);
#if defined(PICO_BUILD)
  int Run(const g2p::PhoneTokenList* phones, NeuralTts::EmitFn emit,
          void* user, bool plan_only);
#endif

  // Materialize + post-process frames until frame t is final; called by
  // the vocoder's get_frame bridge (lazy decode-behind-render).
  void EnsureFinal(int t);

 private:
  // -- front end --------------------------------------------------------
  int PhoneId(const char* token) const;
  int BuildRuns(const std::vector<std::string>& tokens);
  int BuildRunsFromPtrs(const char* const* tokens, int n_tokens);
  int RunAfterBuild(NeuralTts::EmitFn emit, void* user, bool plan_only);

  // -- pack lookups ------------------------------------------------------
  int FindDiphoneType(int a, int b) const;
  int FindWord(const uint8_t* key, int len) const;  // first match or -1

  bool IsSil(int pid) const {
    return pid == static_cast<int>(pk_.h().sil_id) ||
           pid == static_cast<int>(pk_.h().dot_id);
  }
  bool IsGap(int pid) const {
    return pid == static_cast<int>(pk_.h().gap_id);
  }
  int Canon(int pid) const {
    return pid == static_cast<int>(pk_.h().dot_id)
               ? static_cast<int>(pk_.h().sil_id)
               : pid;
  }

  // -- chunk pipeline ----------------------------------------------------
  int SynthesizeChunk(int lo, int hi, bool first, bool last,
                      NeuralTts::EmitFn emit, void* user, bool plan_only);
  void ComputeProsodyBuckets(int n);
  float ProsOff(const float* table, int chunk_start) const;
  float SegOff(const float* table, int seg) const;
  void MatchWords(int n);
  void SelectDiphones(int n);
  void BuildParts(int n);

  // -- materialization (f0 prepass + lazy track pass) ---------------------
  // Warp ranges of one part (diphone halves / single word range).
  struct Range {
    int src0, src_n, out_n;
    bool anchor_end, plain;
    float f0s, eoff;
  };
  int BuildRanges(const Part& p, int T, Range ranges[2]) const;
  void MaterializeF0();               // fills f0_[0..T_) for the chunk
  void MaterializePartTrack(int pi);  // fills track_ rows for one part
  void GainEqAt(int pi);              // gain EQ across boundary pi/pi+1
  void SmoothJoinAt(int j);           // join smoothing at frame j
  void AdvanceJoins();                // run join passes behind mat frontier
  float FrameLnEnergy(int t) const;   // ln(sum benv linear) of track row t
  void PlanLoudness();                // full-lookahead loudness contour plan
  void F0Pass();

  // -- unit decode / cache ------------------------------------------------
  const int16_t* DecodedRows(bool word, int idx, int frame_base,
                             int* n_frames, float* mean_lnb /*48*/);

  // emit wrapper that timestamps the first PCM chunk of the call
  struct EmitShim {
    NeuralTts::EmitFn emit;
    void* user;
    Engine* eng;
  };
  static void TimedEmit(void* user, const int16_t* samples, int n) {
    auto* sh = static_cast<EmitShim*>(user);
    if (sh->eng->stats_->first_pcm_us == 0) {
      sh->eng->stats_->first_pcm_us =
          static_cast<uint32_t>(NowUs() - sh->eng->t_call_);
    }
    sh->emit(sh->user, samples, n);
  }
  void UnpackCodes(uint32_t codes_off, int n_latents, uint16_t* out);

  static int BlendLenUnit(int rule_n, int nat_n) {
    // dur_mode="unit": natural length, clipped to [0.6, 1.6] x rule.
    int lo = static_cast<int>(0.6f * rule_n + 0.5f);
    int hi = static_cast<int>(1.6f * rule_n + 0.5f);
    if (lo < 1) lo = 1;
    if (hi < 1) hi = 1;
    int n = nat_n;
    if (n < lo) n = lo;
    if (n > hi) n = hi;
    return n;
  }

  const Pack& pk_;
  Bump bump_;
  NeuralTts::Stats* stats_;
  uint64_t t_call_ = 0;  // Run() entry, for first_pcm_us

  // whole-utterance runs (canonical phone id + rule frames)
  struct RunSeg {
    uint8_t pid;
    uint16_t rule_frames;
  };
  RunSeg* runs_ = nullptr;
  int n_runs_ = 0;

  // ---- per-chunk state (indices are chunk-local) ----
  int c_n_ = 0;                 // segments in chunk
  const RunSeg* c_ = nullptr;   // chunk segments
  uint16_t* D_ = nullptr;       // scaled durations
  // prosody buckets
  int16_t* seg_chunk_ = nullptr;   // segment -> chunk-start segment (-1)
  int8_t* chunk_func_ = nullptr;   // per segment (valid at chunk starts)
  int8_t* chunk_pos_ = nullptr;    // 0 init 1 medial 2 final 3 single
  // word coverage
  int32_t* word_at_ = nullptr;     // boundary -> word unit index or -1
  int16_t* word_end_ = nullptr;    // boundary -> jend
  uint8_t* covered_ = nullptr;     // boundary covered by a word unit
  // diphone selection
  uint16_t (*cands_)[kCap] = nullptr;
  uint8_t* n_cands_ = nullptr;
  int32_t* chosen_ = nullptr;      // boundary -> diphone unit or -1
  // parts + track
  Part* parts_ = nullptr;
  int n_parts_ = 0;
  uint16_t* part_start_ = nullptr;  // output frame of each part
  uint16_t* joins_ = nullptr;
  int n_joins_ = 0;
  int16_t (*track_)[60] = nullptr;
  float* f0_ = nullptr;
  int T_ = 0;

  // lazy-materialization pipeline state (see EnsureFinal): the vocoder
  // pulls frames, and parts are decoded just ahead of the render cursor
  // so the first PCM ships after ~one decoded tile, not the whole chunk.
  int mat_part_ = 0;    // parts [0, mat_part_) have track rows written
  int eq_bound_ = 0;    // gain-EQ boundaries [0, eq_bound_) applied
  int smooth_idx_ = 0;  // joins [0, smooth_idx_) smoothed
  int final_ = 0;       // frames [0, final_) are final (all passes)
  // planning-time loudness contour (PlanLoudness): per-frame log10-amplitude
  // boost applied at render, computed once with full-utterance lookahead.
  float* loud_boost_ = nullptr;  // [T_] log10-amplitude offset, or null
  // MaterializePartTrack scratch (runs below the vocoder render frame;
  // the stack is a fixed 4 KB scratch bank)
  float mean_lnb_[48];
  int16_t timbre_q_[48];

  // decode stream + one-unit copy cache -- allocated in the decode phase
  PbDecoder* dec_ = nullptr;
  PbCodedUtterance stream_;         // concatenated unit codes, part order
  int32_t* part_base_ = nullptr;    // part -> frame offset in stream (-1)
  int16_t* cache_rows_ = nullptr;
  float* cache_f0_ = nullptr;
  // arena scratch (too big for the 4 KB scratch-bank stack)
  F0RunSpan* f0_runs_ = nullptr;    // DecodeF0Stream voiced-run table
  float* posbuf_ = nullptr;         // Materialize warp positions
  float cache_mean_[48];
  int32_t cache_id_ = -1;  // (word << 30) | idx
  int cache_frames_ = 0;

  // word reuse penalty
  int32_t used_words_[kMaxUsedWords];
  uint8_t used_count_[kMaxUsedWords];
  int n_used_ = 0;
  float prev_tail_f0_ = 0.0f;
};

int Engine::PhoneId(const char* token) const {
  const char* t = token;
  if (t[0] == ' ' && t[1] == '\0') t = "_";
  for (uint32_t i = 0; i < pk_.h().n_phones; ++i) {
    if (strncmp(pk_.phone_token(i), t, 8) == 0) return static_cast<int>(i);
  }
  return -1;
}

int Engine::BuildRunsFromPtrs(const char* const* tokens, int n_tokens) {
  // Segment durations from the shared Klatt rule engine (context-dependent
  // duration rules; the same engine produced the host's "rule timeline").
  static const tts::VoiceParams* voice = [] {
    static tts::VoiceParams vp = tts::DefaultVoiceParams();
    return &vp;
  }();
  const int max_segs = n_tokens * 4 + 8;
  const size_t segs_mark = bump_.Mark();
  tts::synth_detail::Segment* segs =
      bump_.AllocArray<tts::synth_detail::Segment>(static_cast<size_t>(max_segs));
  if (segs == nullptr) return -1;
  const int n_segs = tts::synth_detail::BuildSegments(
      tokens, n_tokens, *voice, segs, max_segs);
  if (n_segs < 0) return -1;

  // Drop the temporary segment list before runs_ -- runs_ must live at the
  // bump head (offset 0) so SynthesizeChunk can allocate after it.
  bump_.Reset(segs_mark);

  runs_ = bump_.AllocArray<RunSeg>(static_cast<size_t>(n_segs) + 2);
  if (runs_ == nullptr) return -1;
  n_runs_ = 0;
  const int sil = static_cast<int>(pk_.h().sil_id);
  for (int si = 0; si < n_segs; ++si) {
    const tts::synth_detail::Segment& s = segs[si];
    int pid = sil;
    if (s.src_token >= 0 && s.src_token < n_tokens) {
      const int mapped = PhoneId(tokens[s.src_token]);
      if (mapped < 0) continue;  // stress marks etc. -> no frames
      pid = Canon(mapped);
    }
    const int frames = static_cast<int>(s.dur_ms / 5.0f + 0.5f);
    if (frames <= 0 && !s.is_silence) continue;
    if (n_runs_ > 0 && runs_[n_runs_ - 1].pid == pid) {
      runs_[n_runs_ - 1].rule_frames = static_cast<uint16_t>(
          runs_[n_runs_ - 1].rule_frames + frames);
    } else {
      runs_[n_runs_++] = {static_cast<uint8_t>(pid),
                          static_cast<uint16_t>(frames < 0 ? 0 : frames)};
    }
  }
  return n_runs_ > 0 ? 0 : -1;
}

int Engine::BuildRuns(const std::vector<std::string>& tokens) {
  std::vector<const char*> ptrs;
  ptrs.reserve(tokens.size());
  for (const std::string& t : tokens) ptrs.push_back(t.c_str());
  return BuildRunsFromPtrs(ptrs.data(), static_cast<int>(ptrs.size()));
}

int Engine::FindDiphoneType(int a, int b) const {
  const DiphoneTypeRec* t = pk_.dtypes();
  int lo = 0, hi = static_cast<int>(pk_.h().n_diphone_types) - 1;
  const int key = (a << 8) | b;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    const int k = (t[mid].a << 8) | t[mid].b;
    if (k == key) return mid;
    if (k < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

// Lexicographic compare of two phone-id keys.
static int KeyCompare(const uint8_t* a, int la, const uint8_t* b, int lb) {
  const int n = la < lb ? la : lb;
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return la == lb ? 0 : (la < lb ? -1 : 1);
}

int Engine::FindWord(const uint8_t* key, int len) const {
  const WordUnitRec* w = pk_.wunits();
  const uint8_t* blob = pk_.wkeys();
  int lo = 0, hi = static_cast<int>(pk_.h().n_words) - 1;
  int found = -1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    const uint8_t* k = blob + w[mid].key_off;
    const int c = KeyCompare(k + 1, k[0], key, len);
    if (c == 0) {
      found = mid;
      hi = mid - 1;  // first of the candidate run
    } else if (c < 0) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

// ---------------------------------------------------------------------------

int Engine::RunAfterBuild(NeuralTts::EmitFn emit, void* user, bool plan_only) {
  // Split into chunks at silence (fallback: word gap) boundaries so the
  // per-chunk control track stays bounded. If unit selection still plans
  // more frames than the arena allows, split again at the last gap inside
  // the range (never truncate audio).
  int total = 0;
  int lo = 0;
  bool first = true;
  while (lo < n_runs_) {
    int frames = 0;
    int hi = lo;
    int last_split = -1;
    while (hi < n_runs_ && frames < kMaxChunkFrames &&
           (hi - lo) < kMaxRuns - 4) {
      frames += static_cast<int>(
          runs_[hi].rule_frames *
          pk_.dur_ratio()[runs_[hi].pid]) + 2;
      if (IsSil(runs_[hi].pid) || IsGap(runs_[hi].pid)) last_split = hi;
      ++hi;
    }
    if (hi < n_runs_ && last_split > lo) hi = last_split + 1;

    int split_hi = hi;
    for (;;) {
      const bool last = split_hi >= n_runs_;
      const int n =
          SynthesizeChunk(lo, split_hi, first, last, emit, user, plan_only);
      if (n == kChunkTooLong) {
        int new_hi = lo + 1;
        for (int i = split_hi - 1; i > lo; --i) {
          if (IsSil(runs_[i].pid) || IsGap(runs_[i].pid)) {
            new_hi = i + 1;
            break;
          }
        }
        if (new_hi >= split_hi) return -1;
        split_hi = new_hi;
        continue;
      }
      if (n < 0) return n;
      total += n;
      hi = split_hi;
      break;
    }
    first = false;
    lo = hi;
  }
  return total;
}

int Engine::Run(std::vector<std::string>* tokens, NeuralTts::EmitFn emit,
                void* user, bool plan_only) {
  t_call_ = NowUs();
  const int rc = BuildRuns(*tokens);
  stats_->runs_us += static_cast<uint32_t>(NowUs() - t_call_);
  tokens->clear();
  tokens->shrink_to_fit();
  if (rc != 0) return -1;
  return RunAfterBuild(emit, user, plan_only);
}

#if defined(PICO_BUILD)
int Engine::Run(const g2p::PhoneTokenList* phones, NeuralTts::EmitFn emit,
                void* user, bool plan_only) {
  if (phones == nullptr || phones->count <= 0) return -1;
  t_call_ = NowUs();
  const char* ptrs[g2p::PhoneTokenList::kMaxTokens];
  for (int i = 0; i < phones->count; ++i) ptrs[i] = phones->tokens[i];
  const int rc = BuildRunsFromPtrs(ptrs, phones->count);
  stats_->runs_us += static_cast<uint32_t>(NowUs() - t_call_);
  if (rc != 0) return -1;
  return RunAfterBuild(emit, user, plan_only);
}
#endif

void Engine::ComputeProsodyBuckets(int n) {
  // synth_diphone_world.py: chunks (words) split by GAP, phrases by SIL;
  // bucket = (is_function_word, position in phrase).
  for (int i = 0; i < n; ++i) {
    seg_chunk_[i] = -1;
    chunk_func_[i] = 0;
    chunk_pos_[i] = 1;
  }
  // find word chunks (tables in arena scratch: 1.6 KB is too much for the
  // 4 KB scratch-bank stack)
  const size_t mark = bump_.Mark();
  int* chunk_starts = bump_.AllocArray<int>(kMaxRuns);
  int* chunk_ends = bump_.AllocArray<int>(kMaxRuns);
  if (chunk_starts == nullptr || chunk_ends == nullptr) return;
  int n_chunks = 0;
  int i = 0;
  while (i < n) {
    if (IsSil(c_[i].pid) || IsGap(c_[i].pid)) {
      ++i;
      continue;
    }
    int j = i;
    while (j < n && !IsSil(c_[j].pid) && !IsGap(c_[j].pid)) ++j;
    chunk_starts[n_chunks] = i;
    chunk_ends[n_chunks] = j;
    ++n_chunks;
    i = j;
  }
  // phrase grouping: SIL between chunks starts a new phrase
  int k = 0;
  while (k < n_chunks) {
    int pe = k;
    while (pe + 1 < n_chunks) {
      bool sil_between = false;
      for (int s = chunk_ends[pe]; s < chunk_starts[pe + 1]; ++s) {
        if (IsSil(c_[s].pid)) sil_between = true;
      }
      if (sil_between) break;
      ++pe;
    }
    const int len = pe - k + 1;
    for (int ci = k; ci <= pe; ++ci) {
      int8_t pos;
      if (len == 1)
        pos = 3;  // single
      else if (ci == k)
        pos = 0;  // initial
      else if (ci == pe)
        pos = 2;  // final
      else
        pos = 1;  // medial
      const int cs = chunk_starts[ci], ce = chunk_ends[ci];
      // function-word test: exact phone-id key lookup
      uint8_t key[kMaxWordKey];
      int klen = 0;
      for (int s = cs; s < ce && klen < kMaxWordKey; ++s)
        key[klen++] = c_[s].pid;
      bool is_func = false;
      const uint16_t* fidx = pk_.func_idx();
      const uint8_t* fblob = pk_.func_blob();
      for (uint32_t f = 0; f < pk_.h().n_func_keys; ++f) {
        const uint8_t* fk = fblob + fidx[f];
        if (fk[0] == klen && memcmp(fk + 1, key, klen) == 0) {
          is_func = true;
          break;
        }
      }
      chunk_func_[cs] = is_func ? 1 : 0;
      chunk_pos_[cs] = pos;
      for (int s = cs; s < ce; ++s) seg_chunk_[s] = static_cast<int16_t>(cs);
    }
    k = pe + 1;
  }
  bump_.Reset(mark);
}

float Engine::ProsOff(const float* table, int chunk_start) const {
  if (chunk_start < 0) return 0.0f;
  return table[chunk_func_[chunk_start] * 4 + chunk_pos_[chunk_start]];
}

float Engine::SegOff(const float* table, int seg) const {
  return seg < c_n_ ? ProsOff(table, seg_chunk_[seg]) : 0.0f;
}

void Engine::MatchWords(int n) {
  const int n_bound = n - 1;
  for (int j = 0; j < n_bound; ++j) {
    word_at_[j] = -1;
    word_end_[j] = -1;
    covered_[j] = 0;
  }
  const WordUnitRec* wrecs = pk_.wunits();
  const uint8_t* blob = pk_.wkeys();

  int i = 0;
  while (i < n) {
    if (IsSil(c_[i].pid) || IsGap(c_[i].pid)) {
      ++i;
      continue;
    }
    int j = i;
    while (j < n && !IsSil(c_[j].pid) && !IsGap(c_[j].pid)) ++j;

    // longest-phrase-first: extend over (gap, chunk) pairs
    int ends[8];
    int n_ends = 0;
    ends[n_ends++] = j;
    {
      int j2 = j;
      while (n_ends < 8 && j2 < n - 1 && IsGap(c_[j2].pid) &&
             !IsSil(c_[j2 + 1].pid) && !IsGap(c_[j2 + 1].pid)) {
        ++j2;
        while (j2 < n && !IsSil(c_[j2].pid) && !IsGap(c_[j2].pid)) ++j2;
        if (j2 <= n - 1) ends[n_ends++] = j2;
      }
    }
    int match = -1, jend = j;
    for (int e = n_ends - 1; e >= 0; --e) {
      const int je = ends[e];
      uint8_t key[kMaxWordKey];
      int klen = 0;
      bool fits = (je - i) <= kMaxWordKey;
      if (!fits) continue;
      for (int s = i; s < je; ++s) key[klen++] = c_[s].pid;
      const int m = FindWord(key, klen);
      if (m >= 0) {
        match = m;
        jend = je;
        break;
      }
    }

    if (match >= 0 && i >= 1 && jend <= n - 1) {
      // duration target: half the segment before + chunk + half after
      int need = (D_[i - 1] - D_[i - 1] / 2);
      for (int s = i; s < jend; ++s) need += D_[s];
      need += D_[jend] / 2;

      // pick among candidates with the same key (wcost)
      const uint8_t* mk = blob + wrecs[match].key_off;
      int best = match;
      float best_c = 1e30f;
      for (int m2 = match; m2 < static_cast<int>(pk_.h().n_words); ++m2) {
        const uint8_t* k2 = blob + wrecs[m2].key_off;
        if (KeyCompare(k2 + 1, k2[0], mk + 1, mk[0]) != 0) break;
        float c = kWWordDur *
                  fabsf(logf(static_cast<float>(wrecs[m2].n_frames) /
                             static_cast<float>(need > 1 ? need : 1)));
        const float f0m = F0FromCode(wrecs[m2].f0med_q);
        if (prev_tail_f0_ > 0.0f && f0m > 0.0f) {
          c += kWWordF0 * fabsf(logf(f0m / prev_tail_f0_));
        }
        for (int u = 0; u < n_used_; ++u) {
          if (used_words_[u] == m2) c += kWordReuse * used_count_[u];
        }
        if (c < best_c) {
          best_c = c;
          best = m2;
        }
      }
      // reuse bookkeeping
      bool seen = false;
      for (int u = 0; u < n_used_; ++u) {
        if (used_words_[u] == best) {
          ++used_count_[u];
          seen = true;
        }
      }
      if (!seen && n_used_ < kMaxUsedWords) {
        used_words_[n_used_] = best;
        used_count_[n_used_] = 1;
        ++n_used_;
      }
      const float f0m = F0FromCode(wrecs[best].f0med_q);
      if (f0m > 0.0f) prev_tail_f0_ = f0m;

      word_at_[i - 1] = best;
      word_end_[i - 1] = static_cast<int16_t>(jend - 1);
      for (int b = i - 1; b < jend; ++b) covered_[b] = 1;
    }
    i = jend > j ? jend : j;
  }
}

void Engine::SelectDiphones(int n) {
  const int n_bound = n - 1;
  const DiphoneTypeRec* types = pk_.dtypes();
  const DiphoneUnitRec* units = pk_.dunits();
  const uint8_t* cls = pk_.phone_class();

  // -- candidate sets (synth_diphone_world.py CarrierInventory.candidates)
  for (int j = 0; j < n_bound; ++j) {
    n_cands_[j] = 0;
    chosen_[j] = -1;
    if (covered_[j]) continue;
    const int a = c_[j].pid, b = c_[j + 1].pid;
    const int t = FindDiphoneType(a, b);
    if (t >= 0) {
      const int nn = types[t].n_units < kCap ? types[t].n_units : kCap;
      for (int u = 0; u < nn; ++u)
        cands_[j][n_cands_[j]++] =
            static_cast<uint16_t>(types[t].first_unit + u);
      continue;
    }
    // fallback: same second phone, first-phone class match (then any)
    for (int pass = 0; pass < 2 && n_cands_[j] == 0; ++pass) {
      // gather up to kCap best-scoring units across matching types
      for (uint32_t ti = 0; ti < pk_.h().n_diphone_types; ++ti) {
        if (types[ti].b != b) continue;
        if (pass == 0 && cls[types[ti].a] != cls[a]) continue;
        for (int u = 0; u < types[ti].n_units; ++u) {
          const uint16_t cand = static_cast<uint16_t>(
              types[ti].first_unit + u);
          const float sc = units[cand].score;
          // insertion sort by score, capped
          int m = n_cands_[j];
          if (m < kCap) {
            cands_[j][m] = cand;
            ++n_cands_[j];
          } else if (sc < units[cands_[j][kCap - 1]].score) {
            cands_[j][kCap - 1] = cand;
          } else {
            continue;
          }
          for (int q = n_cands_[j] - 1;
               q > 0 && units[cands_[j][q]].score <
                            units[cands_[j][q - 1]].score;
               --q) {
            const uint16_t tmp = cands_[j][q];
            cands_[j][q] = cands_[j][q - 1];
            cands_[j][q - 1] = tmp;
          }
        }
      }
    }
  }

  // -- Viterbi over runs of non-empty candidate sets --------------------
  const float escale = pk_.h().edge_scale;
  float cost[kCap], ncost[kCap];
  uint8_t (*back)[kCap] = reinterpret_cast<uint8_t(*)[kCap]>(
      bump_.AllocArray<uint8_t>(static_cast<size_t>(n) * kCap));
  if (back == nullptr) return;

  auto target_cost = [&](int j, const DiphoneUnitRec& u) -> float {
    const int need = (D_[j] / 2) + (D_[j + 1] - D_[j + 1] / 2);
    const float d_dur = fabsf(
        logf(static_cast<float>(u.n_frames > 1 ? u.n_frames : 1) /
             static_cast<float>(need > 1 ? need : 1)));
    float ctx = 0.0f;
    if (j > 0 && u.prev >= 0 && Canon(u.prev) != c_[j - 1].pid) ctx += 1.0f;
    if (j + 2 < n && u.next >= 0 && Canon(u.next) != c_[j + 2].pid)
      ctx += 1.0f;
    return u.score + kWDur * d_dur + kWCtx * ctx;
  };
  auto join_cost = [&](const DiphoneUnitRec& ua,
                       const DiphoneUnitRec& ub) -> float {
    int acc = 0;
    for (int d = 0; d < kPackEdgeBands; ++d) {
      int diff = static_cast<int>(ua.edge_tail[d]) - ub.edge_head[d];
      acc += diff < 0 ? -diff : diff;
    }
    return kWJoin * escale * acc / kPackEdgeBands;
  };

  int j = 0;
  while (j < n_bound) {
    if (n_cands_[j] == 0) {
      ++j;
      continue;
    }
    const int lo = j;
    while (j < n_bound && n_cands_[j] > 0) ++j;
    const int hi = j;
    // DP
    for (int u = 0; u < n_cands_[lo]; ++u)
      cost[u] = target_cost(lo, units[cands_[lo][u]]);
    for (int b = lo + 1; b < hi; ++b) {
      for (int u = 0; u < n_cands_[b]; ++u) {
        float bestc = 1e30f;
        int bestp = 0;
        for (int p = 0; p < n_cands_[b - 1]; ++p) {
          const float cjoin =
              join_cost(units[cands_[b - 1][p]], units[cands_[b][u]]);
          if (cost[p] + cjoin < bestc) {
            bestc = cost[p] + cjoin;
            bestp = p;
          }
        }
        ncost[u] = bestc + target_cost(b, units[cands_[b][u]]);
        back[b][u] = static_cast<uint8_t>(bestp);
      }
      memcpy(cost, ncost, sizeof(cost));
    }
    int k = 0;
    for (int u = 1; u < n_cands_[hi - 1]; ++u)
      if (cost[u] < cost[k]) k = u;
    for (int b = hi - 1; b > lo; --b) {
      chosen_[b] = cands_[b][k];
      k = back[b][k];
    }
    chosen_[lo] = cands_[lo][k];
  }
}

void Engine::BuildParts(int n) {
  const int n_bound = n - 1;
  const DiphoneUnitRec* units = pk_.dunits();
  const WordUnitRec* wrecs = pk_.wunits();
  const float* pdur = pk_.h().prosody_dur;
  const float* pf0 = pk_.h().prosody_f0;
  const float* pen = pk_.h().prosody_energy;
  const float en_base = pk_.h().energy_base;
  const float median_f0 = pk_.h().median_f0;

  n_parts_ = 0;
  n_joins_ = 0;
  int pos = 0;

  auto add_part = [&](const Part& p) {
    if (n_parts_ < kMaxParts) {
      part_start_[n_parts_] = static_cast<uint16_t>(pos);
      parts_[n_parts_++] = p;
      pos += p.out_frames;
    }
  };

  // leading silence (D[0] - D[0]//2 frames)
  {
    Part p{};
    p.kind = Part::kSilence;
    p.unit = -1;
    p.h2 = static_cast<uint16_t>(D_[0] - D_[0] / 2);
    p.h1 = 0;
    p.out_frames = p.h2;
    p.centroid_type = -1;
    p.f0_scale_a = p.f0_scale_b = 1.0f;
    add_part(p);
  }

  int j = 0;
  while (j < n_bound) {
    if (word_at_[j] >= 0) {
      const int w = word_at_[j];
      const int jend = word_end_[j];
      int need_rule = 0;
      for (int b = j; b <= jend; ++b)
        need_rule += (D_[b] / 2) + (D_[b + 1] - D_[b + 1] / 2);
      int need = BlendLenUnit(need_rule, wrecs[w].n_frames);

      // prosody offsets averaged over the chunk starts the unit covers
      float d_off = 0.0f, f_off = 0.0f, e_off = 0.0f;
      int n_off = 0;
      for (int s = j + 1; s <= jend; ++s) {
        if (seg_chunk_[s] == s) {  // s is a chunk start
          d_off += ProsOff(pdur, s);
          f_off += ProsOff(pf0, s);
          e_off += ProsOff(pen, s) - en_base;
          ++n_off;
        }
      }
      if (n_off > 0) {
        d_off /= n_off;
        f_off /= n_off;
        e_off /= n_off;
        need = static_cast<int>(need * expf(d_off) + 0.5f);
        if (need < 4) need = 4;
      }
      if (need > kMaxChunkFrames / 2) need = kMaxChunkFrames / 2;

      Part p{};
      p.kind = Part::kWord;
      p.unit = w;
      p.out_frames = static_cast<uint16_t>(need);
      p.centroid_type = -1;
      // f0_norm: partial correction toward the inventory median
      const float f0m = F0FromCode(wrecs[w].f0med_q);
      p.f0_scale_a = p.f0_scale_b =
          (f0m > 0.0f ? median_f0 / f0m : 1.0f) * expf(f_off);
      float e = e_off;
      if (e > 0.8f) e = 0.8f;
      if (e < -0.8f) e = -0.8f;
      p.e_off_a = p.e_off_b = e;
      if (n_joins_ < kMaxParts) joins_[n_joins_++] = pos;
      add_part(p);
      if (n_joins_ < kMaxParts)
        joins_[n_joins_++] = static_cast<uint16_t>(pos);
      j = jend + 1;
      continue;
    }

    int h2 = D_[j] / 2;
    int h1 = D_[j + 1] - D_[j + 1] / 2;
    const int u = chosen_[j];
    Part p{};
    p.unit = u;
    p.f0_scale_a = p.f0_scale_b = 1.0f;
    p.e_off_a = p.e_off_b = 0.0f;
    p.centroid_type = -1;
    if (u < 0) {
      p.kind = Part::kSilence;
      p.h2 = static_cast<uint16_t>(h2);
      p.h1 = static_cast<uint16_t>(h1);
    } else {
      const DiphoneUnitRec& ur = units[u];
      // natural durations only for speech-side halves
      if (!IsSil(c_[j].pid) && !IsGap(c_[j].pid))
        h2 = BlendLenUnit(h2, ur.cut);
      if (!IsSil(c_[j + 1].pid) && !IsGap(c_[j + 1].pid))
        h1 = BlendLenUnit(h1, ur.n_frames - ur.cut);
      // prosodic duration offsets per side
      h2 = static_cast<int>(h2 * expf(SegOff(pdur, j)) + 0.5f);
      h1 = static_cast<int>(h1 * expf(SegOff(pdur, j + 1)) + 0.5f);
      if (h2 < 1) h2 = 1;
      if (h1 < 1) h1 = 1;
      p.kind = Part::kDiphone;
      p.h2 = static_cast<uint16_t>(h2);
      p.h1 = static_cast<uint16_t>(h1);
      p.cut = ur.cut;
      p.centroid_type =
          static_cast<int16_t>(FindDiphoneType(c_[j].pid, c_[j + 1].pid));
      // f0: f0_norm plus per-side prosody offset
      const float f0m = F0FromCode(ur.f0med_q);
      const float fnorm = f0m > 0.0f ? median_f0 / f0m : 1.0f;
      p.f0_scale_a = fnorm * expf(SegOff(pf0, j));
      p.f0_scale_b = fnorm * expf(SegOff(pf0, j + 1));
      auto eclamp = [&](int seg) {
        float e = seg_chunk_[seg] >= 0 ? ProsOff(pen, seg_chunk_[seg]) -
                                             en_base
                                       : 0.0f;
        if (e > 0.8f) e = 0.8f;
        if (e < -0.8f) e = -0.8f;
        return e;
      };
      p.e_off_a = eclamp(j);
      p.e_off_b = eclamp(j + 1);
    }
    p.out_frames = static_cast<uint16_t>(p.h2 + p.h1);
    if (n_joins_ < kMaxParts) joins_[n_joins_++] = static_cast<uint16_t>(pos);
    add_part(p);
    ++j;
  }

  // trailing silence
  {
    Part p{};
    p.kind = Part::kSilence;
    p.unit = -1;
    p.h2 = static_cast<uint16_t>(D_[n - 1] / 2);
    p.h1 = 0;
    p.out_frames = p.h2;
    p.centroid_type = -1;
    p.f0_scale_a = p.f0_scale_b = 1.0f;
    add_part(p);
  }
  T_ = pos;
}

void Engine::UnpackCodes(uint32_t codes_off, int n_latents, uint16_t* out) {
  BitReader br(pk_.codes(codes_off));
  const uint32_t* ks = pk_.h().k;
  int bits[kPackStages];
  for (int s = 0; s < kPackStages; ++s) {
    int b = 0;
    while ((1u << b) < ks[s]) ++b;
    bits[s] = b;
  }
  for (int l = 0; l < n_latents; ++l) {
    for (int s = 0; s < kPackStages; ++s) {
      out[l * kPackStages + s] = static_cast<uint16_t>(br.get(bits[s]));
    }
  }
}

const int16_t* Engine::DecodedRows(bool word, int idx, int frame_base,
                                   int* n_frames, float* mean_lnb) {
  const int32_t id = (word ? (1 << 30) : 0) | idx;
  int T = word ? pk_.wunits()[idx].n_frames : pk_.dunits()[idx].n_frames;
  if (T > kMaxUnitFrames) T = kMaxUnitFrames;
  *n_frames = T;
  if (cache_id_ == id) {
    memcpy(mean_lnb, cache_mean_, sizeof(cache_mean_));
    return cache_rows_;
  }

  // this part's span of the concatenated code stream (see SynthesizeChunk)
  NT_CHECKPOINT2(30);
  dec_->ReadRows(frame_base, T, cache_rows_);
  NT_CHECKPOINT2(31);

  // per-band mean natural-log benv (timbre anchor); rows are log10-amp
  // quantized by output_scale
  const float to_ln = pk_.h().output_scale * kLn10;
  for (int b = 0; b < 48; ++b) {
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) acc += cache_rows_[t * 60 + b];
    cache_mean_[b] = acc * to_ln / T;
  }
  memcpy(mean_lnb, cache_mean_, sizeof(cache_mean_));
  cache_id_ = id;
  cache_frames_ = T;
  NT_CHECKPOINT2(33);
  return cache_rows_;
}

// warp positions (synth_diphone_world.py warp / warp_anchored)
static void WarpPositions(int m, int n, float* pos) {
  if (n <= 0) return;
  if (n == 1) {
    pos[0] = 0.0f;
    return;
  }
  const float step = static_cast<float>(m - 1) / (n - 1);
  for (int i = 0; i < n; ++i) pos[i] = step * i;
}

static void WarpAnchoredPositions(int m, int n, bool anchor_end,
                                  float* pos) {
  if (n <= 0) return;
  if (m <= 0) {
    for (int i = 0; i < n; ++i) pos[i] = 0.0f;
    return;
  }
  if (n >= m) {
    WarpPositions(m, n, pos);
    return;
  }
  int keep = n / 2 > 1 ? n / 2 : 1;
  if (keep > m) keep = m;
  const int nw = n - keep;
  if (anchor_end) {
    // steady = [0, m-keep) warped to nw, then transition [m-keep, m) 1:1
    if (nw > 0) WarpPositions(m - keep, nw, pos);
    for (int i = 0; i < keep; ++i)
      pos[nw + i] = static_cast<float>(m - keep + i);
  } else {
    for (int i = 0; i < keep && i < n; ++i) pos[i] = static_cast<float>(i);
    if (nw > 0) {
      WarpPositions(m - keep, nw, pos + keep);
      for (int i = 0; i < nw; ++i) pos[keep + i] += keep;
    }
  }
}

int Engine::BuildRanges(const Part& p, int T, Range ranges[2]) const {
  // two ranges: (a) source [0, cut) -> h2 frames anchored end,
  //             (b) source [cut, T) -> h1 frames anchored start.
  // words are a single plain-warp range.
  if (p.kind == Part::kWord) {
    ranges[0] = {0, T, p.out_frames, false, true, p.f0_scale_a, p.e_off_a};
    return 1;
  }
  int cut = p.cut;
  if (cut > T) cut = T;
  ranges[0] = {0, cut, p.h2, true, false, p.f0_scale_a, p.e_off_a};
  ranges[1] = {cut, T - cut, p.h1, false, false, p.f0_scale_b, p.e_off_b};
  return 2;
}

// f0 prepass: the pitch track comes entirely from the flash-side f0
// streams (never from the TFLM decoder), so the full-utterance F0Pass
// (boxcar + declination + terminal falls, which needs global context)
// can run BEFORE any neural decode. That frees the track materialization
// to run lazily behind the vocoder cursor (EnsureFinal).
void Engine::MaterializeF0() {
  for (int pi = 0; pi < n_parts_; ++pi) {
    const Part& p = parts_[pi];
    int out0 = part_start_[pi];
    if (p.kind == Part::kSilence) {
      for (int t = 0; t < p.out_frames; ++t) f0_[out0 + t] = 0.0f;
      continue;
    }
    int T = p.kind == Part::kWord ? pk_.wunits()[p.unit].n_frames
                                  : pk_.dunits()[p.unit].n_frames;
    if (T > kMaxUnitFrames) T = kMaxUnitFrames;
    const uint32_t f0_off = p.kind == Part::kWord
                                ? pk_.wunits()[p.unit].f0_off
                                : pk_.dunits()[p.unit].f0_off;
    DecodeF0Stream(pk_.f0_stream(f0_off), T, cache_f0_, f0_runs_);

    Range ranges[2];
    const int n_ranges = BuildRanges(p, T, ranges);
    for (int r = 0; r < n_ranges; ++r) {
      const Range& rg = ranges[r];
      if (rg.out_n <= 0) continue;
      if (rg.src_n <= 0) {
        for (int t = 0; t < rg.out_n; ++t) f0_[out0 + t] = 0.0f;
        out0 += rg.out_n;
        continue;
      }
      int out_n = rg.out_n;
      if (out_n > kMaxUnitFrames) out_n = kMaxUnitFrames;
      if (rg.plain)
        WarpPositions(rg.src_n, out_n, posbuf_);
      else
        WarpAnchoredPositions(rg.src_n, out_n, rg.anchor_end, posbuf_);
      for (int t = 0; t < rg.out_n; ++t) {
        const float sp = posbuf_[t < out_n ? t : out_n - 1];
        int s0 = static_cast<int>(sp);
        if (s0 > rg.src_n - 1) s0 = rg.src_n - 1;
        int s1 = s0 + 1 < rg.src_n ? s0 + 1 : s0;
        const float a = sp - s0;
        // f0: nearest-voiced sampling + scale
        const float fa = cache_f0_[rg.src0 + s0];
        const float fb = cache_f0_[rg.src0 + s1];
        float f;
        if (fa > 1.0f && fb > 1.0f)
          f = fa + (fb - fa) * a;
        else
          f = a < 0.5f ? fa : fb;
        f0_[out0 + t] = f > 1.0f ? f * rg.f0s : 0.0f;
      }
      out0 += rg.out_n;
    }
  }
}

// Write the track rows (benv/bap, NOT f0_: that's already final) of one
// part. Parts are materialized strictly in order, matching the decoder's
// monotonic stream window.
void Engine::MaterializePartTrack(int pi) {
  NT_CHECKPOINT2(1000 + pi);  // materializing part pi
  const uint64_t t0us = NowUs();
  const float os = pk_.h().output_scale;
  const float cscale = pk_.h().centroid_scale;
  // silence row: benv 1e-6 (log10 = -6), bap 1.0
  const int16_t sil_benv =
      static_cast<int16_t>(fmaxf(-32768.0f, -6.0f / os));
  const int16_t sil_bap = static_cast<int16_t>(fminf(32767.0f, 1.0f / os));

  const Part& p = parts_[pi];
  int out0 = part_start_[pi];
  if (p.kind == Part::kSilence) {
    for (int t = 0; t < p.out_frames; ++t) {
      int16_t* row = track_[out0 + t];
      for (int b = 0; b < 48; ++b) row[b] = sil_benv;
      for (int b = 48; b < 60; ++b) row[b] = sil_bap;
    }
    stats_->decode_us += static_cast<uint32_t>(NowUs() - t0us);
    return;
  }

  // scratch lives in members (mean_lnb_/timbre_q_): this now runs below
  // the vocoder's render frame, and the stack is a fixed 4 KB bank with
  // the TFLM Invoke path still to come underneath.
  int T;
  const int16_t* rows = DecodedRows(p.kind == Part::kWord, p.unit,
                                    part_base_[pi], &T, mean_lnb_);

  // Per-unit loudness makeup (pack_format.h gain_q): a constant log-amplitude
  // offset that equalizes the pack's per-unit level swing (~30-100% FS on
  // single letters). It is folded into the same benv energy offset as the
  // prosody energy below, so intra-unit dynamics are untouched and diphone
  // boundary steps are still smoothed by the runtime gain-EQ. gain_q == 0
  // (legacy packs / --loudness-norm-strength 0) is a no-op.
  const int8_t unit_gq = (p.kind == Part::kWord)
                             ? pk_.wunits()[p.unit].gain_q
                             : pk_.dunits()[p.unit].gain_q;
  const float unit_eoff = unit_gq * kPackUnitGainStep;  // natural-log amplitude

  // timbre offset (natural log -> track quant units)
  int16_t* timbre_q = timbre_q_;
  memset(timbre_q, 0, sizeof(int16_t) * 48);
  if (p.kind == Part::kDiphone && p.centroid_type >= 0) {
    const int8_t* cen = pk_.centroid(p.centroid_type);
    for (int b = 0; b < 48; ++b) {
      const float off_nat = kTimbreNorm * (cen[b] * cscale - mean_lnb_[b]);
      timbre_q[b] = static_cast<int16_t>(off_nat / kLn10 / os);
    }
  }

  Range ranges[2];
  const int n_ranges = BuildRanges(p, T, ranges);
  for (int r = 0; r < n_ranges; ++r) {
    const Range& rg = ranges[r];
    if (rg.out_n <= 0) continue;
    if (rg.src_n <= 0) {
      // empty source half: silence fill
      for (int t = 0; t < rg.out_n; ++t) {
        int16_t* row = track_[out0 + t];
        for (int b = 0; b < 48; ++b) row[b] = sil_benv;
        for (int b = 48; b < 60; ++b) row[b] = sil_bap;
      }
      out0 += rg.out_n;
      continue;
    }
    int out_n = rg.out_n;
    if (out_n > kMaxUnitFrames) out_n = kMaxUnitFrames;
    if (rg.plain)
      WarpPositions(rg.src_n, out_n, posbuf_);
    else
      WarpAnchoredPositions(rg.src_n, out_n, rg.anchor_end, posbuf_);

    const int16_t eq = static_cast<int16_t>(
        (rg.eoff + unit_eoff) / kLn10 / os);  // energy + per-unit makeup
    for (int t = 0; t < rg.out_n; ++t) {
      const float sp = posbuf_[t < out_n ? t : out_n - 1];
      int s0 = static_cast<int>(sp);
      if (s0 > rg.src_n - 1) s0 = rg.src_n - 1;
      int s1 = s0 + 1 < rg.src_n ? s0 + 1 : s0;
      const float a = sp - s0;
      const int16_t* r0 = rows + (rg.src0 + s0) * 60;
      const int16_t* r1 = rows + (rg.src0 + s1) * 60;
      int16_t* row = track_[out0 + t];
      for (int b = 0; b < 48; ++b) {
        const float v = r0[b] + (r1[b] - r0[b]) * a;
        int q = static_cast<int>(v) + timbre_q[b] + eq;
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        row[b] = static_cast<int16_t>(q);
      }
      for (int b = 48; b < 60; ++b) {
        const float v = r0[b] + (r1[b] - r0[b]) * a;
        row[b] = static_cast<int16_t>(v);
      }
    }
    out0 += rg.out_n;
  }
  stats_->decode_us += static_cast<uint32_t>(NowUs() - t0us);
}

void Engine::GainEqAt(int pi) {
  // equalize_gains(): remove loudness steps at part boundary pi/pi+1
  // (hw=3, ramps over <= 8 frames), in the log domain. Applied strictly
  // in boundary order (== the original full pass).
  const float os = pk_.h().output_scale;
  const float lo_gate = logf(1e-4f);
  const int a_end = part_start_[pi] + parts_[pi].out_frames;
  const int b_start = part_start_[pi + 1];
  const int an = parts_[pi].out_frames;
  const int bn = parts_[pi + 1].out_frames;
  if (an < 1 || bn < 1) return;
  const int ha = an < kXfadeHw ? an : kXfadeHw;
  const int hb = bn < kXfadeHw ? bn : kXfadeHw;
  float ea = 0.0f, eb = 0.0f;
  for (int t = 0; t < ha; ++t) ea += FrameLnEnergy(a_end - 1 - t);
  for (int t = 0; t < hb; ++t) eb += FrameLnEnergy(b_start + t);
  ea /= ha;
  eb /= hb;
  if (ea < lo_gate || eb < lo_gate) return;  // silence-adjacent
  float step = (ea - eb) * 0.5f;
  if (step > 0.6f) step = 0.6f;
  if (step < -0.6f) step = -0.6f;
  const int na = an < 8 ? an : 8;
  const int nb = bn < 8 ? bn : 8;
  for (int i = 0; i < na; ++i) {
    // ramp 0 .. -step over the tail of part a
    const float off = -step * i / (na > 1 ? na - 1 : 1);
    const int16_t q = static_cast<int16_t>(off / kLn10 / os);
    int16_t* row = track_[a_end - na + i];
    for (int b = 0; b < 48; ++b) {
      int v = row[b] + q;
      row[b] = static_cast<int16_t>(v > 32767 ? 32767
                                              : (v < -32768 ? -32768 : v));
    }
  }
  for (int i = 0; i < nb; ++i) {
    const float off = step * (1.0f - static_cast<float>(i) /
                                         (nb > 1 ? nb - 1 : 1));
    const int16_t q = static_cast<int16_t>(off / kLn10 / os);
    int16_t* row = track_[b_start + i];
    for (int b = 0; b < 48; ++b) {
      int v = row[b] + q;
      row[b] = static_cast<int16_t>(v > 32767 ? 32767
                                              : (v < -32768 ? -32768 : v));
    }
  }
}

void Engine::SmoothJoinAt(int j) {
  // smooth_joins_residual(): measure the step across join j and spread it
  // as decaying offsets on both sides. Track is already log-domain, so
  // benv and bap are both plain adds. Applied strictly in join order.
  if (j < 1 || j >= T_) return;
  int16_t d[60];
  for (int b = 0; b < 60; ++b)
    d[b] = static_cast<int16_t>(track_[j][b] - track_[j - 1][b]);
  const int nb_ = j < kXfadeHw ? j : kXfadeHw;
  for (int i = 1; i <= nb_; ++i) {
    // frames j-nb .. j-1 get +ramp*d, ramp rises toward the join to 0.5
    const float ramp = 0.5f * i / nb_;
    int16_t* row = track_[j - 1 - (nb_ - i)];
    for (int b = 0; b < 60; ++b) {
      int v = row[b] + static_cast<int>(ramp * d[b]);
      row[b] = static_cast<int16_t>(v > 32767 ? 32767
                                              : (v < -32768 ? -32768 : v));
    }
  }
  const int mf = (T_ - j) < kXfadeHw ? (T_ - j) : kXfadeHw;
  for (int i = 0; i < mf; ++i) {
    const float ramp = 0.5f * (mf - i) / mf;
    int16_t* row = track_[j + i];
    for (int b = 0; b < 60; ++b) {
      int v = row[b] - static_cast<int>(ramp * d[b]);
      row[b] = static_cast<int16_t>(v > 32767 ? 32767
                                              : (v < -32768 ? -32768 : v));
    }
  }
}

float Engine::FrameLnEnergy(int t) const {
  const float os = pk_.h().output_scale;
  float acc = 0.0f;
  const int16_t* row = track_[t];
  for (int b = 0; b < 48; ++b) acc += Exp10(row[b] * os);
  return logf(fmaxf(acc, 1e-8f));
}

// Interpolate a unit's baked loudness knots at unit fraction u in [0, 1],
// returning LSA (natural-log summed-band amplitude).
static float LoudKnotAt(const int8_t* k, float scale, float u) {
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;
  const float x = u * (kPackLoudKnots - 1);
  int i = static_cast<int>(x);
  if (i > kPackLoudKnots - 2) i = kPackLoudKnots - 2;
  if (i < 0) i = 0;
  const float a = x - i;
  return (k[i] + (k[i + 1] - k[i]) * a) * scale;
}

void Engine::PlanLoudness() {
  // See the kLoud* block at the top of the file. Reconstruct the utterance's
  // loudness contour (LSA per output frame) from the baked per-unit knots plus
  // the exact offsets MaterializePartTrack will apply (gain_q makeup + prosody
  // energy), then lift every too-quiet region toward a floor below the loud
  // reference. Full lookahead, no decode. Output: loud_boost_[t] (log10 amp).
  if (!kLoudLevelEnabled || T_ <= 0 || loud_boost_ == nullptr) return;
  const float lsa_scale = pk_.h().loud_scale;
  const size_t mark = bump_.Mark();
  float* lsa = bump_.AllocArray<float>(T_);
  if (lsa == nullptr) return;

  // 1) assemble the per-frame LSA the synth is about to render.
  for (int t = 0; t < T_; ++t) lsa[t] = kLoudSilentLsa;
  for (int pi = 0; pi < n_parts_; ++pi) {
    const Part& p = parts_[pi];
    const int out0 = part_start_[pi];
    if (p.kind == Part::kSilence) continue;  // stays sentinel
    const int8_t* knots;
    int T;
    float unit_eoff;
    if (p.kind == Part::kWord) {
      const WordUnitRec& w = pk_.wunits()[p.unit];
      knots = w.loud;
      T = w.n_frames;
      unit_eoff = w.gain_q * kPackUnitGainStep;
    } else {
      const DiphoneUnitRec& d = pk_.dunits()[p.unit];
      knots = d.loud;
      T = d.n_frames;
      unit_eoff = d.gain_q * kPackUnitGainStep;
    }
    if (T < 1) T = 1;
    // fill_range: output [o0, o0+n) sampled over unit fraction [u0, u1].
    auto fill_range = [&](int o0, int n, float u0, float u1, float eoff) {
      for (int t = 0; t < n; ++t) {
        const float u = n > 1 ? u0 + (u1 - u0) * t / (n - 1) : u0;
        lsa[o0 + t] = LoudKnotAt(knots, lsa_scale, u) + eoff + unit_eoff;
      }
    };
    if (p.kind == Part::kWord) {
      fill_range(out0, p.out_frames, 0.0f, 1.0f, p.e_off_a);
    } else {
      const float cutf = static_cast<float>(p.cut) / T;  // unit split point
      if (p.h2 > 0) fill_range(out0, p.h2, 0.0f, cutf, p.e_off_a);
      if (p.h1 > 0) fill_range(out0 + p.h2, p.h1, cutf, 1.0f, p.e_off_b);
    }
  }

  // 2) loud reference: max of a short-window mean of the voiced contour.
  float ref = -1e30f;
  for (int t = 0; t < T_; ++t) {
    if (lsa[t] < -100.0f) continue;  // silence
    float acc = 0.0f;
    int cnt = 0;
    for (int d = -kLoudRefHw; d <= kLoudRefHw; ++d) {
      const int tt = t + d;
      if (tt >= 0 && tt < T_ && lsa[tt] > -100.0f) {
        acc += lsa[tt];
        ++cnt;
      }
    }
    if (cnt > 0) {
      const float m = acc / cnt;
      if (m > ref) ref = m;
    }
  }
  if (ref <= -1e29f) {  // fully silent utterance
    bump_.Reset(mark);
    return;
  }

  // 3) raw boost: lift frames between the gate and the floor up to the floor.
  const float floor = ref - kLoudFloorDrop;
  const float gate = ref - kLoudGateDrop;
  for (int t = 0; t < T_; ++t) {
    float b = 0.0f;
    if (lsa[t] > -100.0f && lsa[t] > gate) {
      b = floor - lsa[t];
      if (b < 0.0f) b = 0.0f;
      if (b > kLoudMaxBoost) b = kLoudMaxBoost;
    }
    loud_boost_[t] = b;  // natural-log amplitude for now
  }

  // 4) smooth the boost curve (boxcar) so gain changes never click; reuse the
  // lsa scratch as the smoothing source. Silence frames carry 0, so the boost
  // ramps down gracefully into pauses.
  for (int t = 0; t < T_; ++t) lsa[t] = loud_boost_[t];
  const int hw = kLoudSmoothHw;
  const float inv = 1.0f / (2 * hw + 1);
  auto at = [&](int t) -> float {
    return lsa[t < 0 ? 0 : (t >= T_ ? T_ - 1 : t)];
  };
  float acc = 0.0f;
  for (int t = -hw; t <= hw; ++t) acc += at(t);
  for (int t = 0; t < T_; ++t) {
    loud_boost_[t] = acc * inv / kLn10;  // -> log10 amplitude (decl domain)
    acc += at(t + hw + 1) - at(t - hw);
  }
  bump_.Reset(mark);
}

void Engine::AdvanceJoins() {
  // Frames fully gain-equalized: every unapplied boundary b >= eq_bound_
  // only touches frames >= part_start_[b+1] - 8, and boundaries are in
  // increasing frame order, so the EQ-final prefix ends at the first
  // unapplied boundary's reach (or T_ when all are done).
  const int eq_final = (eq_bound_ + 1 < n_parts_)
                           ? static_cast<int>(part_start_[eq_bound_ + 1]) - 8
                           : T_;
  // A join at frame j reads j-1/j and writes [j-3, j+3); apply it once no
  // pending EQ can touch that range. Joins are in increasing frame order
  // and each is applied exactly once, preserving the sequential
  // all-EQ-then-all-joins semantics (pending EQ reads start at
  // part_start_[b+1]-3 > j+2, so nothing reads a join-written frame
  // before its own EQ ran).
  while (smooth_idx_ < n_joins_ &&
         static_cast<int>(joins_[smooth_idx_]) + kXfadeHw <= eq_final) {
    SmoothJoinAt(joins_[smooth_idx_]);
    ++smooth_idx_;
  }
  // Frames final for the vocoder: fully EQ'd and past every pending join's
  // write range.
  int f = eq_final;
  if (smooth_idx_ < n_joins_) {
    const int jf = static_cast<int>(joins_[smooth_idx_]) - kXfadeHw;
    if (jf < f) f = jf;
  }
  if (mat_part_ >= n_parts_ && eq_bound_ + 1 >= n_parts_ &&
      smooth_idx_ >= n_joins_) {
    f = T_;  // everything applied
  }
  if (f > final_) final_ = f;
}

void Engine::EnsureFinal(int t) {
  while (final_ <= t && mat_part_ < n_parts_) {
    MaterializePartTrack(mat_part_);
    ++mat_part_;
    // gain EQ needs both sides materialized
    while (eq_bound_ + 1 < mat_part_) {
      GainEqAt(eq_bound_);
      ++eq_bound_;
    }
    AdvanceJoins();
  }
  if (mat_part_ >= n_parts_) {
    while (eq_bound_ + 1 < n_parts_) {
      GainEqAt(eq_bound_);
      ++eq_bound_;
    }
    AdvanceJoins();
  }
}

void Engine::F0Pass() {
  // f0_mode="unit_decl": interpolate across unvoiced gaps, 9-frame boxcar,
  // gentle declination ramp, terminal falls at phrase ends; gate by the
  // original voicing.
  const int T = T_;
  if (T <= 0) return;
  const size_t mark = bump_.Mark();
  float* cont = bump_.AllocArray<float>(T);
  float* sm = bump_.AllocArray<float>(T);
  if (cont == nullptr || sm == nullptr) {
    bump_.Reset(mark);
    return;
  }
  int first_v = -1;
  for (int t = 0; t < T && first_v < 0; ++t) {
    if (f0_[t] > 1.0f) first_v = t;
  }
  if (first_v < 0) {
    bump_.Reset(mark);
    return;  // fully unvoiced
  }
  // linear interp across unvoiced spans
  int prev = -1;
  for (int t = 0; t < T; ++t) {
    if (f0_[t] > 1.0f) {
      if (prev < 0) {
        for (int q = 0; q < t; ++q) cont[q] = f0_[t];
      } else if (prev < t - 1) {
        for (int q = prev + 1; q < t; ++q) {
          const float a = static_cast<float>(q - prev) / (t - prev);
          cont[q] = f0_[prev] + (f0_[t] - f0_[prev]) * a;
        }
      }
      cont[t] = f0_[t];
      prev = t;
    }
  }
  for (int t = prev + 1; t < T; ++t) cont[t] = f0_[prev];
  // 9-frame boxcar
  const int w = 4;
  float acc = 0.0f;
  for (int t = -w; t <= w; ++t)
    acc += cont[t < 0 ? 0 : (t >= T ? T - 1 : t)];
  for (int t = 0; t < T; ++t) {
    sm[t] = acc / (2 * w + 1);
    const int add = t + w + 1;
    const int sub = t - w;
    acc += cont[add >= T ? T - 1 : add] - cont[sub < 0 ? 0 : sub];
  }
  // declination ramp + terminal falls
  for (int t = 0; t < T; ++t) {
    const float ramp =
        1.06f + (0.92f - 1.06f) * (T > 1 ? static_cast<float>(t) / (T - 1)
                                         : 0.0f);
    sm[t] *= ramp;
  }
  // terminal falls: last voiced frame of each span followed by > 50
  // unvoiced frames (or utterance end) gets a 0.85 fall over <= 80 frames
  int span_end = -1;
  int gap = 0;
  auto apply_fall = [&](int last) {
    const int n = last + 1 < 80 ? last + 1 : 80;
    for (int i = 0; i < n; ++i) {
      const float fall =
          1.0f + (0.85f - 1.0f) * (n > 1 ? static_cast<float>(i) / (n - 1)
                                         : 1.0f);
      sm[last - n + 1 + i] *= fall;
    }
  };
  for (int t = 0; t < T; ++t) {
    if (f0_[t] > 1.0f) {
      if (span_end >= 0 && gap > 50) apply_fall(span_end);
      span_end = t;
      gap = 0;
    } else {
      ++gap;
    }
  }
  if (span_end >= 0) apply_fall(span_end);
  for (int t = 0; t < T; ++t) f0_[t] = f0_[t] > 1.0f ? sm[t] : 0.0f;
  bump_.Reset(mark);
}

// GetFrame bridge for WorldLiteSynth: convert one track row to linear.
// Pulls the lazy materialization frontier forward first, so decode work
// happens just ahead of the render cursor.
struct RenderCtx {
  Engine* eng;
  const int16_t (*track)[60];
  const float* f0;
  const float* loud_boost;  // per-frame log10-amplitude loudness plan
  float out_scale;
  int T;
};

static void RenderGetFrame(void* user, int t, WorldFrame* frame) {
  const RenderCtx* c = static_cast<const RenderCtx*>(user);
  if (t >= c->T) t = c->T - 1;
  c->eng->EnsureFinal(t);
  const int16_t* row = c->track[t];
  // prosody_shape declination: benv * exp(linspace(0.10, -0.10, T))
  const float decl =
      (0.10f - 0.20f * (c->T > 1 ? static_cast<float>(t) / (c->T - 1)
                                 : 0.0f)) /
      kLn10;
  frame->f0 = c->f0[t];
  const float os = c->out_scale;
  // Planning-time loudness lift (PlanLoudness): a smooth per-frame log10-
  // amplitude offset that raises quiet onsets toward the utterance reference.
  const float lift = c->loud_boost ? c->loud_boost[t] : 0.0f;
  for (int b = 0; b < kWorldNumBenv; ++b)
    frame->benv[b] = Exp10(row[b] * os + decl + lift);
  for (int b = 0; b < kWorldNumBap; ++b) {
    float v = row[48 + b] * os;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    frame->bap[b] = v;
  }
}

int Engine::SynthesizeChunk(int lo, int hi, bool first, bool last,
                            NeuralTts::EmitFn emit, void* user,
                            bool plan_only) {
  const uint64_t t_plan0 = NowUs();
  const size_t chunk_mark = bump_.Mark();
  int n = hi - lo;
  if (n > kMaxRuns - 2) n = kMaxRuns - 2;

  // chunk-local segment list, silences enforced at both ends
  RunSeg* segs = bump_.AllocArray<RunSeg>(n + 2);
  if (segs == nullptr) return -2;
  int cn = 0;
  const uint8_t sil = static_cast<uint8_t>(pk_.h().sil_id);
  if (!IsSil(runs_[lo].pid)) segs[cn++] = {sil, 0};
  for (int i = 0; i < n; ++i) {
    RunSeg s = runs_[lo + i];
    s.pid = static_cast<uint8_t>(Canon(s.pid));
    segs[cn++] = s;
  }
  if (!IsSil(segs[cn - 1].pid)) segs[cn++] = {sil, 0};
  c_ = segs;
  c_n_ = cn;

  // scaled durations D (synthesize(): base * ratio, floors, edge pauses)
  D_ = bump_.AllocArray<uint16_t>(cn);
  seg_chunk_ = bump_.AllocArray<int16_t>(cn);
  chunk_func_ = bump_.AllocArray<int8_t>(cn);
  chunk_pos_ = bump_.AllocArray<int8_t>(cn);
  const int n_bound = cn - 1;
  word_at_ = bump_.AllocArray<int32_t>(n_bound + 1);
  word_end_ = bump_.AllocArray<int16_t>(n_bound + 1);
  covered_ = bump_.AllocArray<uint8_t>(n_bound + 1);
  cands_ = reinterpret_cast<uint16_t(*)[kCap]>(
      bump_.AllocArray<uint16_t>((n_bound + 1) * kCap));
  n_cands_ = bump_.AllocArray<uint8_t>(n_bound + 1);
  chosen_ = bump_.AllocArray<int32_t>(n_bound + 1);
  parts_ = bump_.AllocArray<Part>(kMaxParts);
  part_start_ = bump_.AllocArray<uint16_t>(kMaxParts);
  joins_ = bump_.AllocArray<uint16_t>(kMaxParts);
  if (chosen_ == nullptr || joins_ == nullptr) return -2;

  const float* ratio = pk_.dur_ratio();
  for (int i = 0; i < cn; ++i) {
    const bool s = IsSil(segs[i].pid);
    int base = segs[i].rule_frames;
    if (base < (s ? 4 : 2)) base = s ? 4 : 2;
    int d = static_cast<int>(base * ratio[segs[i].pid] + 0.5f);
    if (d < 2) d = 2;
    D_[i] = static_cast<uint16_t>(d);
  }
  // utterance-edge pauses; chunk-internal edges get a shorter breath. The
  // host reference pads 40 lead / 60 tail frames, but 200 ms of leading
  // silence is pure time-to-first-audio in a voice interface, so the lead
  // is trimmed to 12 (60 ms) -- enough for the vocoder's overlap-add to
  // settle before speech onset.
  D_[0] = static_cast<uint16_t>(D_[0] > (first ? 12 : 16) ? D_[0]
                                                          : (first ? 12 : 16));
  D_[cn - 1] = static_cast<uint16_t>(
      D_[cn - 1] > (last ? 60 : 16) ? D_[cn - 1] : (last ? 60 : 16));

  ComputeProsodyBuckets(cn);
  MatchWords(cn);
  SelectDiphones(cn);
  BuildParts(cn);
  stats_->plan_us += static_cast<uint32_t>(NowUs() - t_plan0);
  ++stats_->chunks;
  if (T_ <= 0) {
    bump_.Reset(chunk_mark);
    return 0;
  }
  if (T_ > kMaxChunkFrames) {
    bump_.Reset(chunk_mark);
    return kChunkTooLong;
  }

  if (plan_only) {
    const int samples = T_ * kWorldFrameSamples;
    bump_.Reset(chunk_mark);
    return samples;
  }

  track_ = reinterpret_cast<int16_t(*)[60]>(
      bump_.AllocArray<int16_t>(static_cast<size_t>(T_) * 60));
  f0_ = bump_.AllocArray<float>(T_);
  // Persistent per-frame loudness boost (log10 amplitude), filled by
  // PlanLoudness before the decoder claims the arena and read at render time.
  loud_boost_ = bump_.AllocArray<float>(T_);
  if (track_ == nullptr || f0_ == nullptr || loud_boost_ == nullptr) {
    bump_.Reset(chunk_mark);
    return -2;
  }
  for (int t = 0; t < T_; ++t) loud_boost_[t] = 0.0f;

  // ---- decode phase (rolled back afterwards) ----
  const size_t decode_mark = bump_.Mark();
  cache_rows_ = bump_.AllocArray<int16_t>(
      static_cast<size_t>(kMaxUnitFrames) * 60);
  cache_f0_ = bump_.AllocArray<float>(kMaxUnitFrames);
  f0_runs_ = bump_.AllocArray<F0RunSpan>(kMaxF0Runs);
  posbuf_ = bump_.AllocArray<float>(kMaxUnitFrames);
  if (cache_f0_ == nullptr || posbuf_ == nullptr) {
    bump_.Reset(chunk_mark);
    return -2;
  }
  cache_id_ = -1;

  // Concatenate every unit part's RVQ codes into ONE stream in part order,
  // so the tiled TFLM decode amortizes across units (each Invoke keeps
  // tile_hop*4 frames) instead of paying >= one full-tile Invoke per unit.
  // Materialize reads each part's span [part_base_[pi], +n_frames) from
  // the stream sequentially, matching the decoder's monotonic window.
  // Side effect: a unit's edge context in the decoder's receptive field is
  // now its actual output neighbor rather than zero padding -- benign, and
  // arguably closer to the encode-after-assembly host reference.
  int total_latents = 0;
  for (int pi = 0; pi < n_parts_; ++pi) {
    const Part& p = parts_[pi];
    if (p.kind == Part::kSilence) continue;
    int T = p.kind == Part::kWord ? pk_.wunits()[p.unit].n_frames
                                  : pk_.dunits()[p.unit].n_frames;
    if (T > kMaxUnitFrames) T = kMaxUnitFrames;
    total_latents += (T + 3) / 4;
  }
  part_base_ = bump_.AllocArray<int32_t>(n_parts_ > 0 ? n_parts_ : 1);
  uint16_t* stream_codes = bump_.AllocArray<uint16_t>(
      static_cast<size_t>(total_latents > 0 ? total_latents : 1) *
      kPackStages);
  if (part_base_ == nullptr || stream_codes == nullptr) {
    bump_.Reset(chunk_mark);
    return -2;
  }
  const uint64_t t_stream0 = NowUs();
  int base_latent = 0;
  for (int pi = 0; pi < n_parts_; ++pi) {
    const Part& p = parts_[pi];
    if (p.kind == Part::kSilence) {
      part_base_[pi] = -1;
      continue;
    }
    uint32_t codes_off;
    int T;
    if (p.kind == Part::kWord) {
      const WordUnitRec& r = pk_.wunits()[p.unit];
      codes_off = r.codes_off;
      T = r.n_frames;
    } else {
      const DiphoneUnitRec& r = pk_.dunits()[p.unit];
      codes_off = r.codes_off;
      T = r.n_frames;
    }
    if (T > kMaxUnitFrames) T = kMaxUnitFrames;
    const int nl = (T + 3) / 4;
    UnpackCodes(codes_off, nl, stream_codes + base_latent * kPackStages);
    part_base_[pi] = base_latent * 4;
    base_latent += nl;
  }
  stream_.n_frames = base_latent * 4;
  stream_.n_latents = base_latent;
  stream_.codes = stream_codes;
  stream_.f0q = nullptr;  // f0 comes from the per-unit side streams
  stats_->stream_us += static_cast<uint32_t>(NowUs() - t_stream0);

  void* dec_obj = bump_.Alloc(sizeof(PbDecoder), alignof(PbDecoder));
  if (cache_rows_ == nullptr || dec_obj == nullptr) {
    bump_.Reset(chunk_mark);
    return -2;
  }
  PbDecoder::Config dc;
  dc.model_data = pk_.model();
  for (int s = 0; s < kPackStages; ++s) {
    dc.codebooks[s] = pk_.codebook(s);
    dc.codebook_scales[s] = pk_.codebook_scale(s);
  }
  dc.n_stages = kPackStages;
  dc.latent_dim = static_cast<int>(pk_.h().latent_dim);
  dc.tile_latents = static_cast<int>(pk_.h().tile_latents);
  dc.tile_hop = static_cast<int>(pk_.h().tile_hop);
  dc.input_scale = pk_.h().input_scale;
  dc.output_scale = pk_.h().output_scale;

  // ---- f0 prepass + whole-track f0 shaping (no neural decode needed) ----
  // Must run BEFORE the decoder claims the rest of the arena: F0Pass takes
  // bump scratch of its own.
  const uint64_t t_post0 = NowUs();
  MaterializeF0();
  NT_CHECKPOINT2(60);
  F0Pass();
  NT_CHECKPOINT2(61);
  // Full-lookahead loudness plan from the baked per-unit contours; like the f0
  // prepass it needs no neural decode, so it runs here before the decoder
  // claims the arena and does not add to time-to-first-audio.
  PlanLoudness();
  NT_CHECKPOINT2(62);
  stats_->post_us += static_cast<uint32_t>(NowUs() - t_post0);

  // The vocoder runs concurrently with the lazy decode (EnsureFinal pulls
  // parts just ahead of the render cursor), so it must be placed before
  // the decoder. Kissfft plans (~21 KiB) and the decoder arena are carved
  // from the shared bump so the malloc heap stays free for CYW43/lwIP.
  const size_t fft_bytes = KissFftrPairBytes(kWorldFftSize);
  void* fft_mem = bump_.Alloc(fft_bytes, 16);
  void* synth_mem = bump_.Alloc(sizeof(WorldLiteSynth), 16);
  uint8_t* dec_arena = static_cast<uint8_t*>(bump_.Alloc(0, 16));
  const size_t dec_bytes = bump_.remaining();
  if (fft_mem == nullptr || synth_mem == nullptr || dec_arena == nullptr ||
      bump_.Alloc(dec_bytes, 1) == nullptr) {
    bump_.Reset(chunk_mark);
    return -2;
  }
  const uint64_t t_alloc0 = NowUs();
  dec_ = new (dec_obj) PbDecoder(dc, dec_arena, dec_bytes);
  stats_->alloc_us += static_cast<uint32_t>(NowUs() - t_alloc0);
  if (!dec_->ok()) {
    bump_.Reset(chunk_mark);
    return -3;
  }
  dec_->BeginUtterance(&stream_);

  // ---- render, decoding lazily behind the cursor ----
  mat_part_ = 0;
  eq_bound_ = 0;
  smooth_idx_ = 0;
  final_ = 0;
  auto* synth = new (synth_mem) WorldLiteSynth(fft_mem, fft_bytes);
  if (!synth->ok()) {
    synth->~WorldLiteSynth();
    bump_.Reset(chunk_mark);
    return -3;
  }
  RenderCtx rc{this, track_, f0_, loud_boost_, pk_.h().output_scale, T_};
  EmitShim shim{emit, user, this};
  // render_us = wall time minus the decode work interleaved into it
  const uint32_t dec_before = stats_->decode_us;
  const uint64_t t_render0 = NowUs();
  synth->Synthesize(RenderGetFrame, &rc, T_, pk_.h().default_gain,
                    &TimedEmit, &shim);
  stats_->render_us += static_cast<uint32_t>(NowUs() - t_render0) -
                       (stats_->decode_us - dec_before);
  stats_->invoke_us += static_cast<uint32_t>(dec_->decode_us());
  stats_->tiles += dec_->tiles_decoded();
  synth->~WorldLiteSynth();
  dec_ = nullptr;
  bump_.Reset(chunk_mark);
  return T_ * kWorldFrameSamples;
}

}  // namespace

// ---------------------------------------------------------------------------

NeuralTts::NeuralTts(const uint8_t* pack, uint8_t* arena, size_t arena_size)
    : pack_(pack), arena_(arena), arena_size_(arena_size) {
  ok_ = pack_.ok() && arena_ != nullptr && arena_size_ >= kMinArenaBytes;
}

int NeuralTts::SynthesizeTokens(void* tokens_vec, EmitFn emit, void* user,
                                bool plan_only) {
  if (!ok_) return -1;
  auto* tokens = static_cast<std::vector<std::string>*>(tokens_vec);
  if (tokens->empty()) return -1;
  Engine eng(pack_, arena_, arena_size_, &stats_);
  return eng.Run(tokens, emit, user, plan_only);
}

int NeuralTts::Synthesize(const char* text, EmitFn emit, void* user) {
  if (text == nullptr) return -1;
  stats_ = {};
  const uint64_t t0 = NowUs();
#if defined(PICO_BUILD)
  g2p::PhoneTokenList phones;
  if (!g2p::TextToPhoneList(text, &phones, nullptr)) return -1;
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  Engine eng(pack_, arena_, arena_size_, &stats_);
  return eng.Run(&phones, emit, user, false);
#else
  std::vector<std::string> tokens = g2p::TextToPhones(std::string(text));
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  return SynthesizeTokens(&tokens, emit, user, false);
#endif
}

int NeuralTts::SynthesizeIpa(const char* ipa, EmitFn emit, void* user) {
  if (ipa == nullptr) return -1;
  stats_ = {};
  const uint64_t t0 = NowUs();
#if defined(PICO_BUILD)
  g2p::PhoneTokenList phones;
  if (!g2p::TokenizeIpaToList(ipa, &phones)) return -1;
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  Engine eng(pack_, arena_, arena_size_, &stats_);
  return eng.Run(&phones, emit, user, false);
#else
  std::vector<std::string> tokens = g2p::TokenizeIpa(std::string(ipa));
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  return SynthesizeTokens(&tokens, emit, user, false);
#endif
}

int NeuralTts::EstimateSamples(const char* text) {
  if (text == nullptr) return -1;
  stats_ = {};
  const uint64_t t0 = NowUs();
#if defined(PICO_BUILD)
  g2p::PhoneTokenList phones;
  if (!g2p::TextToPhoneList(text, &phones, nullptr)) return -1;
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  Engine eng(pack_, arena_, arena_size_, &stats_);
  return eng.Run(&phones, nullptr, nullptr, true);
#else
  std::vector<std::string> tokens = g2p::TextToPhones(std::string(text));
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  return SynthesizeTokens(&tokens, nullptr, nullptr, true);
#endif
}

int NeuralTts::EstimateSamplesIpa(const char* ipa) {
  if (ipa == nullptr) return -1;
  stats_ = {};
  const uint64_t t0 = NowUs();
#if defined(PICO_BUILD)
  g2p::PhoneTokenList phones;
  if (!g2p::TokenizeIpaToList(ipa, &phones)) return -1;
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  Engine eng(pack_, arena_, arena_size_, &stats_);
  return eng.Run(&phones, nullptr, nullptr, true);
#else
  std::vector<std::string> tokens = g2p::TokenizeIpa(std::string(ipa));
  stats_.g2p_us = static_cast<uint32_t>(NowUs() - t0);
  return SynthesizeTokens(&tokens, nullptr, nullptr, true);
#endif
}

}  // namespace neural_tts
