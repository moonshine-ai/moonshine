// Binary layout of the neural-TTS flash pack, mirroring
// scripts/export_neural_tts_pack.py (the writer is the source of truth;
// bump kNeuralTtsPackVersion here AND VERSION there together).
//
// The pack is a single read-only blob (g_neural_tts_pack, memory-mapped
// from flash) holding the s16x8 decoder .tflite, int8 RVQ codebooks, the
// pruned diphone/word unit inventory as bit-packed RVQ code streams +
// f0 side streams, and all selection/prosody metadata. All offsets are
// bytes from the pack start; all multi-byte fields little-endian; record
// arrays 4-byte aligned.

#ifndef NEURAL_TTS_PACK_FORMAT_H_
#define NEURAL_TTS_PACK_FORMAT_H_

#include <cstdint>

namespace neural_tts {

constexpr uint32_t kNeuralTtsPackMagic = 0x3150544E;  // 'NTP1'
constexpr uint32_t kNeuralTtsPackVersion = 3;

constexpr int kPackStages = 3;
constexpr int kPackEdgeBands = 8;  // 48 mel bands mean-pooled by 6
constexpr int kPackBenvBands = 48;

// Per-unit loudness contour: kPackLoudKnots int8 knots sampled uniformly over
// the unit's frames, each holding ln(sum_b benv[t,b]) (natural-log summed-band
// amplitude, "LSA") * (1 / loud_scale). This records the unit's loudness
// *shape* so the synth's planning pass can assemble the whole utterance's
// loudness contour from flash -- without decoding -- and lift quiet onsets
// toward the loud reference with full lookahead (see PlanLoudness).
constexpr int kPackLoudKnots = 8;

// Per-unit loudness makeup: each unit record carries a signed gain_q whose
// value in NATURAL-LOG amplitude units is gain_q * kPackUnitGainStep. The synth
// adds this to the unit's benv (a constant per-band offset in the log-amplitude
// domain, folded into the same path as the prosody energy offset), equalizing
// the pack's per-utterance level swing without buffering. gain_q == 0 (the old
// pad-byte value) is a no-op. int8 range * step ~= +/- 2.0 nat (~+/-17 dB).
constexpr float kPackUnitGainStep = 1.0f / 64.0f;

// f0 codes: hz = 55 * 2^(code / 48); code 0 = unvoiced.
constexpr float kPackF0BaseHz = 55.0f;
constexpr float kPackF0StepsPerOctave = 48.0f;

struct PackHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t total_size;
  float median_f0;
  float default_gain;  // float-PCM -> int16 scale for WorldLiteSynth
  uint32_t n_phones;
  uint32_t sil_id;  // "<sil>"
  uint32_t dot_id;  // "." sentence pause (canonicalized to sil)
  uint32_t gap_id;  // "_" word gap
  uint32_t n_stages;
  uint32_t k[kPackStages];  // RVQ codebook sizes (2048, 1024, 1024)
  uint32_t latent_dim;
  uint32_t tile_latents;  // decoder graph TL
  uint32_t tile_hop;
  float input_scale;   // graph int16 input quantization
  float output_scale;  // graph int16 output quantization
  uint32_t model_off;  // s16x8 .tflite (decoder + fixup fused)
  uint32_t model_size;
  uint32_t cb_off[kPackStages];        // int8 [k][latent_dim]
  uint32_t cb_scale_off[kPackStages];  // float [latent_dim]
  uint32_t n_diphone_types;
  uint32_t dtype_off;  // DiphoneTypeRec[], sorted by (a, b)
  uint32_t n_diphone_units;
  uint32_t dunit_off;  // DiphoneUnitRec[], grouped by type
  uint32_t n_words;
  uint32_t wunit_off;  // WordUnitRec[], sorted by phone-id key
  uint32_t wkeys_off;  // key blob: u8 len, u8 ids[len]
  uint32_t centroid_off;  // int8 [n_diphone_types][48] mean log benv
  float edge_scale;       // int8 edge value * edge_scale = ln(benv)
  float centroid_scale;
  uint32_t codes_off;  // bit-packed RVQ code streams (11+10+10 b/latent)
  uint32_t f0_off;     // f0 side streams (see F0 stream format below)
  uint32_t phones_off;     // u8 [n_phones][8] NUL-padded UTF-8 tokens
  uint32_t dur_ratio_off;  // float [n_phones] natural/rule duration ratio
  uint32_t phone_class_off;  // u8 [n_phones] (0 sil 1 vowel 2 stop
                             //  3 nasal 4 fricative 5 approximant)
  uint32_t n_func_keys;      // function-word phone-id keys (prosody)
  uint32_t func_idx_off;     // u16 [n_func_keys] offsets into func blob
  uint32_t func_blob_off;    // u8 len, u8 ids[len]
  // Prosody offset tables, bucket = [func 0|1][initial, medial, final,
  // single]: log-duration, log-f0, log-energy medians from natural speech.
  float prosody_dur[8];
  float prosody_f0[8];
  float prosody_energy[8];
  float energy_base;  // content-medial energy baseline (offsets relative)
  float loud_scale;   // int8 loudness knot * loud_scale = LSA (natural log)
};
static_assert(sizeof(PackHeader) == 4 * (44 + 26), "layout drift");

struct DiphoneTypeRec {
  uint8_t a, b;      // phone ids
  uint8_t n_units;   // candidates, best-first
  uint8_t pad;
  uint16_t first_unit;  // index into DiphoneUnitRec[]
  uint16_t pad2;
};
static_assert(sizeof(DiphoneTypeRec) == 8, "layout drift");

struct DiphoneUnitRec {
  uint16_t n_frames;
  uint16_t cut;  // phone-boundary frame within the unit
  int8_t prev, next;  // mining context phone ids (-1 = none)
  uint8_t f0med_q;    // f0 code of the unit's median voiced f0
  int8_t gain_q;      // loudness makeup, gain_q * kPackUnitGainStep nat (0=none)
  float score;        // mining score (lower = better)
  uint32_t codes_off;  // bytes into the codes blob (byte-aligned start)
  uint32_t f0_off;     // bytes into the f0 blob
  int8_t edge_head[kPackEdgeBands];  // first-frame pooled ln benv / scale
  int8_t edge_tail[kPackEdgeBands];
  int8_t loud[kPackLoudKnots];       // loudness-shape knots (see kPackLoudKnots)
};
static_assert(sizeof(DiphoneUnitRec) == 44, "layout drift");

struct WordUnitRec {
  uint32_t key_off;  // into wkeys blob
  uint32_t codes_off;
  uint32_t f0_off;
  uint16_t n_frames;
  uint8_t f0med_q;
  int8_t gain_q;  // loudness makeup, gain_q * kPackUnitGainStep nat (0=none)
  int8_t loud[kPackLoudKnots];  // loudness-shape knots (see kPackLoudKnots)
};
static_assert(sizeof(WordUnitRec) == 24, "layout drift");

// F0 stream, per unit:
//   u8 n_runs
//   per run: varu8 gap (unvoiced frames before run; 255 = +255, continue),
//            varu8 len-1
//   per run: u8 first knot code, then (n_knots - 1) signed 4-bit deltas
//            between consecutive decoded codes (low nibble first, byte-
//            padded per run). Knots at run-local frames 0, 4, 8, ... plus
//            the last frame if off-grid; piecewise-linear in code space.

// Read-only view with typed accessors.
class Pack {
 public:
  explicit Pack(const uint8_t* base)
      : base_(base),
        h_(reinterpret_cast<const PackHeader*>(base)) {}

  bool ok() const {
    return h_->magic == kNeuralTtsPackMagic &&
           h_->version == kNeuralTtsPackVersion;
  }
  const PackHeader& h() const { return *h_; }
  const uint8_t* raw(uint32_t off) const { return base_ + off; }

  const unsigned char* model() const { return base_ + h_->model_off; }
  const int8_t* codebook(int s) const {
    return reinterpret_cast<const int8_t*>(base_ + h_->cb_off[s]);
  }
  const float* codebook_scale(int s) const {
    return reinterpret_cast<const float*>(base_ + h_->cb_scale_off[s]);
  }
  const DiphoneTypeRec* dtypes() const {
    return reinterpret_cast<const DiphoneTypeRec*>(base_ + h_->dtype_off);
  }
  const DiphoneUnitRec* dunits() const {
    return reinterpret_cast<const DiphoneUnitRec*>(base_ + h_->dunit_off);
  }
  const WordUnitRec* wunits() const {
    return reinterpret_cast<const WordUnitRec*>(base_ + h_->wunit_off);
  }
  const uint8_t* wkeys() const { return base_ + h_->wkeys_off; }
  const int8_t* centroid(int type_idx) const {
    return reinterpret_cast<const int8_t*>(base_ + h_->centroid_off) +
           type_idx * kPackBenvBands;
  }
  const uint8_t* codes(uint32_t off) const {
    return base_ + h_->codes_off + off;
  }
  const uint8_t* f0_stream(uint32_t off) const {
    return base_ + h_->f0_off + off;
  }
  const char* phone_token(int id) const {
    return reinterpret_cast<const char*>(base_ + h_->phones_off) + id * 8;
  }
  const float* dur_ratio() const {
    return reinterpret_cast<const float*>(base_ + h_->dur_ratio_off);
  }
  const uint8_t* phone_class() const { return base_ + h_->phone_class_off; }
  const uint16_t* func_idx() const {
    return reinterpret_cast<const uint16_t*>(base_ + h_->func_idx_off);
  }
  const uint8_t* func_blob() const { return base_ + h_->func_blob_off; }

 private:
  const uint8_t* base_;
  const PackHeader* h_;
};

}  // namespace neural_tts

#endif  // NEURAL_TTS_PACK_FORMAT_H_
