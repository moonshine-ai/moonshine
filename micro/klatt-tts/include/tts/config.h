// Externalized, tunable voice parameters.
//
// Every "magic number" in the synthesizer lives here so it can be overridden at
// runtime from a plain-text config file -- no recompile. The compiled-in
// defaults (DefaultVoiceParams) are what the RP2350 firmware uses; the config
// file is a desktop convenience for listening tests.
//
// File format (line-based, '#' comments, whitespace-separated):
//   <global_key> <value>
//   phone <ipa> <field> <value> [<field> <value> ...]
// where phone <field> is one of: f1 f2 f3 b1 b2 b3 dur fnp fnz fric_cf av af ah
// Unknown keys are reported and ignored.

#ifndef TTS_CONFIG_H_
#define TTS_CONFIG_H_

#include <string>
#include <vector>

#include "phonemes.h"

namespace tts {

struct VoiceParams {
  // --- Source gains (only ratios matter; output is peak-normalized) ---
  float voice_gain = 23.49f;
  float fric_gain = 0.578f;
  float asp_gain = 0.295f;
  float fric_q = 1.269f;  // frication band-pass sharpness

  // --- Voice source (LF glottal model + spectral tilt + breathiness) ---
  // lf_rd is Fant's single LF shape parameter: ~0.3 tense/buzzy .. ~2.7
  // lax/breathy. This is the primary naturalness knob; <=0 falls back to the
  // legacy Rosenberg pulse (glottal_open/close below). The LF model is fully
  // implemented; lf_rd <= 0 selects the Rosenberg pulse shipped by default.
  float lf_rd = -1.0f;
  float source_tilt_db =
      0.0f;             // glottal source spectral tilt, dB down at 3 kHz
  float breath = 0.0f;  // open-phase aspiration mixed into voicing, 0..1

  // --- Legacy Rosenberg pulse shape (fractions of a pitch period) ---
  // Only used when lf_rd <= 0.
  float glottal_open = 0.40f;   // rise duration
  float glottal_close = 0.16f;  // fall duration

  // --- Fixed high formants F4/F5/F6 (naturalness, not identity) ---
  // F6 is the higher-pole correction (flattens the upper spectrum). It mainly
  // helps at higher sample rates; at 16 kHz it ships off (f6<=0).
  float f4 = 3500.0f, b4 = 250.0f;
  float f5 = 4500.0f, b5 = 300.0f;
  float f6 = -1.0f, b6 = 500.0f;
  // F0-dependent formant bandwidth widening (per 100 Hz above 100 Hz). 0 = off.
  float bw_f0_coef = 0.0f;

  // --- Voice identity: vocal-tract length & pitch scaling ---
  // Two independent multipliers that shift the apparent speaker without
  // touching the phone table. formant_scale raises every formant / nasal /
  // frication frequency (a shorter vocal tract -> brighter timbre); f0_scale
  // raises the whole pitch contour. Both default to 1.0 (the tuned,
  // male-presenting voice). The CLI --gender/--female flags set them from a
  // single 0..1 control, but they are independently tunable (e.g. raise pitch
  // without changing timbre). A female-presenting voice is roughly
  // formant_scale ~1.18, f0_scale ~1.9.
  float formant_scale =
      1.0f;               // multiplies all formant/nasal/frication frequencies
  float f0_scale = 1.0f;  // multiplies the whole F0 contour

  // --- Streaming output level ---
  // The batch Synthesize() peak-normalizes the finished utterance, but the
  // streaming engine (synth_stream.*) never sees the whole waveform, so instead
  // it applies this fixed makeup gain followed by a soft limiter. Calibrated so
  // the loudest content lands near the batch path's -1 dBFS-ish peak before the
  // limiter; quieter utterances stay proportionally quieter (no per-utterance
  // normalization while streaming). Only used by the streaming path.
  float output_gain = 0.27f;

  // --- Parameter-track smoothing time constants (ms) ---
  float formant_smooth_ms = 21.72f;  // zero-phase, drives coarticulation
  float av_smooth_ms = 6.0f;         // voicing onset/offset
  float af_attack_ms = 16.77f;       // frication attack (slow => voicing leads)
  float af_release_ms = 8.0f;        // frication release
  float ah_smooth_ms = 5.0f;         // aspiration
  float nasal_smooth_ms = 10.0f;     // nasal coupling

  // --- Prosody ---
  float f0_start = 95.33f;      // Hz at utterance onset
  float f0_end = 92.0f;         // Hz at utterance end
  float final_fall_hz = 10.0f;  // extra drop over the last 20%
  // Micro-prosody / intonation (naturalness; break up the monotone).
  float f0_flutter_hz =
      1.82f;               // slow quasi-random F0 drift amplitude (KLSYN88 FL)
  float jitter = 0.0022f;  // cycle-to-cycle period perturbation, fraction
  float shimmer = 0.036f;  // cycle-to-cycle amplitude perturbation, fraction
  float f0_accent_hz =
      9.28f;  // pitch-accent rise on stressed vowels (hat accent)
  float f0_question_rise_hz =
      25.0f;  // final rise for questions (boundary tone)
  // Phrasing: the utterance is split into phrases at sentence pauses (the "."
  // silence phone). Each phrase resets toward a globally-declining top and
  // declines locally, with downstepped accents and a terminal fall/question
  // rise. Declination and downstep default to neutral (off) and are exposed
  // for tuning.
  float f0_declination_hz =
      0.0f;                  // within-phrase F0 fall (top -> end); 0 = off
  float f0_downstep = 1.0f;  // accent height multiplier per accent in a phrase
  // Context-dependent duration. Default to neutral (off); set >1 / <1 to enable.
  float stress_len_scale = 1.0f;      // stressed/accented vowels lengthen
  float unstressed_len_scale = 1.0f;  // unaccented vowels reduce
  float prepausal_len_scale = 1.0f;   // last segment before a pause lengthens

  // --- Segment timing (ms) ---
  // Global duration multiplier (a voice-level speaking rate). >1 = slower.
  // Composes with the CLI --speed flag.
  float duration_scale = 1.336f;
  float lead_ms = 40.0f;  // leading silence
  float tail_ms = 70.0f;  // trailing silence
  float stop_closure_voiced_ms = 61.96f;
  float stop_closure_voiceless_ms = 55.0f;
  float stop_burst_ms = 14.50f;
  float stop_asp_ms = 35.17f;      // voice-onset-time for voiceless stops
  float stop_closure_av = 0.15f;   // voice bar during voiced-stop closure
  float stop_burst_av = 0.20f;     // voicing during voiced-stop burst
  float stop_closure_f1 = 220.0f;  // low closure murmur F1

  // --- Phone inventory (numeric fields overridable; inventory is fixed) ---
  std::vector<Phone> phones;

  const Phone* Lookup(const std::string& ipa) const;
};

// The compiled-in defaults (matches the constants above + DefaultPhoneTable()).
VoiceParams DefaultVoiceParams();

// Apply overrides from a config file on top of `vp`. Returns false on I/O
// failure; unknown keys are printed to stderr and skipped.
bool LoadVoiceConfig(const std::string& path, VoiceParams& vp);

// Write the full current parameter set to `path` in the file format above
// (round-trips through LoadVoiceConfig). Returns false on I/O failure.
bool DumpVoiceConfig(const std::string& path, const VoiceParams& vp);

}  // namespace tts

#endif  // TTS_CONFIG_H_
