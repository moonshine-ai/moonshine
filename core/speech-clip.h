#ifndef SPEECH_CLIP_H
#define SPEECH_CLIP_H

#include <cstddef>
#include <cstdint>
#include <vector>

// Extracts a short, mostly-speech clip from a longer recording, for use as the
// reference clip in zero-shot voice cloning. Clients used to do this
// themselves: run a transcriber in ``skip_transcription`` mode purely for its
// voice-activity detector, slide a window over the resulting segments, and
// slice the best one out of their own copy of the audio. That logic lives here
// now so the JavaScript, Swift and Java bindings all share one implementation.
//
// The clip is a *contiguous* window of the original recording rather than the
// speech segments spliced together, because the cloning vocoder wants natural
// running speech, pauses and all.
//
// This step is VAD-only so streaming capture can call it frequently without a
// model. Word-boundary refinement (extend past the requested duration to finish
// the last word, and emit a matching transcript) lives in
// ``moonshine_extract_speech_clip`` when the TTS synthesizer owns a clone ASR;
// the algorithm is ``refine_clone_clip_bounds`` in clone-clip.h.

struct SpeechClipOptions {
  // Length of the window to extract.
  float clip_duration_seconds = 4.0f;
  // How much of the window has to be speech for it to qualify. Windows are
  // ranked by speech coverage and the best one wins, but a window that never
  // reaches this much speech is not considered good enough to clone from.
  float minimum_speech_seconds = 2.0f;
  // Silero VAD speech probability threshold.
  float vad_threshold = 0.5f;
  // Extra audio to include after the VAD window (clamped to the recording).
  // Used when a follow-on word-timestamp refine step will extend the end to
  // finish the last word. Streaming capture leaves this at zero.
  float tail_pad_seconds = 0.0f;
};

struct SpeechClip {
  // The extracted window, 16 kHz mono. Empty unless ``is_complete``.
  std::vector<float> audio;
  // Where the window starts in the input recording.
  float start_time_seconds = 0.0f;
  // How much of the window is speech.
  float speech_seconds = 0.0f;
  // True when a window meeting ``minimum_speech_seconds`` was found. Callers
  // feeding audio in incrementally use this as the "you can stop talking now"
  // signal.
  bool is_complete = false;
};

// ``sample_rate`` is the rate of ``audio_data``; the returned clip is always
// 16 kHz. Safe to call repeatedly on a growing buffer, which is how the
// streaming capture APIs in the bindings are built.
SpeechClip extract_speech_clip(const float *audio_data, size_t audio_data_size,
                               int32_t sample_rate,
                               const SpeechClipOptions &options);

inline SpeechClip extract_speech_clip(const float *audio_data,
                                      size_t audio_data_size,
                                      int32_t sample_rate) {
  return extract_speech_clip(audio_data, audio_data_size, sample_rate,
                             SpeechClipOptions{});
}

#endif
