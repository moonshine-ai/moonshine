#ifndef CLONE_CLIP_H
#define CLONE_CLIP_H

#include <string>
#include <vector>

// Word-aware refinement of a VAD speech-clip window for zero-shot voice
// cloning. ``extract_speech_clip`` stays VAD-only (fast enough for streaming
// capture); this step runs once at clone finalize when word timestamps are
// available, so the clip end can overshoot the requested duration just enough
// to finish the last word, and the transcript only includes complete words.

struct CloneClipWord {
  std::string text;
  float start = 0.0f;
  float end = 0.0f;
};

struct CloneClipBounds {
  float start_seconds = 0.0f;
  float end_seconds = 0.0f;
  std::string transcript;
};

// ``words`` times are absolute in the same timeline as
// ``window_start_seconds``. Words whose start falls in [window_start,
// window_start + requested_duration) are candidates. The result may end after
// the requested duration (up to
// ``max_extension_seconds``) so the last candidate word is fully captured. If
// finishing a candidate would exceed the extension budget, that word is
// dropped. Falls back to the exact VAD window with an empty transcript when no
// usable words remain.
CloneClipBounds refine_clone_clip_bounds(
    float window_start_seconds, float requested_duration_seconds,
    const std::vector<CloneClipWord> &words, float max_extension_seconds = 1.5f,
    float end_pad_seconds = 0.05f);

#endif
