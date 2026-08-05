#include "clone-clip.h"

#include <algorithm>
#include <cmath>

CloneClipBounds refine_clone_clip_bounds(
    float window_start_seconds, float requested_duration_seconds,
    const std::vector<CloneClipWord> &words, float max_extension_seconds,
    float end_pad_seconds) {
  CloneClipBounds fallback;
  fallback.start_seconds = window_start_seconds;
  fallback.end_seconds = window_start_seconds + requested_duration_seconds;
  fallback.transcript.clear();

  if (!(requested_duration_seconds > 0.0f) ||
      !(max_extension_seconds >= 0.0f) || !(end_pad_seconds >= 0.0f)) {
    return fallback;
  }

  const float window_end = window_start_seconds + requested_duration_seconds;
  const float hard_end =
      window_start_seconds + requested_duration_seconds + max_extension_seconds;

  std::vector<CloneClipWord> selected;
  selected.reserve(words.size());
  for (const CloneClipWord &word : words) {
    if (word.text.empty()) {
      continue;
    }
    if (!(word.end > word.start)) {
      continue;
    }
    // Words that begin inside the requested window are candidates. A word that
    // started before the window is treated as already clipped at the head and
    // is dropped so the transcript matches audible speech.
    if (word.start >= window_start_seconds && word.start < window_end) {
      selected.push_back(word);
    }
  }

  if (selected.empty()) {
    return fallback;
  }

  // Drop trailing candidates that cannot finish within the extension budget.
  while (!selected.empty()) {
    const float needed_end = selected.back().end + end_pad_seconds;
    if (needed_end <= hard_end + 1e-6f) {
      break;
    }
    selected.pop_back();
  }

  if (selected.empty()) {
    return fallback;
  }

  float start = selected.front().start;
  // If a non-selected word began before ``start`` and still overlaps it, push
  // the clip start forward to that word's end when that keeps the first kept
  // word intact.
  for (const CloneClipWord &word : words) {
    if (word.start < start && word.end > start &&
        word.end < selected.front().end) {
      start = word.end;
    }
  }
  start = std::max(start, window_start_seconds);

  float end = selected.back().end + end_pad_seconds;
  // Resolve any word the tentative end would bisect: extend to finish it when
  // the extension budget allows (overlapping ASR times, or a word that began
  // just past the requested window), otherwise pull the end back to that
  // word's start and drop any selected word that no longer fits.
  for (;;) {
    bool changed = false;
    for (const CloneClipWord &word : words) {
      if (!(word.start + 1e-6f < end && end + 1e-6f < word.end)) {
        continue;
      }
      const float finish = word.end + end_pad_seconds;
      if (finish <= hard_end + 1e-6f) {
        if (finish > end + 1e-6f) {
          end = finish;
          changed = true;
        }
      } else if (word.start < end) {
        end = word.start;
        changed = true;
      }
    }
    end = std::min(end, hard_end);

    while (!selected.empty() && selected.back().end > end + 1e-6f) {
      selected.pop_back();
      changed = true;
    }
    if (selected.empty()) {
      return fallback;
    }
    if (!changed) {
      break;
    }
    // Re-anchor end to the (possibly new) last kept word before looping again.
    end = std::max(end, selected.back().end + end_pad_seconds);
    end = std::min(end, hard_end);
  }

  if (!(end > start)) {
    return fallback;
  }

  // Words fully inside the final bounds, in time order. Include any word we
  // extended to finish even if it began after the requested window end.
  std::vector<CloneClipWord> in_bounds;
  for (const CloneClipWord &word : words) {
    if (word.start + 1e-6f >= start && word.end <= end + 1e-6f) {
      in_bounds.push_back(word);
    }
  }
  std::sort(in_bounds.begin(), in_bounds.end(),
            [](const CloneClipWord &a, const CloneClipWord &b) {
              return a.start < b.start;
            });

  std::string transcript;
  for (const CloneClipWord &word : in_bounds) {
    if (!transcript.empty()) {
      transcript.push_back(' ');
    }
    transcript += word.text;
  }

  CloneClipBounds result;
  result.start_seconds = start;
  result.end_seconds = end;
  result.transcript = std::move(transcript);
  if (result.transcript.empty()) {
    return fallback;
  }
  return result;
}
