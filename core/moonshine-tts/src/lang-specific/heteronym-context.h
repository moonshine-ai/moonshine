#ifndef MOONSHINE_TTS_HETERONYM_CONTEXT_H
#define MOONSHINE_TTS_HETERONYM_CONTEXT_H

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace moonshine_tts {

// Python heteronym_centered_context_window on Unicode code points.
// *full_cells*: UTF-8 text split into one string per code point.
// *span_s*, *span_e*: code point indices [span_s, span_e) into *full_cells*.
// Returns joined window text, and span start/end as code point indices within the window.
std::optional<std::tuple<std::string, int, int>> heteronym_centered_context_window_cells(
    const std::vector<std::string>& full_cells,
    int span_s,
    int span_e,
    int max_chars = 32);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_HETERONYM_CONTEXT_H
