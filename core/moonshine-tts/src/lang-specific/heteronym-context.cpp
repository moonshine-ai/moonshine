#include "heteronym-context.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace moonshine_tts {

namespace {

std::string join_cells(const std::vector<std::string>& cells) {
  std::string out;
  for (const auto& c : cells) {
    out += c;
  }
  return out;
}

}  // namespace

std::optional<std::tuple<std::string, int, int>> heteronym_centered_context_window_cells(
    const std::vector<std::string>& full_cells,
    int span_s,
    int span_e,
    int max_chars) {
  int L = static_cast<int>(full_cells.size());
  int s = span_s;
  int e = span_e;
  if (e > L || s < 0 || s >= e || max_chars < 1) {
    return std::nullopt;
  }
  const int span_w = e - s;
  if (span_w > max_chars) {
    return std::nullopt;
  }

  std::vector<std::string> cells(full_cells.begin(), full_cells.end());
  if (L > max_chars) {
    const int w0_lo = std::max(0, e - max_chars);
    const int w0_hi = std::min(s, L - max_chars);
    if (w0_lo > w0_hi) {
      return std::nullopt;
    }
    const double ideal = (static_cast<double>(s) + e) / 2.0 - static_cast<double>(max_chars) / 2.0;
    int w0 = static_cast<int>(std::llround(ideal));
    w0 = std::max(w0_lo, std::min(w0, w0_hi));
    cells = std::vector<std::string>(cells.begin() + w0, cells.begin() + w0 + max_chars);
    s -= w0;
    e -= w0;
    L = static_cast<int>(cells.size());
  }

  if (L < max_chars) {
    const int total_pad = max_chars - L;
    const double center = (static_cast<double>(s) + e) / 2.0;
    int left_pad = static_cast<int>(std::llround(static_cast<double>(max_chars) / 2.0 - center));
    left_pad = std::max(0, std::min(left_pad, total_pad));
    const int right_pad = total_pad - left_pad;
    std::vector<std::string> padded;
    padded.reserve(static_cast<size_t>(max_chars));
    for (int i = 0; i < left_pad; ++i) {
      padded.push_back(" ");
    }
    padded.insert(padded.end(), cells.begin(), cells.end());
    for (int i = 0; i < right_pad; ++i) {
      padded.push_back(" ");
    }
    cells = std::move(padded);
    s += left_pad;
    e += left_pad;
  }

  return std::make_tuple(join_cells(cells), s, e);
}

}  // namespace moonshine_tts
