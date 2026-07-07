// SPDX-License-Identifier: MIT
//
// Port of pyannote.core Timeline.support / Annotation.support (MIT-licensed).
// References:
//   - pyannote/core/timeline.py  support_iter / support
//   - pyannote/core/annotation.py  Annotation.support
//   - pyannote/core/segment.py  Segment.__or__ (union), __xor__ (gap), __bool__

#ifndef ANNOTATION_SUPPORT_H_
#define ANNOTATION_SUPPORT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cppannote {

inline constexpr double kSegmentPrecision = 1e-6;

struct Segment {
  double start = 0.;
  double end = 0.;

  bool empty() const { return (end - start) <= kSegmentPrecision; }

  double duration() const {
    const double d = end - start;
    return d > 0. ? d : 0.;
  }
};

// Union (|): covers both segments including any gap between them.
inline Segment segment_union(const Segment& a, const Segment& b) {
  if (a.empty()) {
    return b;
  }
  if (b.empty()) {
    return a;
  }
  return Segment{std::min(a.start, b.start), std::max(a.end, b.end)};
}

// Gap (^): self is first operand, other is second (matches Python `self ^
// other`).
inline Segment segment_gap(const Segment& self_, const Segment& other) {
  if (self_.empty() || other.empty()) {
    throw std::runtime_error(
        "segment_gap: gap with empty segment is undefined");
  }
  return Segment{std::min(self_.end, other.end),
                 std::max(self_.start, other.start)};
}

// Timeline.support_iter / Timeline.support(collar)
// `segments` must be sorted by increasing start (Timeline order).
inline std::vector<Segment> timeline_support_sorted(
    const std::vector<Segment>& segments, double collar) {
  if (segments.empty()) {
    return {};
  }
  std::vector<Segment> segs = segments;
  std::sort(segs.begin(), segs.end(), [](const Segment& x, const Segment& y) {
    return x.start < y.start;
  });

  Segment new_segment = segs[0];
  std::vector<Segment> out;
  for (size_t i = 1; i < segs.size(); ++i) {
    const Segment& segment = segs[i];
    const Segment possible_gap = segment_gap(segment, new_segment);
    if (possible_gap.empty() || possible_gap.duration() < collar) {
      new_segment = segment_union(new_segment, segment);
    } else {
      out.push_back(new_segment);
      new_segment = segment;
    }
  }
  out.push_back(new_segment);
  return out;
}

// Annotation.support(collar): per-label timelines, fill gaps < collar, then
// merge strictly contiguous. Input: map label -> list of segments for that
// label (any order). Output: flat list of (label, segment) with new synthetic
// track order irrelevant to callers.
inline std::vector<std::pair<int, Segment>> annotation_support(
    const std::map<int, std::vector<Segment>>& by_label, double collar) {
  std::vector<std::pair<int, Segment>> flat;
  for (const auto& kv : by_label) {
    const int label = kv.first;
    std::vector<Segment> segs = kv.second;
    if (segs.empty()) {
      continue;
    }
    std::vector<Segment> pass1 = timeline_support_sorted(segs, collar);
    std::vector<Segment> pass2 = timeline_support_sorted(pass1, 0.0);
    for (const Segment& s : pass2) {
      flat.emplace_back(label, s);
    }
  }
  return flat;
}

}  // namespace cppannote

#endif  // ANNOTATION_SUPPORT_H_
