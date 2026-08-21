#include "tts-stream.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "utf8-utils.h"

namespace moonshine_tts {

WholeUtteranceChunkSource::WholeUtteranceChunkSource(SynthesizeFn synthesize,
                                                     int sample_rate)
    : synthesize_(std::move(synthesize)), sample_rate_(sample_rate) {}

void WholeUtteranceChunkSource::begin(std::string_view text) {
  text_ = std::string(text);
  pending_ = !text_.empty();
}

std::vector<float> WholeUtteranceChunkSource::next() {
  if (!pending_) {
    return {};
  }
  pending_ = false;
  if (!synthesize_) {
    return {};
  }
  return synthesize_(text_);
}

bool WholeUtteranceChunkSource::has_more() const { return pending_; }

int WholeUtteranceChunkSource::sample_rate() const { return sample_rate_; }

SlicedDecodeChunkSource::SlicedDecodeChunkSource(
    AnalyzeFn analyze, DecodeFn decode, FallbackFn fallback,
    ChunkPolicyOptions policy, int samples_per_frame, int sample_rate)
    : analyze_(std::move(analyze)),
      decode_(std::move(decode)),
      fallback_(std::move(fallback)),
      policy_(policy),
      samples_per_frame_(samples_per_frame),
      sample_rate_(sample_rate) {}

void SlicedDecodeChunkSource::begin(std::string_view text) {
  spans_.clear();
  next_span_ = 0;
  carry_.clear();
  whole_text_.clear();
  if (!analyze_ || !decode_ || text.empty()) {
    return;
  }
  const Prosody prosody = analyze_(text);
  if (prosody.frames <= 0) {
    // The stages declined this utterance. Speak it in one piece rather than
    // dropping it; the caller loses the latency win, not the words.
    if (fallback_) {
      whole_text_ = std::string(text);
    }
    return;
  }
  const int frames_per_second =
      samples_per_frame_ > 0 ? sample_rate_ / samples_per_frame_ : 0;
  const std::vector<float> cost =
      boundary_cost(prosody.f0, prosody.energy, prosody.frames);
  const std::vector<int> boundaries =
      plan_boundaries(cost, prosody.frames, frames_per_second, policy_);
  spans_ = plan_spans(boundaries, prosody.frames, samples_per_frame_,
                      sample_rate_, policy_);
}

std::vector<float> SlicedDecodeChunkSource::next() {
  if (!whole_text_.empty()) {
    const std::string text = std::move(whole_text_);
    whole_text_.clear();
    return fallback_(text);
  }
  if (next_span_ >= spans_.size()) {
    return {};
  }
  const ChunkSpan span = spans_[next_span_];
  ++next_span_;

  std::vector<float> decoded =
      decode_(span.decode_first_frame, span.decode_last_frame);
  // Drop the padding, and the part of the crossfade overhang that belongs to
  // the neighbours, leaving exactly the span this chunk is responsible for.
  const size_t decoded_start = static_cast<size_t>(span.decode_first_frame) *
                               static_cast<size_t>(samples_per_frame_);
  const size_t from = span.keep_first_sample > decoded_start
                          ? span.keep_first_sample - decoded_start
                          : 0;
  const size_t to = span.keep_last_sample > decoded_start
                        ? span.keep_last_sample - decoded_start
                        : 0;
  std::vector<float> piece;
  if (from < decoded.size()) {
    piece.assign(
        decoded.begin() + static_cast<long>(from),
        decoded.begin() + static_cast<long>(std::min(to, decoded.size())));
  }

  // Blend the previous chunk's held-back tail into the front of this one. The
  // result is contiguous with what was already emitted, so a caller can play
  // chunks back to back without knowing a crossfade happened.
  const size_t overlap = carry_.size();
  std::vector<float> joined = std::move(carry_);
  carry_.clear();
  crossfade_append(joined, piece, overlap);

  const bool last = next_span_ >= spans_.size();
  if (last) {
    return joined;
  }
  // Hold back the tail that the next chunk will fade in over.
  const size_t hold = std::min(
      joined.size(),
      static_cast<size_t>(std::max(
          0, static_cast<int>(std::lround(policy_.crossfade_seconds *
                                          static_cast<float>(sample_rate_))))));
  carry_.assign(joined.end() - static_cast<long>(hold), joined.end());
  joined.resize(joined.size() - hold);
  return joined;
}

bool SlicedDecodeChunkSource::has_more() const {
  return !whole_text_.empty() || next_span_ < spans_.size();
}

int SlicedDecodeChunkSource::sample_rate() const { return sample_rate_; }

ExactSliceChunkSource::ExactSliceChunkSource(AnalyzeFn analyze, DecodeFn decode,
                                             FallbackFn fallback,
                                             ChunkPolicyOptions policy,
                                             int frames_per_second,
                                             int sample_rate)
    : analyze_(std::move(analyze)),
      decode_(std::move(decode)),
      fallback_(std::move(fallback)),
      policy_(policy),
      frames_per_second_(frames_per_second),
      sample_rate_(sample_rate) {}

void ExactSliceChunkSource::begin(std::string_view text) {
  boundaries_.clear();
  next_chunk_ = 0;
  whole_text_.clear();
  if (!analyze_ || !decode_ || text.empty()) {
    return;
  }
  const int frames = analyze_(text);
  if (frames <= 0) {
    // The stages declined this utterance. Speak it in one piece rather than
    // dropping it; the caller loses the latency win, not the words.
    if (fallback_) {
      whole_text_ = std::string(text);
    }
    return;
  }
  // No cost curve: with the joins exact there is nothing to hide, so the
  // boundaries stay on the nominal grid. Passing an empty one is how
  // plan_boundaries is told to skip snapping.
  boundaries_ = plan_boundaries({}, frames, frames_per_second_, policy_);
}

std::vector<float> ExactSliceChunkSource::next() {
  if (!whole_text_.empty()) {
    const std::string text = std::move(whole_text_);
    whole_text_.clear();
    return fallback_(text);
  }
  if (next_chunk_ + 1 >= boundaries_.size()) {
    return {};
  }
  const int first = boundaries_[next_chunk_];
  const int last = boundaries_[next_chunk_ + 1];
  ++next_chunk_;
  return decode_(first, last);
}

bool ExactSliceChunkSource::has_more() const {
  return !whole_text_.empty() || next_chunk_ + 1 < boundaries_.size();
}

int ExactSliceChunkSource::sample_rate() const { return sample_rate_; }

TtsStream::TtsStream(std::unique_ptr<ChunkSource> source,
                     SentenceSplitOptions split_options)
    : source_(std::move(source)), split_options_(std::move(split_options)) {}

TtsStream::~TtsStream() = default;

void TtsStream::push_text(std::string_view text) {
  if (text.empty()) {
    return;
  }
  buffer_.append(text);
  SentenceSplit split = split_sentences_incremental(buffer_, split_options_);
  for (std::string& unit : split.units) {
    units_.push_back(std::move(unit));
  }
  buffer_ = std::move(split.remainder);
}

void TtsStream::flush() {
  const std::string tail = trim_ascii_ws_copy(buffer_);
  buffer_.clear();
  if (!tail.empty()) {
    units_.push_back(tail);
  }
}

void TtsStream::end_input() {
  flush();
  input_ended_ = true;
}

void TtsStream::cancel() {
  buffer_.clear();
  units_.clear();
  source_active_ = false;
  current_text_.clear();
}

bool TtsStream::start_next_unit() {
  while (!units_.empty()) {
    current_text_ = std::move(units_.front());
    units_.pop_front();
    if (current_text_.empty()) {
      continue;
    }
    current_utterance_id_ = next_utterance_id_++;
    source_->begin(current_text_);
    source_active_ = true;
    return true;
  }
  return false;
}

TtsStream::Status TtsStream::next_chunk(TtsChunk& out) {
  out = TtsChunk{};
  if (!source_) {
    return Status::kEndOfStream;
  }
  for (;;) {
    if (source_active_) {
      std::vector<float> audio = source_->next();
      if (!audio.empty()) {
        out.audio = std::move(audio);
        out.sample_rate = source_->sample_rate();
        out.utterance_id = current_utterance_id_;
        out.is_final = !source_->has_more();
        // Attribute text on the first chunk only: later chunks of a sliced
        // utterance cover acoustic frames, not a knowable span of characters.
        if (!current_text_.empty()) {
          out.text = std::move(current_text_);
          current_text_.clear();
        }
        if (out.is_final) {
          source_active_ = false;
        }
        return Status::kChunk;
      }
      source_active_ = false;
    }
    if (!start_next_unit()) {
      return input_ended_ ? Status::kEndOfStream : Status::kNeedText;
    }
  }
}

}  // namespace moonshine_tts
