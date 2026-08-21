#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "tts-stream.h"

#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

using namespace moonshine_tts;

namespace {

/// Stands in for a real engine: one sample per character, so a test can tell
/// which text produced which audio without loading a model.
class CountingChunkSource : public ChunkSource {
 public:
  void begin(std::string_view text) override {
    text_ = std::string(text);
    remaining_ = chunks_per_utterance_;
    ++begin_count_;
  }
  std::vector<float> next() override {
    if (remaining_ <= 0) {
      return {};
    }
    --remaining_;
    return std::vector<float>(text_.size(), static_cast<float>(text_.size()));
  }
  bool has_more() const override { return remaining_ > 0; }
  int sample_rate() const override { return 24000; }

  void set_chunks_per_utterance(int n) { chunks_per_utterance_ = n; }
  int begin_count() const { return begin_count_; }

 private:
  std::string text_{};
  int chunks_per_utterance_ = 1;
  int remaining_ = 0;
  int begin_count_ = 0;
};

std::unique_ptr<TtsStream> make_stream(CountingChunkSource** out_source,
                                       int chunks_per_utterance = 1) {
  auto source = std::make_unique<CountingChunkSource>();
  source->set_chunks_per_utterance(chunks_per_utterance);
  *out_source = source.get();
  SentenceSplitOptions options;
  options.language = "en";
  return std::make_unique<TtsStream>(std::move(source), options);
}

/// A stand-in two-stage engine. `decode` returns a ramp keyed to the absolute
/// frame index, so a test can check that every sample of the utterance comes
/// out exactly once and in the right order.
class FakeSlicedEngine {
 public:
  explicit FakeSlicedEngine(int frames) : frames_(frames) {}

  SlicedDecodeChunkSource::Prosody analyze(std::string_view text) {
    ++analyze_count_;
    SlicedDecodeChunkSource::Prosody prosody;
    prosody.frames = text.empty() ? 0 : frames_;
    prosody.f0.assign(static_cast<size_t>(prosody.frames), 100.0f);
    prosody.energy.assign(static_cast<size_t>(prosody.frames), 1.0f);
    return prosody;
  }

  std::vector<float> decode(int first, int last) {
    ++decode_count_;
    decoded_frames_ += last - first;
    std::vector<float> out;
    out.reserve(static_cast<size_t>((last - first) * kSamplesPerFrame));
    for (int frame = first; frame < last; ++frame) {
      for (int sample = 0; sample < kSamplesPerFrame; ++sample) {
        out.push_back(static_cast<float>(frame * kSamplesPerFrame + sample));
      }
    }
    return out;
  }

  /// Stands in for the whole-utterance path used when slicing is declined.
  std::vector<float> whole(std::string_view text) {
    ++whole_count_;
    return std::vector<float>(text.size(), 1.0f);
  }

  /// Makes `analyze` decline, the way an over-long utterance would.
  void refuse_to_slice() { frames_ = 0; }

  int analyze_count() const { return analyze_count_; }
  int decode_count() const { return decode_count_; }
  int decoded_frames() const { return decoded_frames_; }
  int whole_count() const { return whole_count_; }
  static constexpr int kSamplesPerFrame = 600;

 private:
  int frames_;
  int analyze_count_ = 0;
  int decode_count_ = 0;
  int decoded_frames_ = 0;
  int whole_count_ = 0;
};

std::unique_ptr<SlicedDecodeChunkSource> make_sliced(FakeSlicedEngine* engine,
                                                     float growth = 2.0f) {
  ChunkPolicyOptions policy;
  policy.first_chunk_seconds = 0.5f;
  policy.tolerance_seconds = 0.0f;
  policy.crossfade_seconds = 0.025f;
  policy.growth = growth;
  policy.pad_frames = 8;
  return std::make_unique<SlicedDecodeChunkSource>(
      [engine](std::string_view text) { return engine->analyze(text); },
      [engine](int first, int last) { return engine->decode(first, last); },
      [engine](std::string_view text) { return engine->whole(text); }, policy,
      FakeSlicedEngine::kSamplesPerFrame, 24000);
}

}  // namespace

TEST_CASE("sliced chunks join up to the whole utterance, sample for sample") {
  const int frames = 400;  // 10 seconds
  FakeSlicedEngine engine(frames);
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  source->begin("anything");

  std::vector<float> joined;
  int chunks = 0;
  while (source->has_more()) {
    const std::vector<float> chunk = source->next();
    joined.insert(joined.end(), chunk.begin(), chunk.end());
    ++chunks;
  }

  CHECK(chunks > 1);
  REQUIRE(joined.size() ==
          static_cast<size_t>(frames) * FakeSlicedEngine::kSamplesPerFrame);

  // Away from the joins every sample must be exactly where it belongs, which
  // is what catches a misplaced, dropped or duplicated span.
  ChunkPolicyOptions policy;
  policy.first_chunk_seconds = 0.5f;
  policy.tolerance_seconds = 0.0f;
  policy.crossfade_seconds = 0.025f;
  policy.growth = 2.0f;
  policy.pad_frames = 8;
  const std::vector<int> boundaries = plan_boundaries({}, frames, 40, policy);
  const size_t crossfade = static_cast<size_t>(0.025f * 24000);

  auto near_a_join = [&](size_t index) {
    for (size_t b = 1; b + 1 < boundaries.size(); ++b) {
      const size_t seam = static_cast<size_t>(boundaries[b]) *
                          FakeSlicedEngine::kSamplesPerFrame;
      if (index + crossfade > seam && index < seam + crossfade) {
        return true;
      }
    }
    return false;
  };

  size_t checked = 0;
  for (size_t index = 0; index < joined.size(); index += 97) {
    const float expected = static_cast<float>(index);
    if (near_a_join(index)) {
      // Sine-law crossfading holds power constant, so two correlated pieces
      // sum to between one and root two times their common value. That bulge
      // is the price of not dipping when they are uncorrelated, which is the
      // case that matters: neighbouring chunks carry different noise.
      CHECK(joined[index] >= expected - 1.0f);
      CHECK(joined[index] <= expected * 1.4143f + 1.0f);
    } else {
      CHECK(joined[index] == doctest::Approx(expected));
      ++checked;
    }
  }
  CHECK(checked > 0);
}

TEST_CASE("prosody runs once per utterance, the decoder once per chunk") {
  FakeSlicedEngine engine(400);
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  source->begin("anything");

  int chunks = 0;
  while (source->has_more()) {
    source->next();
    ++chunks;
  }
  CHECK(engine.analyze_count() == 1);
  CHECK(engine.decode_count() == chunks);
}

TEST_CASE("growing chunks decode less than a uniform grid") {
  FakeSlicedEngine growing(400);
  FakeSlicedEngine uniform(400);
  std::unique_ptr<SlicedDecodeChunkSource> a = make_sliced(&growing, 2.0f);
  std::unique_ptr<SlicedDecodeChunkSource> b = make_sliced(&uniform, 1.0f);
  a->begin("x");
  b->begin("x");
  while (a->has_more()) {
    a->next();
  }
  while (b->has_more()) {
    b->next();
  }
  CHECK(growing.decode_count() < uniform.decode_count());
  // Padding is per chunk, so fewer chunks is also less decoder work.
  CHECK(growing.decoded_frames() < uniform.decoded_frames());
}

TEST_CASE("an utterance shorter than one chunk is a single slice") {
  FakeSlicedEngine engine(10);  // 0.25 s, under the 0.5 s first chunk
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  source->begin("short");

  CHECK(source->has_more());
  const std::vector<float> only = source->next();
  CHECK(only.size() ==
        static_cast<size_t>(10) * FakeSlicedEngine::kSamplesPerFrame);
  CHECK_FALSE(source->has_more());
}

TEST_CASE("an utterance the stages decline is spoken whole, not dropped") {
  FakeSlicedEngine engine(400);
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  engine.refuse_to_slice();
  source->begin("too long to slice");

  REQUIRE(source->has_more());
  const std::vector<float> only = source->next();
  CHECK(only.size() == std::string_view("too long to slice").size());
  CHECK(engine.whole_count() == 1);
  CHECK(engine.decode_count() == 0);
  CHECK_FALSE(source->has_more());
}

TEST_CASE("empty text produces no chunks") {
  FakeSlicedEngine engine(400);
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  source->begin("");
  CHECK_FALSE(source->has_more());
  CHECK(source->next().empty());
}

TEST_CASE("beginning another utterance resets the previous one") {
  FakeSlicedEngine engine(400);
  std::unique_ptr<SlicedDecodeChunkSource> source = make_sliced(&engine);
  source->begin("first");
  source->next();
  CHECK(source->has_more());

  source->begin("second");
  std::vector<float> joined;
  while (source->has_more()) {
    const std::vector<float> chunk = source->next();
    joined.insert(joined.end(), chunk.begin(), chunk.end());
  }
  CHECK(joined.size() ==
        static_cast<size_t>(400) * FakeSlicedEngine::kSamplesPerFrame);
}

TEST_CASE("a fragment is held back until it forms a sentence") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  TtsChunk chunk;

  stream->push_text("Hello ");
  CHECK(stream->next_chunk(chunk) == TtsStream::Status::kNeedText);
  CHECK(source->begin_count() == 0);

  stream->push_text("there. ");
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.text == "Hello there.");
  CHECK(chunk.audio.size() == 12);
  CHECK(chunk.is_final);
  CHECK(chunk.utterance_id == 1);
}

TEST_CASE("pushes are concatenated without inserted whitespace") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  TtsChunk chunk;
  for (const char* piece : {"Hel", "lo wor", "ld."}) {
    stream->push_text(piece);
  }
  stream->end_input();
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.text == "Hello world.");
}

TEST_CASE("flush speaks an unterminated fragment") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  TtsChunk chunk;
  stream->push_text("no terminator");
  CHECK(stream->next_chunk(chunk) == TtsStream::Status::kNeedText);
  stream->flush();
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.text == "no terminator");
}

TEST_CASE("end_input reports end of stream once the queue drains") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  TtsChunk chunk;
  stream->push_text("One. Two. Three.");
  stream->end_input();
  std::vector<std::string> spoken;
  while (stream->next_chunk(chunk) == TtsStream::Status::kChunk) {
    spoken.push_back(chunk.text);
  }
  REQUIRE(spoken.size() == 3);
  CHECK(spoken[0] == "One.");
  CHECK(spoken[2] == "Three.");
  CHECK(stream->next_chunk(chunk) == TtsStream::Status::kEndOfStream);
}

TEST_CASE("only the last chunk of an utterance is final") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source, 3);
  TtsChunk chunk;
  stream->push_text("A sliced sentence.");
  stream->end_input();

  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.text == "A sliced sentence.");
  CHECK_FALSE(chunk.is_final);
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  // Text is attributed once: later chunks cover frames, not characters.
  CHECK(chunk.text.empty());
  CHECK_FALSE(chunk.is_final);
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.is_final);
  CHECK(stream->next_chunk(chunk) == TtsStream::Status::kEndOfStream);
}

TEST_CASE("utterance ids increment across the session") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source, 2);
  TtsChunk chunk;
  stream->push_text("First one. Second one.");
  stream->end_input();
  std::vector<uint64_t> ids;
  while (stream->next_chunk(chunk) == TtsStream::Status::kChunk) {
    ids.push_back(chunk.utterance_id);
  }
  REQUIRE(ids.size() == 4);
  CHECK(ids[0] == 1);
  CHECK(ids[1] == 1);
  CHECK(ids[2] == 2);
  CHECK(ids[3] == 2);
}

TEST_CASE("cancel drops queued work but leaves the stream usable") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  TtsChunk chunk;
  stream->push_text("Drop this. And this. ");
  stream->cancel();
  CHECK(stream->next_chunk(chunk) == TtsStream::Status::kNeedText);
  CHECK(source->begin_count() == 0);
  stream->push_text("Keep this. ");
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.text == "Keep this.");
}

TEST_CASE("nothing is synthesized before a chunk is asked for") {
  CountingChunkSource* source = nullptr;
  std::unique_ptr<TtsStream> stream = make_stream(&source);
  stream->push_text("Several. Complete. Sentences. ");
  CHECK(source->begin_count() == 0);
  TtsChunk chunk;
  REQUIRE(stream->next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(source->begin_count() == 1);
}

TEST_CASE("whole-utterance source returns one chunk per sentence") {
  auto source = std::make_unique<WholeUtteranceChunkSource>(
      [](std::string_view text) {
        return std::vector<float>(text.size(), 1.F);
      },
      24000);
  SentenceSplitOptions options;
  options.language = "en";
  TtsStream stream(std::move(source), options);
  stream.push_text("Alpha. Beta.");
  stream.end_input();
  TtsChunk chunk;
  REQUIRE(stream.next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.audio.size() == 6);
  CHECK(chunk.is_final);
  REQUIRE(stream.next_chunk(chunk) == TtsStream::Status::kChunk);
  CHECK(chunk.audio.size() == 5);
  CHECK(chunk.is_final);
  CHECK(stream.next_chunk(chunk) == TtsStream::Status::kEndOfStream);
}
