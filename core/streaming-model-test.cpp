// Tests for the streaming model's latency paths: appended cross K/V, draft
// realignment in decode_full, and the cached encoder graph. Run from
// test-assets/ (see scripts/test-core.sh). The published tiny-streaming-en
// carries the cached encoder; tiny-streaming-en-stateless is the last pin with
// the stateless graph, fetched by scripts/fetch-voice-assets.sh so the two can
// be compared. The comparison skips itself when either is absent.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "moonshine-c-api.h"
#include "moonshine-streaming-model.h"
#include "speculative-realign.h"

namespace {

constexpr int kChunk = 1280;
const char *kModel = "tiny-streaming-en";
const char *kStatelessModel = "tiny-streaming-en-stateless";
const char *kCachedModel = kModel;
const char *kWav = "beckett.wav";
// Long enough that the draft diverges on several updates (beckett never
// does). Only the first 16 s are used to keep the test quick.
const char *kDivergingWav = "two_cities_16k.wav";
constexpr size_t kDivergingSamples = 16 * 16000;

std::vector<float> load_wav(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "missing " << path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
  size_t pos = 12;
  const uint8_t *data = nullptr;
  size_t data_len = 0;
  while (pos + 8 <= bytes.size()) {
    uint32_t size = 0;
    std::memcpy(&size, &bytes[pos + 4], 4);
    if (std::memcmp(&bytes[pos], "data", 4) == 0) {
      data = &bytes[pos + 8];
      data_len = std::min<size_t>(size, bytes.size() - pos - 8);
      break;
    }
    pos += 8 + size + (size & 1);
  }
  REQUIRE(data != nullptr);
  std::vector<float> out(data_len / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    int16_t s = 0;
    std::memcpy(&s, data + i * 2, 2);
    out[i] = s / 32768.0f;
  }
  out.resize(out.size() / kChunk * kChunk);
  return out;
}

std::unique_ptr<MoonshineStreamingModel> load_model(const char *dir) {
  auto model = std::make_unique<MoonshineStreamingModel>(false);
  const std::string tokenizer = std::string(dir) + "/tokenizer.bin";
  if (model->load(dir, tokenizer.c_str(),
                  MOONSHINE_MODEL_ARCH_TINY_STREAMING) != 0) {
    return nullptr;
  }
  return model;
}

// Feed audio[from, to) in 80 ms chunks and encode once.
void feed(MoonshineStreamingModel &model, MoonshineStreamingState *state,
          const std::vector<float> &audio, size_t from, size_t to,
          bool is_final) {
  for (size_t p = from; p + kChunk <= to; p += kChunk) {
    REQUIRE(model.process_audio_chunk(state, audio.data() + p, kChunk,
                                      nullptr) == 0);
  }
  int new_frames = 0;
  REQUIRE(model.encode(state, is_final, &new_frames) == 0);
}

std::vector<int> decode(MoonshineStreamingModel &model,
                        MoonshineStreamingState *state,
                        const std::vector<int> &draft) {
  model.decoder_reset(state);
  int *out = nullptr;
  int out_len = 0;
  REQUIRE(model.decode_full(state, draft.empty() ? nullptr : draft.data(),
                            static_cast<int>(draft.size()), &out,
                            &out_len) == 0);
  std::unique_ptr<int, decltype(&std::free)> owned(out, &std::free);
  return std::vector<int>(out, out + out_len);
}

struct Diff {
  double max_abs = 0;
  double max_ref = 0;
  double mean_abs = 0;
};

Diff compare(const std::vector<float> &a, const std::vector<float> &b) {
  REQUIRE(a.size() == b.size());
  Diff d;
  double sum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
    sum += diff;
    d.max_abs = std::max(d.max_abs, diff);
    d.max_ref = std::max(d.max_ref, static_cast<double>(std::fabs(b[i])));
  }
  d.mean_abs = a.empty() ? 0 : sum / a.size();
  return d;
}

// A scripted decoder for the realignment loop: it "wants" to emit ``target``
// and predicts target[n] whenever the n tokens accepted so far are exactly
// target[0, n). The candidate prefix it is handed always is (that is what
// verification guarantees), so predictions are well defined.
struct ScriptedDecoder {
  std::vector<int> target;  // ends with eos
  int eos;
  std::vector<int> emitted;
  int calls = 0;
  int64_t tokens_fed = 0;

  SpeculativeDecodeHooks hooks() {
    SpeculativeDecodeHooks h;
    h.accept = [this](int token) {
      if (token == eos) return false;
      emitted.push_back(token);
      return true;
    };
    h.verify = [this](int /*head*/, const int *cand, int cand_len, int *next) {
      ++calls;
      tokens_fed += cand_len + 1;
      int accepted = 0;
      for (int i = 0; i <= cand_len; ++i) {
        const size_t pos = emitted.size();
        const int pred = pos < target.size() ? target[pos] : eos;
        if (i == cand_len || pred != cand[i]) {
          *next = pred;
          return accepted;
        }
        emitted.push_back(pred);
        ++accepted;
      }
      *next = eos;
      return accepted;
    };
    return h;
  }
};

// Runs the live loop over audio in ``interval`` second updates and returns
// the final memory and the transcript of the last update.
struct StreamRun {
  std::vector<float> memory;
  std::vector<int> tokens;
  int64_t encoder_frames = 0;
};

StreamRun stream(MoonshineStreamingModel &model,
                 const std::vector<float> &audio, float interval_sec,
                 bool final_flush = true) {
  std::unique_ptr<MoonshineStreamingState> state(model.create_state());
  const size_t per_update = std::max<size_t>(
      kChunk, static_cast<size_t>(interval_sec * 16000) / kChunk * kChunk);
  model.reset_stats();
  std::vector<int> tokens;
  for (size_t p = 0; p < audio.size(); p += per_update) {
    const size_t end = std::min(audio.size(), p + per_update);
    const bool is_final = final_flush && end == audio.size();
    feed(model, state.get(), audio, p, end, is_final);
    if (state->memory_len > 0) {
      tokens = decode(model, state.get(), tokens);
    }
  }
  StreamRun run;
  run.memory = state->memory;
  run.tokens = tokens;
  run.encoder_frames = model.stats.encoder_frames;
  return run;
}

}  // namespace

TEST_CASE("decode_full with realignment matches decode_full without") {
  // Same decoder calls in a different grouping; on the tiny model the
  // dynamically quantized decoder is noisy enough that a few updates differ
  // between any two groupings, so this checks the rate rather than equality.
  auto model = load_model(kModel);
  REQUIRE(model != nullptr);
  std::vector<float> audio = load_wav(kDivergingWav);
  audio.resize(std::min(audio.size(), kDivergingSamples) / kChunk * kChunk);
  model->append_cross_kv = false;

  std::unique_ptr<MoonshineStreamingState> state(model->create_state());
  std::vector<int> prev_with, prev_without;
  int updates = 0, differing = 0;
  int64_t calls_with = 0, calls_without = 0;
  const size_t per_update = 6 * kChunk;  // 0.48 s
  for (size_t p = 0; p + kChunk <= audio.size(); p += per_update) {
    const size_t end = std::min(audio.size(), p + per_update);
    feed(*model, state.get(), audio, p, end, end == audio.size());
    if (state->memory_len == 0) continue;
    ++updates;

    model->max_realign_attempts = 3;
    model->reset_stats();
    const std::vector<int> with = decode(*model, state.get(), prev_with);
    calls_with += model->stats.decoder_calls;

    model->max_realign_attempts = 0;
    model->reset_stats();
    const std::vector<int> without = decode(*model, state.get(), prev_without);
    calls_without += model->stats.decoder_calls;

    if (with != without) ++differing;
    prev_with = with;
    prev_without = without;
  }
  MESSAGE("updates=" << updates << " differing=" << differing
                     << " calls with/without=" << calls_with << "/"
                     << calls_without);
  REQUIRE(updates > 10);
  REQUIRE(model->stats.realign_attempts >= 0);
  CHECK(differing <= updates / 5);
  CHECK(calls_with < calls_without);
  // The final transcripts agree.
  CHECK(model->tokens_to_text(
            std::vector<int64_t>(prev_with.begin(), prev_with.end())) ==
        model->tokens_to_text(
            std::vector<int64_t>(prev_without.begin(), prev_without.end())));
}

TEST_CASE("cached encoder graph") {
  for (const char *dir : {kStatelessModel, kCachedModel}) {
    if (!std::filesystem::is_regular_file(std::string(dir) + "/encoder.ort")) {
      MESSAGE("skipping: " << dir << " not present");
      return;
    }
  }
  auto stateless = load_model(kStatelessModel);
  auto cached = load_model(kCachedModel);
  REQUIRE(stateless != nullptr);
  REQUIRE(cached != nullptr);
  CHECK_FALSE(stateless->encoder_has_cache);
  CHECK(cached->encoder_has_cache);
  const std::vector<float> audio = load_wav(kWav);

  SUBCASE("memory matches the stateless encoder, with fewer frames encoded") {
    for (float interval : {0.5f, 0.08f, 2.0f}) {
      CAPTURE(interval);
      const StreamRun ref = stream(*stateless, audio, interval);
      const StreamRun got = stream(*cached, audio, interval);
      REQUIRE(got.memory.size() == ref.memory.size());
      const Diff d = compare(got.memory, ref.memory);
      MESSAGE("interval=" << interval << " max_abs=" << d.max_abs
                          << " mean_abs=" << d.mean_abs << " max_ref="
                          << d.max_ref << " frames " << got.encoder_frames
                          << " vs " << ref.encoder_frames);
      CHECK(d.max_abs < 0.1 * d.max_ref);
      CHECK(d.mean_abs < 0.01 * d.max_ref);
      CHECK(got.encoder_frames < ref.encoder_frames);
      CHECK(stateless->tokens_to_text(
                std::vector<int64_t>(ref.tokens.begin(), ref.tokens.end())) ==
            cached->tokens_to_text(
                std::vector<int64_t>(got.tokens.begin(), got.tokens.end())));
    }
  }

  SUBCASE("cache switched off reproduces the stateless window") {
    cached->use_encoder_cache = false;
    const StreamRun ref = stream(*stateless, audio, 0.5f);
    const StreamRun got = stream(*cached, audio, 0.5f);
    CHECK(got.encoder_frames == ref.encoder_frames);
    const Diff d = compare(got.memory, ref.memory);
    CHECK(d.max_abs < 0.1 * d.max_ref);
    cached->use_encoder_cache = true;
  }

  SUBCASE("reset clears the cache: a reused state matches a fresh one") {
    std::unique_ptr<MoonshineStreamingState> reused(cached->create_state());
    // Run some audio, then reset and run the clip of interest.
    feed(*cached, reused.get(), audio, 0, audio.size() / 2 / kChunk * kChunk,
         false);
    REQUIRE(reused->encoder_cache_len > 0);
    reused->reset(cached->config);
    CHECK(reused->encoder_cache_len == 0);
    CHECK(reused->encoder_cache.empty());

    std::unique_ptr<MoonshineStreamingState> fresh(cached->create_state());
    const size_t per_update = 6 * kChunk;
    for (size_t p = 0; p < audio.size(); p += per_update) {
      const size_t end = std::min(audio.size(), p + per_update);
      feed(*cached, reused.get(), audio, p, end, end == audio.size());
      feed(*cached, fresh.get(), audio, p, end, end == audio.size());
    }
    CHECK(reused->memory == fresh->memory);
  }

  SUBCASE("two interleaved streams do not share cache") {
    const std::vector<float> a(
        audio.begin(), audio.begin() + audio.size() / 2 / kChunk * kChunk);
    std::vector<float> b(audio.begin() + a.size(), audio.end());
    b.resize(b.size() / kChunk * kChunk);
    const StreamRun ref_a = stream(*cached, a, 0.5f);
    const StreamRun ref_b = stream(*cached, b, 0.5f);

    std::unique_ptr<MoonshineStreamingState> sa(cached->create_state());
    std::unique_ptr<MoonshineStreamingState> sb(cached->create_state());
    const size_t per_update = 6 * kChunk;
    for (size_t p = 0; p < std::max(a.size(), b.size()); p += per_update) {
      if (p < a.size()) {
        const size_t end = std::min(a.size(), p + per_update);
        feed(*cached, sa.get(), a, p, end, end == a.size());
      }
      if (p < b.size()) {
        const size_t end = std::min(b.size(), p + per_update);
        feed(*cached, sb.get(), b, p, end, end == b.size());
      }
    }
    const Diff da = compare(sa->memory, ref_a.memory);
    const Diff db = compare(sb->memory, ref_b.memory);
    MESSAGE("a: max_abs=" << da.max_abs << " b: max_abs=" << db.max_abs
                          << " sizes " << sa->memory.size() << "/"
                          << ref_a.memory.size() << " " << sb->memory.size()
                          << "/" << ref_b.memory.size());
    CHECK(sa->memory == ref_a.memory);
    CHECK(sb->memory == ref_b.memory);
  }

  SUBCASE("encoding after a final flush falls back to the full window") {
    // Nothing in the library does this, but a caller holding the state could.
    std::unique_ptr<MoonshineStreamingState> s(cached->create_state());
    const size_t half = audio.size() / 2 / kChunk * kChunk;
    feed(*cached, s.get(), audio, 0, half, true);
    CHECK(s->encoder_cache_len == -1);
    cached->reset_stats();
    feed(*cached, s.get(), audio, half, audio.size(), false);
    // Full left context was fed: more than the new frames plus lookahead.
    CHECK(cached->stats.encoder_frames >
          static_cast<int64_t>((audio.size() - half) / 320) +
              cached->config.total_lookahead);
    // And the cache is trusted again afterwards.
    CHECK(s->encoder_cache_len == 16);
    REQUIRE(s->memory_len > 0);
  }
}

TEST_CASE("realignment reproduces the model's sequence for every edit kind") {
  const int eos = 2;
  // Tokens 10..29 stand in for a 20-token line.
  std::vector<int> base;
  for (int t = 10; t < 30; ++t) base.push_back(t);

  struct Case {
    const char *name;
    std::vector<int> draft;
    std::vector<int> target;
    int max_calls;  // with realignment
  };
  std::vector<int> substituted = base;
  substituted[7] = 99;
  std::vector<int> inserted = base;
  inserted.insert(inserted.begin() + 7, 99);
  std::vector<int> deleted = base;
  deleted.erase(deleted.begin() + 7);
  std::vector<int> two_deleted = base;
  two_deleted.erase(two_deleted.begin() + 7, two_deleted.begin() + 9);
  std::vector<int> two_substituted = base;
  two_substituted[7] = 98;
  two_substituted[8] = 99;
  std::vector<int> extended = base;
  extended.push_back(30);
  extended.push_back(31);
  std::vector<int> truncated(base.begin(), base.end() - 5);
  std::vector<int> garbage(20, 7);

  // Draft is last update's text (base, or a variant); target is what the
  // model produces now.
  const std::vector<Case> cases = {
      {"identical", base, base, 2},
      {"line grew", base, extended, 4},
      {"line shrank", base, truncated, 2},
      {"word respelled", base, substituted, 3},
      {"word inserted", base, inserted, 4},
      {"word dropped", base, deleted, 3},
      {"two words dropped", base, two_deleted, 3},
      {"two words respelled", base, two_substituted, 5},
      {"respelled, from the other side", substituted, base, 3},
      {"inserted, from the other side", inserted, base, 3},
      {"dropped, from the other side", deleted, base, 4},
      {"garbage draft", garbage, base, 20 + 4},
      {"empty draft", {}, base, 21},
  };

  for (const Case &c : cases) {
    CAPTURE(c.name);
    std::vector<int> target = c.target;
    target.push_back(eos);

    ScriptedDecoder with;
    with.target = target;
    with.eos = eos;
    const SpeculativeDecodeOutcome out = run_speculative_decode(
        c.draft.data(), static_cast<int>(c.draft.size()), 1, 3, with.hooks());
    CHECK(out.ok);
    CHECK(with.emitted == c.target);
    CHECK(with.calls <= c.max_calls);
    CHECK(out.realign_attempts <= 3);

    // With realignment off the output is the same and never cheaper.
    ScriptedDecoder without;
    without.target = target;
    without.eos = eos;
    const SpeculativeDecodeOutcome out0 =
        run_speculative_decode(c.draft.data(), static_cast<int>(c.draft.size()),
                               1, 0, without.hooks());
    CHECK(out0.ok);
    CHECK(without.emitted == c.target);
    CHECK(out0.realign_attempts == 0);
    CHECK(with.calls <= without.calls);
  }
}

TEST_CASE("realignment attempts are capped on a draft that keeps missing") {
  // Every other token differs: each realignment attempt accepts at most one
  // token before missing again.
  const int eos = 2;
  std::vector<int> draft, target;
  for (int t = 0; t < 40; ++t) {
    draft.push_back(100 + t);
    target.push_back(t % 2 == 0 ? 100 + t : 500 + t);
  }
  std::vector<int> target_eos = target;
  target_eos.push_back(eos);
  for (int cap : {0, 1, 3, 8}) {
    CAPTURE(cap);
    ScriptedDecoder dec;
    dec.target = target_eos;
    dec.eos = eos;
    const SpeculativeDecodeOutcome out = run_speculative_decode(
        draft.data(), static_cast<int>(draft.size()), 1, cap, dec.hooks());
    CHECK(out.ok);
    CHECK(dec.emitted == target);
    CHECK(out.realign_attempts <= cap);
    // Worst case: first pass over the draft, ``cap`` passes over at most
    // the rest of it, and one single-token call per remaining token.
    const int64_t bound = (1 + 40) + cap * 41 + 41;
    CHECK(dec.tokens_fed <= bound);
  }
}

TEST_CASE("verify errors stop decoding") {
  SpeculativeDecodeHooks hooks;
  hooks.verify = [](int, const int *, int, int *) { return -1; };
  hooks.accept = [](int) { return true; };
  const int draft[] = {5, 6};
  const SpeculativeDecodeOutcome out =
      run_speculative_decode(draft, 2, 1, 3, hooks);
  CHECK_FALSE(out.ok);
}

TEST_CASE("appended cross K/V matches a full projection") {
  auto model = load_model(kModel);
  REQUIRE(model != nullptr);
  const std::vector<float> audio = load_wav(kWav);
  const size_t half = audio.size() / 2 / kChunk * kChunk;

  std::unique_ptr<MoonshineStreamingState> state(model->create_state());
  std::vector<float> logits(model->config.vocab_size);

  model->append_cross_kv = true;
  feed(*model, state.get(), audio, 0, half, false);
  REQUIRE(model->decode_step(state.get(), model->config.bos_id,
                             logits.data()) == 0);
  const int first_len = state->cross_len;
  REQUIRE(first_len > 0);
  feed(*model, state.get(), audio, half, audio.size(), true);
  model->reset_stats();
  model->decoder_reset(state.get());
  REQUIRE(model->decode_step(state.get(), model->config.bos_id,
                             logits.data()) == 0);
  CHECK(state->cross_len == state->memory_len);
  CHECK(state->cross_len > first_len);
  // Only the new frames were projected.
  CHECK(model->stats.cross_kv_frames == state->memory_len - first_len);
  const std::vector<float> k_appended = state->k_cross;
  const std::vector<float> v_appended = state->v_cross;

  // Force a full projection of the same memory.
  state->cross_kv_valid = false;
  state->cross_len = 0;
  state->k_cross.clear();
  state->v_cross.clear();
  model->reset_stats();
  model->decoder_reset(state.get());
  REQUIRE(model->decode_step(state.get(), model->config.bos_id,
                             logits.data()) == 0);
  CHECK(model->stats.cross_kv_frames == state->memory_len);
  REQUIRE(state->k_cross.size() == k_appended.size());

  // The projection is dynamically quantized per call, so the two differ at
  // the quantization-noise level. A repacking mistake (blocks in the wrong
  // order, off by a frame) shows up as errors the size of the values.
  const Diff k = compare(k_appended, state->k_cross);
  const Diff v = compare(v_appended, state->v_cross);
  MESSAGE("k: max_abs=" << k.max_abs << " mean_abs=" << k.mean_abs
                        << " max_ref=" << k.max_ref);
  MESSAGE("v: max_abs=" << v.max_abs << " mean_abs=" << v.mean_abs
                        << " max_ref=" << v.max_ref);
  CHECK(k.max_abs < 0.1 * k.max_ref);
  CHECK(v.max_abs < 0.1 * v.max_ref);
  CHECK(k.mean_abs < 0.01 * k.max_ref);
  CHECK(v.mean_abs < 0.01 * v.max_ref);
}
