// Embind bridge over the Moonshine C ABI (core/moonshine-c-api.h).
//
// This is deliberately a *thin* low-level bridge: it hides the C ABI's integer
// handles, raw heap pointers, and manual free_* calls behind small C++ classes
// and returns plain JS values (via emscripten::val / value_objects). The
// idiomatic, JS-native surface (Promises, EventTarget, Float32Array ergonomics,
// error classes) lives in the TypeScript layer under wasm/src and is built on
// top of what we register here.
//
// Everything registered here maps 1:1 onto the C ABI so the higher layers can
// stay faithful to the other bindings (Python/Swift/Android).

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "moonshine-c-api.h"

using emscripten::val;

namespace {

// Throw a JS Error carrying the moonshine error code + message so the TS layer
// can map it onto its MoonshineError hierarchy. We throw std::runtime_error;
// embind surfaces .what() to JS. The message is "moonshine:<code>:<text>" so
// the TS side can parse the numeric code back out reliably.
[[noreturn]] void throw_moonshine_error(int32_t code) {
  const char *text = moonshine_error_to_string(code);
  throw std::runtime_error("moonshine:" + std::to_string(code) + ":" +
                           (text ? text : "unknown error"));
}

inline void check(int32_t code) {
  if (code != MOONSHINE_ERROR_NONE) {
    throw_moonshine_error(code);
  }
}

// Copies a JS TypedArray/Array of numbers into a std::vector.
std::vector<float> to_float_vector(const val &array) {
  return emscripten::convertJSArrayToNumberVector<float>(array);
}

std::vector<uint8_t> to_byte_vector(const val &array) {
  return emscripten::convertJSArrayToNumberVector<uint8_t>(array);
}

// ---- Plain-old-data mirrors of the C structs, returned to JS as objects. ----

struct JsWord {
  std::string text;
  float start = 0.0f;
  float end = 0.0f;
  float confidence = 0.0f;
};

struct JsSpeakerSpan {
  float startTime = 0.0f;
  float duration = 0.0f;
  // 64-bit ids are passed as doubles; JS numbers hold them fine up to 2^53,
  // which is plenty for display/keying. The TS layer treats them as opaque.
  double speakerId = 0.0;
  uint32_t speakerIndex = 0;
  double startChar = 0.0;
  double endChar = 0.0;
};

struct JsLine {
  std::string text;
  float startTime = 0.0f;
  float duration = 0.0f;
  double id = 0.0;
  bool isComplete = false;
  bool isUpdated = false;
  bool isNew = false;
  bool hasTextChanged = false;
  bool haveSpeakersChanged = false;
  uint32_t lastTranscriptionLatencyMs = 0;
  std::vector<JsWord> words;
  std::vector<JsSpeakerSpan> speakerSpans;
};

struct JsTranscript {
  std::vector<JsLine> lines;
};

JsTranscript convert_transcript(const transcript_t *t) {
  JsTranscript out;
  if (t == nullptr) {
    return out;
  }
  out.lines.reserve(t->line_count);
  for (uint64_t i = 0; i < t->line_count; ++i) {
    const transcript_line_t &line = t->lines[i];
    JsLine jl;
    jl.text = line.text ? line.text : "";
    jl.startTime = line.start_time;
    jl.duration = line.duration;
    jl.id = static_cast<double>(line.id);
    jl.isComplete = line.is_complete != 0;
    jl.isUpdated = line.is_updated != 0;
    jl.isNew = line.is_new != 0;
    jl.hasTextChanged = line.has_text_changed != 0;
    jl.haveSpeakersChanged = line.have_speakers_changed != 0;
    jl.lastTranscriptionLatencyMs = line.last_transcription_latency_ms;
    for (uint64_t w = 0; w < line.word_count; ++w) {
      const transcript_word_t &word = line.words[w];
      jl.words.push_back(JsWord{word.text ? word.text : "", word.start,
                                word.end, word.confidence});
    }
    for (uint64_t s = 0; s < line.speaker_span_count; ++s) {
      const speaker_span_t &span = line.speaker_spans[s];
      jl.speakerSpans.push_back(JsSpeakerSpan{
          span.start_time, span.duration, static_cast<double>(span.speaker_id),
          span.speaker_index, static_cast<double>(span.start_char),
          static_cast<double>(span.end_char)});
    }
    out.lines.push_back(std::move(jl));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Transcriber / Stream
// ---------------------------------------------------------------------------

class Stream;

class Transcriber {
 public:
  // Loads a transcriber from in-memory model bytes. The buffers are kept alive
  // for the transcriber's lifetime (the C ABI does not copy the spelling model
  // and treats the others as borrowed for the load).
  Transcriber(val encoder, val decoder, val tokenizer, val spelling,
              uint32_t model_arch)
      : encoder_(to_byte_vector(encoder)),
        decoder_(to_byte_vector(decoder)),
        tokenizer_(to_byte_vector(tokenizer)),
        spelling_(spelling.isUndefined() || spelling.isNull()
                      ? std::vector<uint8_t>{}
                      : to_byte_vector(spelling)) {
    handle_ = moonshine_load_transcriber_from_memory(
        encoder_.data(), encoder_.size(), decoder_.data(), decoder_.size(),
        tokenizer_.data(), tokenizer_.size(),
        spelling_.empty() ? nullptr : spelling_.data(), spelling_.size(),
        model_arch, nullptr, 0, MOONSHINE_HEADER_VERSION);
    if (handle_ < 0) {
      throw_moonshine_error(handle_);
    }
  }

  ~Transcriber() { close(); }

  Transcriber(const Transcriber &) = delete;
  Transcriber &operator=(const Transcriber &) = delete;

  JsTranscript transcribe(val audio, int32_t sample_rate, uint32_t flags) {
    std::vector<float> pcm = to_float_vector(audio);
    transcript_t *transcript = nullptr;
    check(moonshine_transcribe_without_streaming(
        handle_, pcm.data(), pcm.size(), sample_rate, flags, &transcript));
    return convert_transcript(transcript);
  }

  // Stream lifecycle is exposed as a Stream object (see below).
  int32_t createStreamHandle(uint32_t flags) {
    int32_t s = moonshine_create_stream(handle_, flags);
    if (s < 0) {
      throw_moonshine_error(s);
    }
    return s;
  }

  int32_t handle() const { return handle_; }

  void close() {
    if (handle_ >= 0) {
      moonshine_free_transcriber(handle_);
      handle_ = -1;
    }
  }

 private:
  std::vector<uint8_t> encoder_;
  std::vector<uint8_t> decoder_;
  std::vector<uint8_t> tokenizer_;
  std::vector<uint8_t> spelling_;
  int32_t handle_ = -1;
};

class Stream {
 public:
  Stream(Transcriber &transcriber, uint32_t flags)
      : transcriber_handle_(transcriber.handle()),
        stream_handle_(transcriber.createStreamHandle(flags)) {}

  ~Stream() { close(); }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  void start() { check(moonshine_start_stream(transcriber_handle_, stream_handle_)); }
  void stop() { check(moonshine_stop_stream(transcriber_handle_, stream_handle_)); }

  void addAudio(val audio, int32_t sample_rate, uint32_t flags) {
    std::vector<float> pcm = to_float_vector(audio);
    check(moonshine_transcribe_add_audio_to_stream(
        transcriber_handle_, stream_handle_, pcm.data(), pcm.size(),
        sample_rate, flags));
  }

  JsTranscript transcribe(uint32_t flags) {
    transcript_t *transcript = nullptr;
    check(moonshine_transcribe_stream(transcriber_handle_, stream_handle_,
                                      flags, &transcript));
    return convert_transcript(transcript);
  }

  void close() {
    if (stream_handle_ >= 0) {
      moonshine_free_stream(transcriber_handle_, stream_handle_);
      stream_handle_ = -1;
    }
  }

 private:
  int32_t transcriber_handle_ = -1;
  int32_t stream_handle_ = -1;
};

// ---------------------------------------------------------------------------
// Intent recognizer (always compiled into the core)
// ---------------------------------------------------------------------------

struct JsIntentMatch {
  std::string canonicalPhrase;
  float similarity = 0.0f;
};

class IntentRecognizer {
 public:
  IntentRecognizer(const std::string &model_path, uint32_t model_arch,
                   const std::string &model_variant) {
    handle_ = moonshine_create_intent_recognizer(
        model_path.c_str(), model_arch,
        model_variant.empty() ? nullptr : model_variant.c_str());
    if (handle_ < 0) {
      throw_moonshine_error(handle_);
    }
  }

  ~IntentRecognizer() { close(); }
  IntentRecognizer(const IntentRecognizer &) = delete;
  IntentRecognizer &operator=(const IntentRecognizer &) = delete;

  void registerIntent(const std::string &phrase, int32_t priority) {
    check(moonshine_register_intent(handle_, phrase.c_str(), nullptr, 0,
                                    priority));
  }

  void unregisterIntent(const std::string &phrase) {
    check(moonshine_unregister_intent(handle_, phrase.c_str()));
  }

  void clearIntents() { check(moonshine_clear_intents(handle_)); }

  std::vector<JsIntentMatch> closestIntents(const std::string &utterance,
                                            float threshold) {
    moonshine_intent_match_t *matches = nullptr;
    uint64_t count = 0;
    check(moonshine_get_closest_intents(handle_, utterance.c_str(), threshold,
                                        &matches, &count));
    std::vector<JsIntentMatch> out;
    out.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
      out.push_back(JsIntentMatch{
          matches[i].canonical_phrase ? matches[i].canonical_phrase : "",
          matches[i].similarity});
    }
    moonshine_free_intent_matches(matches, count);
    return out;
  }

  void close() {
    if (handle_ >= 0) {
      moonshine_free_intent_recognizer(handle_);
      handle_ = -1;
    }
  }

 private:
  int32_t handle_ = -1;
};

#if defined(MOONSHINE_C_API_MOONSHINE_TTS) && MOONSHINE_C_API_MOONSHINE_TTS

// ---------------------------------------------------------------------------
// Text to speech + grapheme-to-phoneme (only when TTS is compiled in)
// ---------------------------------------------------------------------------

struct JsTtsResult {
  val audio = val::undefined();  // Float32Array
  int32_t sampleRate = 0;
};

class TextToSpeech {
 public:
  // Assets are supplied in memory keyed by canonical filename (see
  // moonshine_create_tts_synthesizer_from_memory). `keys` is an array of
  // strings and `buffers` an array of Uint8Arrays of matching length.
  TextToSpeech(const std::string &language, val keys, val buffers) {
    const size_t count = keys["length"].as<size_t>();
    key_strings_.reserve(count);
    buffers_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      key_strings_.push_back(keys[i].as<std::string>());
      buffers_.push_back(to_byte_vector(buffers[i]));
    }
    std::vector<const char *> key_ptrs;
    std::vector<const uint8_t *> buf_ptrs;
    std::vector<uint64_t> buf_sizes;
    key_ptrs.reserve(count);
    buf_ptrs.reserve(count);
    buf_sizes.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      key_ptrs.push_back(key_strings_[i].c_str());
      buf_ptrs.push_back(buffers_[i].data());
      buf_sizes.push_back(buffers_[i].size());
    }
    handle_ = moonshine_create_tts_synthesizer_from_memory(
        language.c_str(), key_ptrs.data(), count, buf_ptrs.data(),
        buf_sizes.data(), nullptr, 0, MOONSHINE_HEADER_VERSION);
    if (handle_ < 0) {
      throw_moonshine_error(handle_);
    }
  }

  ~TextToSpeech() { close(); }
  TextToSpeech(const TextToSpeech &) = delete;
  TextToSpeech &operator=(const TextToSpeech &) = delete;

  JsTtsResult say(const std::string &text) {
    float *audio = nullptr;
    uint64_t size = 0;
    int32_t sample_rate = 0;
    check(moonshine_text_to_speech(handle_, text.c_str(), nullptr, 0, &audio,
                                   &size, &sample_rate));
    // Copy into a JS Float32Array before freeing the C buffer.
    val result = val::global("Float32Array").new_(static_cast<double>(size));
    val heap = val(emscripten::typed_memory_view(size, audio));
    result.call<void>("set", heap);
    free(audio);
    return JsTtsResult{result, sample_rate};
  }

  void close() {
    if (handle_ >= 0) {
      moonshine_free_tts_synthesizer(handle_);
      handle_ = -1;
    }
  }

 private:
  std::vector<std::string> key_strings_;
  std::vector<std::vector<uint8_t>> buffers_;
  int32_t handle_ = -1;
};

class GraphemeToPhonemizer {
 public:
  GraphemeToPhonemizer(const std::string &language, val keys, val buffers) {
    const size_t count = keys["length"].as<size_t>();
    key_strings_.reserve(count);
    buffers_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      key_strings_.push_back(keys[i].as<std::string>());
      buffers_.push_back(to_byte_vector(buffers[i]));
    }
    std::vector<const char *> key_ptrs;
    std::vector<const uint8_t *> buf_ptrs;
    std::vector<uint64_t> buf_sizes;
    for (size_t i = 0; i < count; ++i) {
      key_ptrs.push_back(key_strings_[i].c_str());
      buf_ptrs.push_back(buffers_[i].data());
      buf_sizes.push_back(buffers_[i].size());
    }
    handle_ = moonshine_create_grapheme_to_phonemizer_from_memory(
        language.c_str(), key_ptrs.data(), count, buf_ptrs.data(),
        buf_sizes.data(), nullptr, 0, MOONSHINE_HEADER_VERSION);
    if (handle_ < 0) {
      throw_moonshine_error(handle_);
    }
  }

  ~GraphemeToPhonemizer() { close(); }
  GraphemeToPhonemizer(const GraphemeToPhonemizer &) = delete;
  GraphemeToPhonemizer &operator=(const GraphemeToPhonemizer &) = delete;

  std::string textToPhonemes(const std::string &text) {
    const char *phonemes = nullptr;
    uint64_t count = 0;
    check(moonshine_text_to_phonemes(handle_, text.c_str(), nullptr, 0,
                                     &phonemes, &count));
    return phonemes ? std::string(phonemes, count) : std::string();
  }

  void close() {
    if (handle_ >= 0) {
      moonshine_free_grapheme_to_phonemizer(handle_);
      handle_ = -1;
    }
  }

 private:
  std::vector<std::string> key_strings_;
  std::vector<std::vector<uint8_t>> buffers_;
  int32_t handle_ = -1;
};

#endif  // MOONSHINE_C_API_MOONSHINE_TTS

// ---------------------------------------------------------------------------
// Free functions: version + JSON manifest helpers (drive the AssetDownloader).
// ---------------------------------------------------------------------------

int32_t version() { return moonshine_get_version(); }

std::string stt_dependencies(const std::string &language,
                             const std::string &model_arch) {
  std::vector<moonshine_option_t> options;
  if (!model_arch.empty()) {
    options.push_back(moonshine_option_t{"model_arch", model_arch.c_str()});
  }
  char *json = nullptr;
  check(moonshine_get_stt_dependencies(
      language.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

std::string intent_dependencies(const std::string &model_name,
                                const std::string &variant) {
  std::vector<moonshine_option_t> options;
  if (!variant.empty()) {
    options.push_back(moonshine_option_t{"variant", variant.c_str()});
  }
  char *json = nullptr;
  check(moonshine_get_intent_dependencies(
      model_name.empty() ? nullptr : model_name.c_str(),
      options.empty() ? nullptr : options.data(), options.size(), &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

#if defined(MOONSHINE_C_API_MOONSHINE_TTS) && MOONSHINE_C_API_MOONSHINE_TTS
std::string tts_dependencies(const std::string &languages,
                             const std::string &voice) {
  std::vector<moonshine_option_t> options;
  if (!voice.empty()) {
    options.push_back(moonshine_option_t{"voice", voice.c_str()});
  }
  char *json = nullptr;
  check(moonshine_get_tts_dependencies(
      languages.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

std::string tts_voices(const std::string &languages) {
  char *json = nullptr;
  check(moonshine_get_tts_voices(languages.c_str(), nullptr, 0, &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

std::string g2p_dependencies(const std::string &languages) {
  char *json = nullptr;
  check(moonshine_get_g2p_dependencies(languages.c_str(), nullptr, 0, &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}
#endif

}  // namespace

EMSCRIPTEN_BINDINGS(moonshine) {
  using namespace emscripten;

  register_vector<JsWord>("MoonshineWordVector");
  register_vector<JsSpeakerSpan>("MoonshineSpeakerSpanVector");
  register_vector<JsLine>("MoonshineLineVector");
  register_vector<JsIntentMatch>("MoonshineIntentMatchVector");

  value_object<JsWord>("MoonshineWord")
      .field("text", &JsWord::text)
      .field("start", &JsWord::start)
      .field("end", &JsWord::end)
      .field("confidence", &JsWord::confidence);

  value_object<JsSpeakerSpan>("MoonshineSpeakerSpan")
      .field("startTime", &JsSpeakerSpan::startTime)
      .field("duration", &JsSpeakerSpan::duration)
      .field("speakerId", &JsSpeakerSpan::speakerId)
      .field("speakerIndex", &JsSpeakerSpan::speakerIndex)
      .field("startChar", &JsSpeakerSpan::startChar)
      .field("endChar", &JsSpeakerSpan::endChar);

  value_object<JsLine>("MoonshineLine")
      .field("text", &JsLine::text)
      .field("startTime", &JsLine::startTime)
      .field("duration", &JsLine::duration)
      .field("id", &JsLine::id)
      .field("isComplete", &JsLine::isComplete)
      .field("isUpdated", &JsLine::isUpdated)
      .field("isNew", &JsLine::isNew)
      .field("hasTextChanged", &JsLine::hasTextChanged)
      .field("haveSpeakersChanged", &JsLine::haveSpeakersChanged)
      .field("lastTranscriptionLatencyMs", &JsLine::lastTranscriptionLatencyMs)
      .field("words", &JsLine::words)
      .field("speakerSpans", &JsLine::speakerSpans);

  value_object<JsTranscript>("MoonshineTranscript")
      .field("lines", &JsTranscript::lines);

  value_object<JsIntentMatch>("MoonshineIntentMatch")
      .field("canonicalPhrase", &JsIntentMatch::canonicalPhrase)
      .field("similarity", &JsIntentMatch::similarity);

  class_<Transcriber>("Transcriber")
      .constructor<val, val, val, val, uint32_t>()
      .function("transcribe", &Transcriber::transcribe)
      .function("close", &Transcriber::close);

  class_<Stream>("Stream")
      .constructor<Transcriber &, uint32_t>()
      .function("start", &Stream::start)
      .function("stop", &Stream::stop)
      .function("addAudio", &Stream::addAudio)
      .function("transcribe", &Stream::transcribe)
      .function("close", &Stream::close);

  class_<IntentRecognizer>("IntentRecognizer")
      .constructor<std::string, uint32_t, std::string>()
      .function("registerIntent", &IntentRecognizer::registerIntent)
      .function("unregisterIntent", &IntentRecognizer::unregisterIntent)
      .function("clearIntents", &IntentRecognizer::clearIntents)
      .function("closestIntents", &IntentRecognizer::closestIntents)
      .function("close", &IntentRecognizer::close);

#if defined(MOONSHINE_C_API_MOONSHINE_TTS) && MOONSHINE_C_API_MOONSHINE_TTS
  value_object<JsTtsResult>("MoonshineTtsResult")
      .field("audio", &JsTtsResult::audio)
      .field("sampleRate", &JsTtsResult::sampleRate);

  class_<TextToSpeech>("TextToSpeech")
      .constructor<std::string, val, val>()
      .function("say", &TextToSpeech::say)
      .function("close", &TextToSpeech::close);

  class_<GraphemeToPhonemizer>("GraphemeToPhonemizer")
      .constructor<std::string, val, val>()
      .function("textToPhonemes", &GraphemeToPhonemizer::textToPhonemes)
      .function("close", &GraphemeToPhonemizer::close);

  function("ttsDependencies", &tts_dependencies);
  function("ttsVoices", &tts_voices);
  function("g2pDependencies", &g2p_dependencies);
#endif

  function("version", &version);
  function("sttDependencies", &stt_dependencies);
  function("intentDependencies", &intent_dependencies);
}
