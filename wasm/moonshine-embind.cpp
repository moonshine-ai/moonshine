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

#include <algorithm>
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

// Copies a JS array of strings into a std::vector<std::string>.
std::vector<std::string> to_string_vector(const val &array) {
  std::vector<std::string> out;
  if (array.isUndefined() || array.isNull()) {
    return out;
  }
  const size_t count = array["length"].as<size_t>();
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(array[i].as<std::string>());
  }
  return out;
}

// Zips parallel name/value string arrays into moonshine_option_t entries. The
// returned options borrow the c_str() pointers of `names` / `values`, so those
// vectors must outlive the options (and the C ABI call they're passed to).
std::vector<moonshine_option_t> make_options(
    const std::vector<std::string> &names,
    const std::vector<std::string> &values) {
  std::vector<moonshine_option_t> options;
  const size_t count = std::min(names.size(), values.size());
  options.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    options.push_back(moonshine_option_t{names[i].c_str(), values[i].c_str()});
  }
  return options;
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
  // Passed as a decimal string, not a double: see JsLine::id.
  std::string speakerId;
  uint32_t speakerIndex = 0;
  // Character offsets into the line text, small enough for a double.
  double startChar = 0.0;
  double endChar = 0.0;
};

struct JsLine {
  std::string text;
  float startTime = 0.0f;
  float duration = 0.0f;
  // A decimal string rather than a double, because the core allocates line ids
  // as a random 64-bit base incremented by one per line (see next_line_id in
  // core/transcriber.cpp). Such ids land above 2^53, where consecutive doubles
  // are 2048 apart, so casting collapses every line in a stream onto a single
  // value and the whole LineStarted/LineCompleted model breaks. The id is
  // documented as opaque, so a string costs callers nothing and keys cleanly.
  std::string id;
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
    jl.id = std::to_string(line.id);
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
          span.start_time, span.duration, std::to_string(span.speaker_id),
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
  // Loads a transcriber from in-memory model bytes keyed by canonical filename
  // (see moonshine_load_transcriber_from_memory_files). `keys` is an array of
  // strings and `buffers` an array of Uint8Arrays of matching length; this maps
  // 1:1 onto the download manifest so every architecture (streaming and
  // non-streaming), the word-timestamp decoders, and the spelling model
  // (`spelling_cnn.ort`) load through the same path. The browser has no natural
  // filesystem, so this in-memory loader is the only STT entry point. The
  // buffers are copied into `buffers_` and kept alive for the transcriber's
  // lifetime (the C ABI references model bytes directly without copying).
  // `option_names` / `option_values` are parallel string arrays of
  // moonshine_option_t entries (e.g. `skip_transcription=true` to run only the
  // VAD + segmentation and skip the STT model entirely). When they select
  // `skip_transcription`, `keys` / `buffers` may be empty since no model files
  // are needed.
  Transcriber(val keys, val buffers, uint32_t model_arch, val option_names,
              val option_values) {
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
    const std::vector<std::string> opt_names = to_string_vector(option_names);
    const std::vector<std::string> opt_values = to_string_vector(option_values);
    const std::vector<moonshine_option_t> options =
        make_options(opt_names, opt_values);
    handle_ = moonshine_load_transcriber_from_memory_files(
        count == 0 ? nullptr : key_ptrs.data(),
        count == 0 ? nullptr : buf_ptrs.data(),
        count == 0 ? nullptr : buf_sizes.data(), count, model_arch,
        options.empty() ? nullptr : options.data(), options.size(),
        MOONSHINE_HEADER_VERSION);
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
  std::vector<std::string> key_strings_;
  std::vector<std::vector<uint8_t>> buffers_;
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

  void start() {
    check(moonshine_start_stream(transcriber_handle_, stream_handle_));
  }
  void stop() {
    check(moonshine_stop_stream(transcriber_handle_, stream_handle_));
  }

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
// Text embeddings (always compiled into the core)
// ---------------------------------------------------------------------------

class EmbeddingModel {
 public:
  // Loads the embedding model from in-memory bytes keyed by canonical filename
  // (see moonshine_create_embedding_model_from_memory). `keys` is an array of
  // strings and `buffers` an array of Uint8Arrays of matching length, matching
  // the download manifest (`model_<variant>.ort` + `tokenizer.bin`). The
  // browser has no natural filesystem, so this in-memory loader is the only
  // entry point. Buffers are copied into `buffers_` for the object's lifetime.
  EmbeddingModel(val keys, val buffers, uint32_t model_arch,
                 const std::string &model_variant) {
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
    handle_ = moonshine_create_embedding_model_from_memory(
        model_arch, model_variant.empty() ? nullptr : model_variant.c_str(),
        key_ptrs.data(), count, buf_ptrs.data(), buf_sizes.data(), nullptr, 0,
        MOONSHINE_HEADER_VERSION);
    if (handle_ < 0) {
      throw_moonshine_error(handle_);
    }
  }

  ~EmbeddingModel() { close(); }
  EmbeddingModel(const EmbeddingModel &) = delete;
  EmbeddingModel &operator=(const EmbeddingModel &) = delete;

  // Returns the embedding for `sentence` as a Float32Array.
  val calculateEmbedding(const std::string &sentence) {
    float *embedding = nullptr;
    uint64_t size = 0;
    check(moonshine_calculate_embedding(handle_, sentence.c_str(), &embedding,
                                        &size, nullptr));
    // Copy into a JS Float32Array before freeing the C buffer.
    val result = val::global("Float32Array").new_(static_cast<double>(size));
    val heap = val(emscripten::typed_memory_view(size, embedding));
    result.call<void>("set", heap);
    moonshine_free_embedding(embedding);
    return result;
  }

  // Cosine similarity of two equal-length embeddings, in [-1, 1].
  float distance(val embedding_a, val embedding_b) {
    const std::vector<float> a = to_float_vector(embedding_a);
    const std::vector<float> b = to_float_vector(embedding_b);
    if (a.empty() || a.size() != b.size()) {
      throw_moonshine_error(MOONSHINE_ERROR_INVALID_ARGUMENT);
    }
    float similarity = 0.0f;
    check(moonshine_calculate_embedding_distance(handle_, a.data(), b.data(),
                                                 a.size(), &similarity));
    return similarity;
  }

  void close() {
    if (handle_ >= 0) {
      moonshine_free_embedding_model(handle_);
      handle_ = -1;
    }
  }

 private:
  std::vector<std::string> key_strings_;
  std::vector<std::vector<uint8_t>> buffers_;
  int32_t handle_ = -1;
};

// ---------------------------------------------------------------------------
// Speech clip extraction (voice cloning reference clips)
// ---------------------------------------------------------------------------

struct JsSpeechClip {
  val audio = val::undefined();  // Float32Array, 16 kHz mono; empty if !ready
  float startTime = 0.0f;
  float speechDuration = 0.0f;
  bool isComplete = false;
  std::string transcript;
};

// Wraps moonshine_extract_speech_clip. `clipDurationSeconds` and
// `minimumSpeechSeconds` mirror the C options of the same name. Safe to call
// repeatedly on a growing buffer: `isComplete` reports whether enough speech
// has been captured yet.
JsSpeechClip extract_speech_clip(val audio, int32_t sample_rate,
                                 int32_t tts_synthesizer_handle,
                                 float clip_duration_seconds,
                                 float minimum_speech_seconds) {
  const std::vector<float> pcm = to_float_vector(audio);
  JsSpeechClip result;
  if (pcm.empty()) {
    return result;
  }
  const std::string clip_duration = std::to_string(clip_duration_seconds);
  const std::string minimum_speech = std::to_string(minimum_speech_seconds);
  const moonshine_option_t options[] = {
      {"clip_duration_seconds", clip_duration.c_str()},
      {"minimum_speech_seconds", minimum_speech.c_str()},
  };
  moonshine_speech_clip_t clip{};
  check(moonshine_extract_speech_clip(
      pcm.data(), pcm.size(), sample_rate, tts_synthesizer_handle, options, 2,
      &clip));
  result.startTime = clip.start_time;
  result.speechDuration = clip.speech_duration;
  result.isComplete = clip.is_complete != 0;
  if (clip.audio_data != nullptr && clip.audio_length > 0) {
    result.audio = val::global("Float32Array")
                       .new_(static_cast<double>(clip.audio_length));
    val heap =
        val(emscripten::typed_memory_view(clip.audio_length, clip.audio_data));
    result.audio.call<void>("set", heap);
    moonshine_free_buffer(clip.audio_data);
  }
  if (clip.transcript != nullptr) {
    result.transcript = clip.transcript;
    moonshine_free_buffer(clip.transcript);
  }
  return result;
}

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
  // `option_names` / `option_values` are parallel string arrays of
  // moonshine_option_t entries, e.g. `voice=kokoro_af_heart` to pick a vocoder
  // + voice, or `voice=zipvoice` together with a `zipvoice/clone_audio` buffer
  // and `zipvoice_clone_sample_rate` for zero-shot voice cloning.
  TextToSpeech(const std::string &language, val keys, val buffers,
               val option_names, val option_values) {
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
    const std::vector<std::string> opt_names = to_string_vector(option_names);
    const std::vector<std::string> opt_values = to_string_vector(option_values);
    const std::vector<moonshine_option_t> options =
        make_options(opt_names, opt_values);
    handle_ = moonshine_create_tts_synthesizer_from_memory(
        language.c_str(), key_ptrs.data(), count, buf_ptrs.data(),
        buf_sizes.data(), options.empty() ? nullptr : options.data(),
        options.size(), MOONSHINE_HEADER_VERSION);
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

  int32_t handle() const { return handle_; }

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
                             const std::string &model_arch,
                             bool include_spelling) {
  std::vector<moonshine_option_t> options;
  if (!model_arch.empty()) {
    options.push_back(moonshine_option_t{"model_arch", model_arch.c_str()});
  }
  if (include_spelling) {
    options.push_back(moonshine_option_t{"include_spelling", "true"});
  }
  char *json = nullptr;
  check(moonshine_get_stt_dependencies(
      language.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

std::string embedding_dependencies(const std::string &model_name,
                                   const std::string &variant) {
  std::vector<moonshine_option_t> options;
  if (!variant.empty()) {
    options.push_back(moonshine_option_t{"variant", variant.c_str()});
  }
  char *json = nullptr;
  check(moonshine_get_embedding_dependencies(
      model_name.empty() ? nullptr : model_name.c_str(),
      options.empty() ? nullptr : options.data(), options.size(), &json));
  std::string out = json ? json : "";
  free(json);
  return out;
}

std::string diarization_dependencies() {
  char *json = nullptr;
  check(moonshine_get_diarization_dependencies(&json));
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

std::string tts_voices(const std::string &languages, val option_names,
                       val option_values) {
  const std::vector<std::string> opt_names = to_string_vector(option_names);
  const std::vector<std::string> opt_values = to_string_vector(option_values);
  const std::vector<moonshine_option_t> options =
      make_options(opt_names, opt_values);
  char *json = nullptr;
  check(moonshine_get_tts_voices(languages.c_str(),
                                 options.empty() ? nullptr : options.data(),
                                 options.size(), &json));
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

  value_object<JsSpeechClip>("MoonshineSpeechClip")
      .field("audio", &JsSpeechClip::audio)
      .field("startTime", &JsSpeechClip::startTime)
      .field("speechDuration", &JsSpeechClip::speechDuration)
      .field("isComplete", &JsSpeechClip::isComplete)
      .field("transcript", &JsSpeechClip::transcript);

  class_<Transcriber>("Transcriber")
      .constructor<val, val, uint32_t, val, val>()
      .function("transcribe", &Transcriber::transcribe)
      .function("close", &Transcriber::close);

  class_<Stream>("Stream")
      .constructor<Transcriber &, uint32_t>()
      .function("start", &Stream::start)
      .function("stop", &Stream::stop)
      .function("addAudio", &Stream::addAudio)
      .function("transcribe", &Stream::transcribe)
      .function("close", &Stream::close);

  class_<EmbeddingModel>("EmbeddingModel")
      .constructor<val, val, uint32_t, std::string>()
      .function("calculateEmbedding", &EmbeddingModel::calculateEmbedding)
      .function("distance", &EmbeddingModel::distance)
      .function("close", &EmbeddingModel::close);

#if defined(MOONSHINE_C_API_MOONSHINE_TTS) && MOONSHINE_C_API_MOONSHINE_TTS
  value_object<JsTtsResult>("MoonshineTtsResult")
      .field("audio", &JsTtsResult::audio)
      .field("sampleRate", &JsTtsResult::sampleRate);

  class_<TextToSpeech>("TextToSpeech")
      .constructor<std::string, val, val, val, val>()
      .function("say", &TextToSpeech::say)
      .function("handle", &TextToSpeech::handle)
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
  function("embeddingDependencies", &embedding_dependencies);
  function("diarizationDependencies", &diarization_dependencies);
  function("extractSpeechClip", &extract_speech_clip);
}
