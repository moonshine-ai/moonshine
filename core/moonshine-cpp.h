#ifndef MOONSHINE_CPP_H
#define MOONSHINE_CPP_H

/* Moonshine C++ API - Header-only library
 *
 * This is a C++11 wrapper around the Moonshine C API, providing a modern
 * object-oriented interface with event-driven streaming support.
 *
 * Example usage:
 * ```cpp
 * #include "moonshine-cpp.h"
 * #include <iostream>
 *
 * class MyListener : public moonshine::TranscriptEventListener {
 * public:
 *     void onLineStarted(const moonshine::LineStarted& event) override {
 *         std::cout << "Line started: " << event.line.text << std::endl;
 *     }
 *     void onLineCompleted(const moonshine::LineCompleted& event) override {
 *         std::cout << "Line completed: " << event.line.text << std::endl;
 *     }
 * };
 *
 * int main() {
 *     moonshine::Transcriber transcriber("path/to/models",
 * moonshine::ModelArch::BASE);
 *
 *     MyListener listener;
 *     transcriber.addListener(&listener);
 *
 *     transcriber.start();
 *
 *     std::vector<float> audio_data = { ... your audio data ... };
 *     transcriber.addAudio(audio_data, 16000);
 *
 *     transcriber.stop();
 *     return 0;
 * }
 * ```
 */

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "moonshine-c-api.h"

namespace moonshine {

// Forward declarations
class Transcriber;
class Stream;
class TranscriptEventListener;
class TextToSpeech;
class GraphemeToPhonemizer;
class IntentRecognizer;

/* ------------------------------ ENUMS -------------------------------- */

/// Model architecture enumeration
enum class ModelArch {
  TINY = MOONSHINE_MODEL_ARCH_TINY,
  BASE = MOONSHINE_MODEL_ARCH_BASE,
  TINY_STREAMING = MOONSHINE_MODEL_ARCH_TINY_STREAMING,
  BASE_STREAMING = MOONSHINE_MODEL_ARCH_BASE_STREAMING,
  SMALL_STREAMING = MOONSHINE_MODEL_ARCH_SMALL_STREAMING,
  MEDIUM_STREAMING = MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
};

/// Embedding model architectures for intent recognition (C API constants).
enum class EmbeddingModelArch : uint32_t {
  GEMMA_300M = MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
};

/* --------------------------- DATA STRUCTURES ----------------------------- */

/// A single word with timing information
struct WordTiming {
  /// The word text
  std::string word;
  /// Start time in seconds (absolute, from start of audio/stream)
  float start;
  /// End time in seconds
  float end;
  /// Model confidence score, 0.0 to 1.0
  float confidence;

  WordTiming() : start(0.0f), end(0.0f), confidence(0.0f) {}
  WordTiming(const std::string &word, float start, float end, float confidence)
      : word(word), start(start), end(end), confidence(confidence) {}
};

/// A single line of transcription
struct TranscriptLine {
  /// UTF-8 encoded transcription text
  std::string text;

  /// Time offset from the start of the audio in seconds
  float startTime;

  /// Duration of the segment in seconds
  float duration;

  /// Stable identifier for the line
  uint64_t lineId;

  /// Whether the line is complete (streaming only)
  bool isComplete;

  /// Whether the line has been updated since the previous call (streaming only)
  bool isUpdated;

  /// Whether the line was newly added since the previous call (streaming only)
  bool isNew;

  /// Whether the text of the line has changed since the previous call
  /// (streaming only)
  bool hasTextChanged;

  /// Whether a speaker ID has been calculated for the line.
  bool hasSpeakerId;

  /// The speaker ID for the line.
  uint64_t speakerId;

  /// The order the speaker appeared in the current transcript.
  uint32_t speakerIndex;

  int32_t lastTranscriptionLatencyMs;

  /// Audio data for this line, if available (16KHz float PCM, -1.0 to 1.0)
  std::vector<float> audioData;

  /// Word-level timestamps (empty if word_timestamps option is not enabled)
  std::vector<WordTiming> words;

  /// Default constructor
  TranscriptLine()
      : startTime(0.0f),
        duration(0.0f),
        lineId(0),
        isComplete(false),
        isUpdated(false),
        isNew(false),
        hasTextChanged(false),
        hasSpeakerId(false),
        speakerId(0),
        speakerIndex(0),
        lastTranscriptionLatencyMs(0) {}

  /// Construct from C API structure
  TranscriptLine(const transcript_line_t &line_c)
      : startTime(line_c.start_time),
        duration(line_c.duration),
        lineId(line_c.id),
        isComplete(line_c.is_complete != 0),
        isUpdated(line_c.is_updated != 0),
        isNew(line_c.is_new != 0),
        hasTextChanged(line_c.has_text_changed != 0),
        hasSpeakerId(line_c.has_speaker_id != 0),
        speakerId(line_c.speaker_id),
        speakerIndex(line_c.speaker_index),
        lastTranscriptionLatencyMs(line_c.last_transcription_latency_ms) {
    if (line_c.text) {
      text = std::string(line_c.text);
    }
    if (line_c.audio_data && line_c.audio_data_count > 0) {
      audioData.assign(line_c.audio_data,
                       line_c.audio_data + line_c.audio_data_count);
    }
    if (line_c.words && line_c.word_count > 0) {
      words.reserve(line_c.word_count);
      for (uint64_t i = 0; i < line_c.word_count; i++) {
        const auto &w = line_c.words[i];
        words.emplace_back(
            w.text ? std::string(w.text) : std::string(),
            w.start, w.end, w.confidence);
      }
    }
  }

  std::string toString() const {
    return "[" + std::to_string(startTime) + "s] '" + text + "' (" +
           std::to_string(duration) + "s) [" + std::to_string(lineId) + "] " +
           (isComplete ? " complete, " : " incomplete, ") +
           (isUpdated ? " updated, " : " not updated, ") +
           (isNew ? " new" : " not new, ") +
           (hasTextChanged ? " text changed" : " text not changed") +
           (hasSpeakerId ? " speaker id=" + std::to_string(speakerId) +
           ", speaker index=" + std::to_string(speakerIndex) : "") +
           ", last transcription latency ms=" +
           std::to_string(lastTranscriptionLatencyMs);
  }
};

/// A complete transcript containing multiple lines
struct Transcript {
  /// All lines of the transcript
  std::vector<TranscriptLine> lines;

  /// Default constructor
  Transcript() {}

  /// Construct from C API structure
  Transcript(const transcript_t *transcript_c) {
    if (!transcript_c) {
      return;
    }
    lines.reserve(transcript_c->line_count);
    for (uint64_t i = 0; i < transcript_c->line_count; ++i) {
      TranscriptLine line(transcript_c->lines[i]);
      lines.push_back(line);
    }
  }

  std::string toString() const {
    std::string result;
    result += "Transcript with " + std::to_string(lines.size()) + " lines:\n";
    for (const auto &line : lines) {
      result += line.toString() + "\n";
    }
    return result;
  }
};

/* ------------------------------ EVENTS -------------------------------- */

/// Base class for all transcript events
class TranscriptEvent {
 public:
  /// Event type enumeration
  enum Type {
    LINE_STARTED,
    LINE_UPDATED,
    LINE_TEXT_CHANGED,
    LINE_COMPLETED,
    ERROR
  };

  /// The transcript line associated with this event
  TranscriptLine line;

  /// The handle of the stream that emitted this event
  int32_t streamHandle;

  /// The type of this event
  Type type;

  virtual ~TranscriptEvent() {}

 protected:
  TranscriptEvent(const TranscriptLine &line, int32_t streamHandle, Type type)
      : line(line), streamHandle(streamHandle), type(type) {}
};

/// Event emitted when a new transcription line starts
class LineStarted : public TranscriptEvent {
 public:
  LineStarted(const TranscriptLine &line, int32_t streamHandle)
      : TranscriptEvent(line, streamHandle, LINE_STARTED) {}
};

/// Event emitted when an existing transcription line is updated
class LineUpdated : public TranscriptEvent {
 public:
  LineUpdated(const TranscriptLine &line, int32_t streamHandle)
      : TranscriptEvent(line, streamHandle, LINE_UPDATED) {}
};

/// Event emitted when the text of a transcription line changes
class LineTextChanged : public TranscriptEvent {
 public:
  LineTextChanged(const TranscriptLine &line, int32_t streamHandle)
      : TranscriptEvent(line, streamHandle, LINE_TEXT_CHANGED) {}
};

/// Event emitted when a transcription line is completed
class LineCompleted : public TranscriptEvent {
 public:
  LineCompleted(const TranscriptLine &line, int32_t streamHandle)
      : TranscriptEvent(line, streamHandle, LINE_COMPLETED) {}
};

/// Event emitted when an error occurs
class Error : public TranscriptEvent {
 public:
  /// The error message
  std::string errorMessage;

  Error(const std::string &errorMessage, int32_t streamHandle)
      : TranscriptEvent(TranscriptLine(), streamHandle, ERROR),
        errorMessage(errorMessage) {}

  Error(const std::string &errorMessage, const TranscriptLine &line,
        int32_t streamHandle)
      : TranscriptEvent(line, streamHandle, ERROR),
        errorMessage(errorMessage) {}
};

/* ------------------------------ LISTENER -------------------------------- */

/// Abstract base class for transcript event listeners
///
/// Subclass this and override the methods you want to handle.
/// All methods have default no-op implementations, so you only need to
/// override the ones you care about.
class TranscriptEventListener {
 public:
  virtual ~TranscriptEventListener() {}

  /// Called when a new transcription line starts
  virtual void onLineStarted(const LineStarted &) {}

  /// Called when an existing transcription line is updated
  virtual void onLineUpdated(const LineUpdated &) {}

  /// Called when the text of a transcription line changes
  virtual void onLineTextChanged(const LineTextChanged &) {}

  /// Called when a transcription line is completed
  virtual void onLineCompleted(const LineCompleted &) {}

  /// Called when an error occurs
  virtual void onError(const Error &) {}
};

/* ------------------------------ EXCEPTION -------------------------------- */

/// Exception class for Moonshine errors
class MoonshineException : public std::runtime_error {
 public:
  MoonshineException(const std::string &message)
      : std::runtime_error(message) {}
};

/* ------------------------------ STREAM -------------------------------- */

/// Stream for real-time transcription with event-based updates
class Stream {
 public:
  /// Flags for transcription operations
  static const uint32_t FLAG_FORCE_UPDATE = MOONSHINE_FLAG_FORCE_UPDATE;

  /// Create a stream (internal use)
  Stream(Transcriber *transcriber, double updateInterval = 0.5,
         uint32_t flags = 0);

  /// Destructor - automatically closes the stream
  ~Stream();

  /// Move constructor
  Stream(Stream &&other);

  /// Move assignment
  Stream &operator=(Stream &&other);

  // Delete copy constructor and assignment
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  /// Start the stream
  void start();

  /// Stop the stream
  /// This will process any remaining audio and emit final events
  void stop();

  /// Add audio data to the stream
  /// @param audioData Array of PCM audio samples (float, -1.0 to 1.0)
  /// @param sampleRate Sample rate in Hz (default: 16000)
  void addAudio(const std::vector<float> &audioData,
                int32_t sampleRate = 16000);

  /// Manually update the transcription from the stream
  /// @param flags Flags for transcription (e.g., Stream::FLAG_FORCE_UPDATE)
  /// @return The current transcript
  Transcript updateTranscription(uint32_t flags = 0);

  /// Add an event listener to the stream
  /// @param listener A TranscriptEventListener instance
  void addListener(TranscriptEventListener *listener);

  /// Add a function-based event listener to the stream
  /// @param listener A function that takes a TranscriptEvent reference
  void addListener(std::function<void(const TranscriptEvent &)> listener);

  /// Remove an event listener from the stream
  /// @param listener The listener to remove
  void removeListener(TranscriptEventListener *listener);

  /// Remove a function-based event listener from the stream
  /// @param listener The listener function to remove
  void removeListener(std::function<void(const TranscriptEvent &)> listener);

  /// Remove all event listeners from the stream
  void removeAllListeners();

  /// Close the stream and free its resources
  void close();

  /// Get the stream handle (for internal use)
  int32_t getHandle() const { return handle_; }

 private:
  Transcriber *transcriber_;
  int32_t handle_;
  double updateInterval_;
  double streamTime_;
  double lastUpdateTime_;

  // Listener storage
  std::vector<TranscriptEventListener *> objectListeners_;
  std::vector<std::function<void(const TranscriptEvent &)>> functionListeners_;

  void notifyFromTranscript(const Transcript &transcript);
  void emit(const TranscriptEvent &event);
  void emitError(const std::string &errorMessage);
  void checkError(int32_t error) const;

  friend class Transcriber;
};

/* ------------------------------ TRANSCRIBER --------------------------------
 */

/// Main transcriber class for Moonshine Voice
class Transcriber {
 public:
  /// Flags for transcription operations
  static const uint32_t FLAG_FORCE_UPDATE = MOONSHINE_FLAG_FORCE_UPDATE;

  /// Initialize a transcriber from model files on disk
  /// @param modelPath Path to the directory containing model files
  /// @param modelArch Model architecture to use (default: ModelArch::BASE)
  /// @param updateInterval Interval in seconds between automatic updates
  /// (default: 0.5)
  /// @throws MoonshineException if the transcriber cannot be loaded
  Transcriber(const std::string &modelPath, ModelArch modelArch,
              double updateInterval = 0.5);

  /// Destructor - automatically closes the transcriber
  ~Transcriber();

  /// Move constructor
  Transcriber(Transcriber &&other);

  /// Move assignment
  Transcriber &operator=(Transcriber &&other);

  // Delete copy constructor and assignment
  Transcriber(const Transcriber &) = delete;
  Transcriber &operator=(const Transcriber &) = delete;

  /// Free the transcriber resources
  void close();

  /// Transcribe audio data without streaming
  /// @param audioData Array of PCM audio samples (float, -1.0 to 1.0)
  /// @param sampleRate Sample rate in Hz (default: 16000)
  /// @param flags Flags for transcription (default: 0)
  /// @return Transcript object containing the transcription lines
  /// @throws MoonshineException if transcription fails
  Transcript transcribeWithoutStreaming(const std::vector<float> &audioData,
                                        int32_t sampleRate = 16000,
                                        uint32_t flags = 0);

  /// Get the version of the loaded Moonshine library
  /// @return The version number
  int32_t getVersion() const;

  /// Create a new stream for real-time transcription
  /// @param updateInterval Interval in seconds between automatic updates
  /// (default: 0.5)
  /// @param flags Flags for stream creation (default: 0)
  /// @return Stream object for real-time transcription
  /// @throws MoonshineException if stream creation fails
  Stream createStream(double updateInterval = 0.5, uint32_t flags = 0);

  /// Get the default stream
  /// @return The default stream
  /// @throws MoonshineException if stream creation fails
  Stream &getDefaultStream();

  /// Start the default stream
  /// @throws MoonshineException if starting fails
  void start();

  /// Stop the default stream
  /// @throws MoonshineException if stopping fails
  void stop();

  /// Add audio data to the default stream
  /// @param audioData Array of PCM audio samples (float, -1.0 to 1.0)
  /// @param sampleRate Sample rate in Hz (default: 16000)
  /// @throws MoonshineException if adding audio fails
  void addAudio(const std::vector<float> &audioData,
                int32_t sampleRate = 16000);

  /// Update the transcription from the default stream
  /// @param flags Flags for transcription (default: 0)
  /// @return The current transcript
  /// @throws MoonshineException if updating fails
  Transcript updateTranscription(uint32_t flags = 0);

  /// Add an event listener to the default stream
  /// @param listener A TranscriptEventListener instance
  void addListener(TranscriptEventListener *listener);

  /// Add a function-based event listener to the default stream
  /// @param listener A function that takes a TranscriptEvent reference
  void addListener(std::function<void(const TranscriptEvent &)> listener);

  /// Remove an event listener from the default stream
  /// @param listener The listener to remove
  void removeListener(TranscriptEventListener *listener);

  /// Remove a function-based event listener from the default stream
  /// @param listener The listener function to remove
  void removeListener(std::function<void(const TranscriptEvent &)> listener);

  /// Remove all event listeners from the default stream
  void removeAllListeners();

  /// Get the transcriber handle (for internal use)
  int32_t getHandle() const { return handle_; }

 private:
  int32_t handle_;
  std::string modelPath_;
  ModelArch modelArch_;
  double updateInterval_;
  std::unique_ptr<Stream> defaultStream_;

  Transcript parseTranscript(const transcript_t *transcript_c);
  void checkError(int32_t error) const;

  friend class Stream;
};

/* --------------------------- TTS DATA STRUCTURES --------------------------- */

/// Result of a text-to-speech synthesis operation
struct TtsSynthesisResult {
  /// PCM audio samples (float, approximately -1.0 to 1.0)
  std::vector<float> samples;
  /// Sample rate in Hz
  int32_t sampleRateHz;

  TtsSynthesisResult() : sampleRateHz(0) {}
  TtsSynthesisResult(std::vector<float> samples, int32_t sampleRateHz)
      : samples(std::move(samples)), sampleRateHz(sampleRateHz) {}
};

/* ----------------------------- TEXT TO SPEECH ------------------------------- */

/// Text-to-speech synthesizer
///
/// Wraps the Moonshine TTS C API with RAII resource management.
///
/// Example usage:
/// ```cpp
/// #include "moonshine-cpp.h"
/// #include <iostream>
///
/// int main() {
///     std::vector<moonshine_option_t> options = {
///         {"g2p_root", "/path/to/assets"},
///         {"voice", "kokoro_af_heart"},
///     };
///     moonshine::TextToSpeech tts("en_us", options);
///     auto result = tts.synthesize("Hello world!");
///     std::cout << "Got " << result.samples.size() << " samples at "
///               << result.sampleRateHz << " Hz" << std::endl;
///     return 0;
/// }
/// ```
class TextToSpeech {
 public:
  /// Create a TTS synthesizer from files on disk
  /// @param language Language tag (e.g., "en_us", "de", "fr")
  /// @param options Configuration options (voice, g2p_root, speed, etc.)
  /// @throws MoonshineException if creation fails
  TextToSpeech(const std::string &language,
               const std::vector<moonshine_option_t> &options = {});

  /// Destructor - automatically frees resources
  ~TextToSpeech();

  /// Move constructor
  TextToSpeech(TextToSpeech &&other);

  /// Move assignment
  TextToSpeech &operator=(TextToSpeech &&other);

  // Delete copy constructor and assignment
  TextToSpeech(const TextToSpeech &) = delete;
  TextToSpeech &operator=(const TextToSpeech &) = delete;

  /// Synthesize text to speech audio
  /// @param text The text to synthesize
  /// @param options Per-call options (e.g., speed), or empty to use defaults
  /// @return TtsSynthesisResult containing PCM samples and sample rate
  /// @throws MoonshineException if synthesis fails
  TtsSynthesisResult synthesize(
      const std::string &text,
      const std::vector<moonshine_option_t> &options = {});

  /// Free the synthesizer resources
  void close();

  /// Get the language tag
  const std::string &getLanguage() const { return language_; }

  /// Get the synthesizer handle (for internal use)
  int32_t getHandle() const { return handle_; }

  /// Get available TTS voices for one or more languages
  /// @param languages Comma-separated language tags (empty for all)
  /// @param options Configuration options (g2p_root for accurate availability)
  /// @return JSON string mapping language tags to voice arrays
  /// @throws MoonshineException on failure
  static std::string getVoices(
      const std::string &languages,
      const std::vector<moonshine_option_t> &options = {});

  /// Get TTS asset dependency keys for one or more languages
  /// @param languages Comma-separated language tags (empty for all)
  /// @param options Configuration options (voice, g2p_root, etc.)
  /// @return JSON array string of asset keys
  /// @throws MoonshineException on failure
  static std::string getDependencies(
      const std::string &languages,
      const std::vector<moonshine_option_t> &options = {});

 private:
  int32_t handle_;
  std::string language_;

  void checkError(int32_t error) const;
};

/* ------------------------- GRAPHEME TO PHONEMIZER -------------------------- */

/// Grapheme-to-phoneme converter
///
/// Converts text to IPA (International Phonetic Alphabet) phonemes.
///
/// Example usage:
/// ```cpp
/// #include "moonshine-cpp.h"
/// #include <iostream>
///
/// int main() {
///     std::vector<moonshine_option_t> options = {
///         {"g2p_root", "/path/to/assets"},
///     };
///     moonshine::GraphemeToPhonemizer g2p("en_us", options);
///     std::string ipa = g2p.toIpa("Hello world!");
///     std::cout << "IPA: " << ipa << std::endl;
///     return 0;
/// }
/// ```
class GraphemeToPhonemizer {
 public:
  /// Create a grapheme-to-phonemizer from files on disk
  /// @param language Language tag (e.g., "en_us", "de", "fr")
  /// @param options Configuration options (g2p_root, etc.)
  /// @throws MoonshineException if creation fails
  GraphemeToPhonemizer(const std::string &language,
                       const std::vector<moonshine_option_t> &options = {});

  /// Destructor - automatically frees resources
  ~GraphemeToPhonemizer();

  /// Move constructor
  GraphemeToPhonemizer(GraphemeToPhonemizer &&other);

  /// Move assignment
  GraphemeToPhonemizer &operator=(GraphemeToPhonemizer &&other);

  // Delete copy constructor and assignment
  GraphemeToPhonemizer(const GraphemeToPhonemizer &) = delete;
  GraphemeToPhonemizer &operator=(const GraphemeToPhonemizer &) = delete;

  /// Convert text to IPA phonemes
  /// @param text The text to convert
  /// @param options Per-call options, or empty to use defaults
  /// @return IPA phoneme string
  /// @throws MoonshineException if conversion fails
  std::string toIpa(const std::string &text,
                    const std::vector<moonshine_option_t> &options = {});

  /// Free the phonemizer resources
  void close();

  /// Get the language tag
  const std::string &getLanguage() const { return language_; }

  /// Get the phonemizer handle (for internal use)
  int32_t getHandle() const { return handle_; }

  /// Get G2P asset dependency keys for one or more languages
  /// @param languages Comma-separated language tags (empty for all)
  /// @param options Configuration options
  /// @return Comma-separated list of asset keys
  /// @throws MoonshineException on failure
  static std::string getDependencies(
      const std::string &languages,
      const std::vector<moonshine_option_t> &options = {});

 private:
  int32_t handle_;
  std::string language_;

  void checkError(int32_t error) const;
};

/* ------------------------------ INTENT RECOGNIZER ------------------------- */

/// One ranked intent from ``IntentRecognizer::getClosestIntents``.
struct IntentMatch {
  std::string canonicalPhrase;
  float similarity{0.0f};

  IntentMatch() = default;
  IntentMatch(std::string phrase, float sim)
      : canonicalPhrase(std::move(phrase)), similarity(sim) {}
};

/// Intent recognizer wrapping the Moonshine intent C API (synchronous ranking).
class IntentRecognizer {
 public:
  /// Load an embedding model from disk.
  /// @throws MoonshineException if the model cannot be loaded
  IntentRecognizer(const std::string &model_path, EmbeddingModelArch arch,
                     const std::string &model_variant = "q4");

  ~IntentRecognizer();

  IntentRecognizer(IntentRecognizer &&other) noexcept;
  IntentRecognizer &operator=(IntentRecognizer &&other) noexcept;

  IntentRecognizer(const IntentRecognizer &) = delete;
  IntentRecognizer &operator=(const IntentRecognizer &) = delete;

  /// Register a canonical phrase to match against.
  /// @throws MoonshineException on failure
  void registerIntent(const std::string &canonical_phrase);

  /// Remove a phrase. Returns false if it was not registered.
  bool unregisterIntent(const std::string &canonical_phrase);

  /// Rank registered intents by similarity (see ``moonshine_get_closest_intents``).
  std::vector<IntentMatch> getClosestIntents(const std::string &utterance,
                                             float tolerance_threshold);

  /// Number of registered intents.
  /// @throws MoonshineException on invalid handle
  int32_t intentCount() const;

  /// Remove all intents.
  /// @throws MoonshineException on failure
  void clearIntents();

  void close();

  int32_t getHandle() const { return handle_; }

 private:
  int32_t handle_;

  void checkError(int32_t error) const;
};

/* ------------------------------ IMPLEMENTATION
 * -------------------------------- */

// Stream implementation
inline Stream::Stream(Transcriber *transcriber, double updateInterval,
                      uint32_t flags)
    : transcriber_(transcriber),
      handle_(-1),
      updateInterval_(updateInterval),
      streamTime_(0.0),
      lastUpdateTime_(0.0) {
  handle_ = moonshine_create_stream(transcriber_->handle_, flags);
  checkError(handle_);
}

inline Stream::~Stream() { close(); }

inline Stream::Stream(Stream &&other)
    : transcriber_(other.transcriber_),
      handle_(other.handle_),
      updateInterval_(other.updateInterval_),
      streamTime_(other.streamTime_),
      lastUpdateTime_(other.lastUpdateTime_),
      objectListeners_(std::move(other.objectListeners_)),
      functionListeners_(std::move(other.functionListeners_)) {
  other.handle_ = -1;
}

inline Stream &Stream::operator=(Stream &&other) {
  if (this != &other) {
    close();
    transcriber_ = other.transcriber_;
    handle_ = other.handle_;
    updateInterval_ = other.updateInterval_;
    streamTime_ = other.streamTime_;
    lastUpdateTime_ = other.lastUpdateTime_;
    objectListeners_ = std::move(other.objectListeners_);
    functionListeners_ = std::move(other.functionListeners_);
    other.handle_ = -1;
  }
  return *this;
}

inline void Stream::start() {
  checkError(moonshine_start_stream(transcriber_->handle_, handle_));
}

inline void Stream::stop() {
  checkError(moonshine_stop_stream(transcriber_->handle_, handle_));
  // There may be some audio left in the stream, so we need to transcribe it
  // to get the final transcript and emit events.
  try {
    updateTranscription(0);
  } catch (const MoonshineException &e) {
    emitError(e.what());
  }
}

inline void Stream::addAudio(const std::vector<float> &audioData,
                             int32_t sampleRate) {
  if (audioData.empty()) {
    return;
  }
  checkError(moonshine_transcribe_add_audio_to_stream(
      transcriber_->handle_, handle_, const_cast<float *>(audioData.data()),
      audioData.size(), sampleRate, 0));
  streamTime_ +=
      static_cast<double>(audioData.size()) / static_cast<double>(sampleRate);
  if (streamTime_ - lastUpdateTime_ >= updateInterval_) {
    updateTranscription(0);
    lastUpdateTime_ = streamTime_;
  }
}

inline Transcript Stream::updateTranscription(uint32_t flags) {
  transcript_t *out_transcript = nullptr;
  checkError(moonshine_transcribe_stream(transcriber_->handle_, handle_, flags,
                                         &out_transcript));
  Transcript transcript = transcriber_->parseTranscript(out_transcript);
  notifyFromTranscript(transcript);
  return transcript;
}

inline void Stream::addListener(TranscriptEventListener *listener) {
  if (listener) {
    objectListeners_.push_back(listener);
  }
}

inline void Stream::addListener(
    std::function<void(const TranscriptEvent &)> listener) {
  functionListeners_.push_back(listener);
}

inline void Stream::removeListener(TranscriptEventListener *listener) {
  objectListeners_.erase(
      std::remove(objectListeners_.begin(), objectListeners_.end(), listener),
      objectListeners_.end());
}

inline void Stream::removeListener(
    std::function<void(const TranscriptEvent &)> listener) {
  // Note: We can't reliably compare std::function objects, so this is a
  // best-effort removal Users should prefer TranscriptEventListener objects if
  // they need to remove listeners
  functionListeners_.erase(
      std::remove_if(
          functionListeners_.begin(), functionListeners_.end(),
          [&listener](const std::function<void(const TranscriptEvent &)> &f) {
            // Best-effort comparison: compare target addresses if both have
            // targets (use function pointer type for target<> to avoid
            // std::function::target const_cast issues with function types)
            using fn_t = void (*)(const TranscriptEvent &);
            auto f_target = f.target<fn_t>();
            auto listener_target = listener.target<fn_t>();
            if (f_target == nullptr || listener_target == nullptr) {
              return false;  // Can't compare if either is null
            }
            return f_target == listener_target;
          }),
      functionListeners_.end());
}

inline void Stream::removeAllListeners() {
  objectListeners_.clear();
  functionListeners_.clear();
}

inline void Stream::close() {
  if (handle_ >= 0) {
    moonshine_free_stream(transcriber_->handle_, handle_);
    handle_ = -1;
  }
  removeAllListeners();
}

inline void Stream::notifyFromTranscript(const Transcript &transcript) {
  for (const auto &line : transcript.lines) {
    if (line.isNew) {
      emit(LineStarted(line, handle_));
    }
    if (line.isUpdated && !line.isNew && !line.isComplete) {
      emit(LineUpdated(line, handle_));
    }
    if (line.hasTextChanged) {
      emit(LineTextChanged(line, handle_));
    }
    if (line.isComplete && line.isUpdated) {
      emit(LineCompleted(line, handle_));
    }
  }
}

inline void Stream::emit(const TranscriptEvent &event) {
  // Emit to object listeners
  for (auto *listener : objectListeners_) {
    try {
      switch (event.type) {
        case TranscriptEvent::LINE_STARTED:
          listener->onLineStarted(static_cast<const LineStarted &>(event));
          break;
        case TranscriptEvent::LINE_UPDATED:
          listener->onLineUpdated(static_cast<const LineUpdated &>(event));
          break;
        case TranscriptEvent::LINE_TEXT_CHANGED:
          listener->onLineTextChanged(
              static_cast<const LineTextChanged &>(event));
          break;
        case TranscriptEvent::LINE_COMPLETED:
          listener->onLineCompleted(static_cast<const LineCompleted &>(event));
          break;
        case TranscriptEvent::ERROR:
          listener->onError(static_cast<const Error &>(event));
          break;
      }
    } catch (const std::exception &e) {
      // Don't let listener errors break the stream
      // Emit an error event if possible, but don't recurse
      try {
        Error errorEvent(e.what(), handle_);
        for (auto *otherListener : objectListeners_) {
          if (otherListener != listener) {
            try {
              otherListener->onError(errorEvent);
            } catch (...) {
              // Ignore errors in error handlers
            }
          }
        }
        for (auto &funcListener : functionListeners_) {
          try {
            funcListener(errorEvent);
          } catch (...) {
            // Ignore errors in error handlers
          }
        }
      } catch (...) {
        // If we can't even emit the error, just continue
      }
    }
  }

  // Emit to function listeners
  for (auto &listener : functionListeners_) {
    try {
      listener(event);
    } catch (const std::exception &e) {
      // Don't let listener errors break the stream
      try {
        Error errorEvent(e.what(), handle_);
        for (auto *objListener : objectListeners_) {
          try {
            objListener->onError(errorEvent);
          } catch (...) {
            // Ignore errors in error handlers
          }
        }
        for (auto &otherFuncListener : functionListeners_) {
          if (&otherFuncListener != &listener) {
            try {
              otherFuncListener(errorEvent);
            } catch (...) {
              // Ignore errors in error handlers
            }
          }
        }
      } catch (...) {
        // If we can't even emit the error, just continue
      }
    }
  }
}

inline void Stream::emitError(const std::string &errorMessage) {
  Error errorEvent(errorMessage, handle_);
  emit(errorEvent);
}

// Transcriber implementation
inline Transcriber::Transcriber(const std::string &modelPath,
                                ModelArch modelArch, double updateInterval)
    : handle_(-1),
      modelPath_(modelPath),
      modelArch_(modelArch),
      updateInterval_(updateInterval) {
  handle_ = moonshine_load_transcriber_from_files(
      modelPath.c_str(), static_cast<uint32_t>(modelArch), nullptr, 0,
      MOONSHINE_HEADER_VERSION);
  checkError(handle_);
}

inline Transcriber::~Transcriber() { close(); }

inline Transcriber::Transcriber(Transcriber &&other)
    : handle_(other.handle_),
      modelPath_(std::move(other.modelPath_)),
      modelArch_(other.modelArch_),
      updateInterval_(other.updateInterval_),
      defaultStream_(std::move(other.defaultStream_)) {
  other.handle_ = -1;
  if (defaultStream_) {
    defaultStream_->transcriber_ = this;
  }
}

inline Transcriber &Transcriber::operator=(Transcriber &&other) {
  if (this != &other) {
    close();
    handle_ = other.handle_;
    modelPath_ = std::move(other.modelPath_);
    modelArch_ = other.modelArch_;
    updateInterval_ = other.updateInterval_;
    defaultStream_ = std::move(other.defaultStream_);
    other.handle_ = -1;
    if (defaultStream_) {
      defaultStream_->transcriber_ = this;
    }
  }
  return *this;
}

inline void Transcriber::close() {
  defaultStream_.reset();
  if (handle_ >= 0) {
    moonshine_free_transcriber(handle_);
    handle_ = -1;
  }
}

inline Transcript Transcriber::transcribeWithoutStreaming(
    const std::vector<float> &audioData, int32_t sampleRate, uint32_t flags) {
  if (handle_ < 0) {
    throw MoonshineException("Transcriber is not initialized");
  }
  if (audioData.empty()) {
    return Transcript();
  }
  transcript_t *out_transcript = nullptr;
  checkError(moonshine_transcribe_without_streaming(
      handle_, const_cast<float *>(audioData.data()), audioData.size(),
      sampleRate, flags, &out_transcript));
  return parseTranscript(out_transcript);
}

inline int32_t Transcriber::getVersion() const {
  return moonshine_get_version();
}

inline Stream Transcriber::createStream(double updateInterval, uint32_t flags) {
  return Stream(this, updateInterval, flags);
}

inline Stream &Transcriber::getDefaultStream() {
  if (!defaultStream_) {
    defaultStream_.reset(new Stream(this, updateInterval_, 0));
  }
  return *defaultStream_;
}

inline void Transcriber::start() { getDefaultStream().start(); }

inline void Transcriber::stop() {
  if (defaultStream_) {
    defaultStream_->stop();
  }
}

inline void Transcriber::addAudio(const std::vector<float> &audioData,
                                  int32_t sampleRate) {
  getDefaultStream().addAudio(audioData, sampleRate);
}

inline Transcript Transcriber::updateTranscription(uint32_t flags) {
  return getDefaultStream().updateTranscription(flags);
}

inline void Transcriber::addListener(TranscriptEventListener *listener) {
  getDefaultStream().addListener(listener);
}

inline void Transcriber::addListener(
    std::function<void(const TranscriptEvent &)> listener) {
  getDefaultStream().addListener(listener);
}

inline void Transcriber::removeListener(TranscriptEventListener *listener) {
  if (defaultStream_) {
    defaultStream_->removeListener(listener);
  }
}

inline void Transcriber::removeListener(
    std::function<void(const TranscriptEvent &)> listener) {
  if (defaultStream_) {
    defaultStream_->removeListener(listener);
  }
}

inline void Transcriber::removeAllListeners() {
  if (defaultStream_) {
    defaultStream_->removeAllListeners();
  }
}

inline Transcript Transcriber::parseTranscript(
    const transcript_t *transcript_c) {
  return Transcript(transcript_c);
}

inline void Transcriber::checkError(int32_t error) const {
  if (error < 0) {
    const char *errorStr = moonshine_error_to_string(error);
    std::string message = errorStr ? std::string(errorStr) : "Unknown error";
    throw MoonshineException(message);
  }
}

inline void Stream::checkError(int32_t error) const {
  if (error < 0) {
    const char *errorStr = moonshine_error_to_string(error);
    std::string message = errorStr ? std::string(errorStr) : "Unknown error";
    throw MoonshineException(message);
  }
}

// TextToSpeech implementation
inline TextToSpeech::TextToSpeech(
    const std::string &language,
    const std::vector<moonshine_option_t> &options)
    : handle_(-1), language_(language) {
  handle_ = moonshine_create_tts_synthesizer_from_files(
      language.c_str(), nullptr, 0, options.empty() ? nullptr : options.data(),
      options.size(), MOONSHINE_HEADER_VERSION);
  checkError(handle_);
}

inline TextToSpeech::~TextToSpeech() { close(); }

inline TextToSpeech::TextToSpeech(TextToSpeech &&other)
    : handle_(other.handle_), language_(std::move(other.language_)) {
  other.handle_ = -1;
}

inline TextToSpeech &TextToSpeech::operator=(TextToSpeech &&other) {
  if (this != &other) {
    close();
    handle_ = other.handle_;
    language_ = std::move(other.language_);
    other.handle_ = -1;
  }
  return *this;
}

inline TtsSynthesisResult TextToSpeech::synthesize(
    const std::string &text,
    const std::vector<moonshine_option_t> &options) {
  if (handle_ < 0) {
    throw MoonshineException("TextToSpeech is not initialized");
  }
  float *out_audio = nullptr;
  uint64_t out_size = 0;
  int32_t out_sample_rate = 0;
  checkError(moonshine_text_to_speech(
      handle_, text.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out_audio, &out_size, &out_sample_rate));
  TtsSynthesisResult result;
  if (out_audio && out_size > 0) {
    result.samples.assign(out_audio, out_audio + out_size);
    std::free(out_audio);
  }
  result.sampleRateHz = out_sample_rate;
  return result;
}

inline void TextToSpeech::close() {
  if (handle_ >= 0) {
    moonshine_free_tts_synthesizer(handle_);
    handle_ = -1;
  }
}

inline std::string TextToSpeech::getVoices(
    const std::string &languages,
    const std::vector<moonshine_option_t> &options) {
  char *out_json = nullptr;
  int32_t err = moonshine_get_tts_voices(
      languages.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out_json);
  if (err < 0) {
    const char *errorStr = moonshine_error_to_string(err);
    throw MoonshineException(errorStr ? std::string(errorStr)
                                      : "Unknown error");
  }
  std::string result;
  if (out_json) {
    result = std::string(out_json);
    std::free(out_json);
  }
  return result;
}

inline std::string TextToSpeech::getDependencies(
    const std::string &languages,
    const std::vector<moonshine_option_t> &options) {
  char *out_json = nullptr;
  int32_t err = moonshine_get_tts_dependencies(
      languages.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out_json);
  if (err < 0) {
    const char *errorStr = moonshine_error_to_string(err);
    throw MoonshineException(errorStr ? std::string(errorStr)
                                      : "Unknown error");
  }
  std::string result;
  if (out_json) {
    result = std::string(out_json);
    std::free(out_json);
  }
  return result;
}

inline void TextToSpeech::checkError(int32_t error) const {
  if (error < 0) {
    const char *errorStr = moonshine_error_to_string(error);
    std::string message = errorStr ? std::string(errorStr) : "Unknown error";
    throw MoonshineException(message);
  }
}

// GraphemeToPhonemizer implementation
inline GraphemeToPhonemizer::GraphemeToPhonemizer(
    const std::string &language,
    const std::vector<moonshine_option_t> &options)
    : handle_(-1), language_(language) {
  handle_ = moonshine_create_grapheme_to_phonemizer_from_files(
      language.c_str(), nullptr, 0, options.empty() ? nullptr : options.data(),
      options.size(), MOONSHINE_HEADER_VERSION);
  checkError(handle_);
}

inline GraphemeToPhonemizer::~GraphemeToPhonemizer() { close(); }

inline GraphemeToPhonemizer::GraphemeToPhonemizer(
    GraphemeToPhonemizer &&other)
    : handle_(other.handle_), language_(std::move(other.language_)) {
  other.handle_ = -1;
}

inline GraphemeToPhonemizer &GraphemeToPhonemizer::operator=(
    GraphemeToPhonemizer &&other) {
  if (this != &other) {
    close();
    handle_ = other.handle_;
    language_ = std::move(other.language_);
    other.handle_ = -1;
  }
  return *this;
}

inline std::string GraphemeToPhonemizer::toIpa(
    const std::string &text,
    const std::vector<moonshine_option_t> &options) {
  if (handle_ < 0) {
    throw MoonshineException("GraphemeToPhonemizer is not initialized");
  }
  const char *out_phonemes = nullptr;
  uint64_t out_count = 0;
  checkError(moonshine_text_to_phonemes(
      handle_, text.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out_phonemes, &out_count));
  if (out_phonemes && out_count > 0) {
    return std::string(out_phonemes, out_count);
  }
  return std::string();
}

inline void GraphemeToPhonemizer::close() {
  if (handle_ >= 0) {
    moonshine_free_grapheme_to_phonemizer(handle_);
    handle_ = -1;
  }
}

inline std::string GraphemeToPhonemizer::getDependencies(
    const std::string &languages,
    const std::vector<moonshine_option_t> &options) {
  char *out_deps = nullptr;
  int32_t err = moonshine_get_g2p_dependencies(
      languages.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out_deps);
  if (err < 0) {
    const char *errorStr = moonshine_error_to_string(err);
    throw MoonshineException(errorStr ? std::string(errorStr)
                                      : "Unknown error");
  }
  std::string result;
  if (out_deps) {
    result = std::string(out_deps);
    std::free(out_deps);
  }
  return result;
}

inline void GraphemeToPhonemizer::checkError(int32_t error) const {
  if (error < 0) {
    const char *errorStr = moonshine_error_to_string(error);
    std::string message = errorStr ? std::string(errorStr) : "Unknown error";
    throw MoonshineException(message);
  }
}

inline IntentRecognizer::IntentRecognizer(const std::string &model_path,
                                           EmbeddingModelArch arch,
                                           const std::string &model_variant)
    : handle_(-1) {
  handle_ = moonshine_create_intent_recognizer(
      model_path.c_str(), static_cast<uint32_t>(arch), model_variant.c_str());
  checkError(handle_);
}

inline IntentRecognizer::~IntentRecognizer() { close(); }

inline IntentRecognizer::IntentRecognizer(IntentRecognizer &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = -1;
}

inline IntentRecognizer &IntentRecognizer::operator=(
    IntentRecognizer &&other) noexcept {
  if (this != &other) {
    close();
    handle_ = other.handle_;
    other.handle_ = -1;
  }
  return *this;
}

inline void IntentRecognizer::registerIntent(
    const std::string &canonical_phrase) {
  checkError(
      moonshine_register_intent(handle_, canonical_phrase.c_str()));
}

inline bool IntentRecognizer::unregisterIntent(
    const std::string &canonical_phrase) {
  int32_t err =
      moonshine_unregister_intent(handle_, canonical_phrase.c_str());
  if (err == MOONSHINE_ERROR_NONE) {
    return true;
  }
  if (err == MOONSHINE_ERROR_INVALID_ARGUMENT) {
    return false;
  }
  checkError(err);
  return false;
}

inline std::vector<IntentMatch> IntentRecognizer::getClosestIntents(
    const std::string &utterance, float tolerance_threshold) {
  std::vector<IntentMatch> out;
  moonshine_intent_match_t *matches = nullptr;
  uint64_t count = 0;
  int32_t err = moonshine_get_closest_intents(
      handle_, utterance.c_str(), tolerance_threshold, &matches, &count);
  if (err != MOONSHINE_ERROR_NONE) {
    if (matches != nullptr) {
      moonshine_free_intent_matches(matches, count);
    }
    checkError(err);
    return out;
  }
  out.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    const char *ph = matches[i].canonical_phrase;
    out.emplace_back(ph ? std::string(ph) : std::string(),
                     matches[i].similarity);
  }
  moonshine_free_intent_matches(matches, count);
  return out;
}

inline int32_t IntentRecognizer::intentCount() const {
  int32_t n = moonshine_get_intent_count(handle_);
  if (n < 0) {
    const char *errorStr = moonshine_error_to_string(n);
    throw MoonshineException(errorStr ? std::string(errorStr)
                                      : "Unknown error");
  }
  return n;
}

inline void IntentRecognizer::clearIntents() {
  checkError(moonshine_clear_intents(handle_));
}

inline void IntentRecognizer::close() {
  if (handle_ >= 0) {
    moonshine_free_intent_recognizer(handle_);
    handle_ = -1;
  }
}

inline void IntentRecognizer::checkError(int32_t error) const {
  if (error < 0) {
    const char *errorStr = moonshine_error_to_string(error);
    std::string message = errorStr ? std::string(errorStr) : "Unknown error";
    throw MoonshineException(message);
  }
}

}  // namespace moonshine

#endif  // MOONSHINE_CPP_H
