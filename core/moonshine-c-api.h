#ifndef MOONSHINE_C_API_H
#define MOONSHINE_C_API_H

/* Moonshine is a library for building interactive voice applications. It
   provides a high-level API for building voice interfaces, including
   voice-activity detection, diarization, transcription, speech understanding,
   and text-to-speech. It is designed to be fast, easy to use and to provide a
   high level of accuracy. It is also designed to be easy to integrate into your
   existing codebase across all major platforms.

   It uses the Moonshine family of speech to text models, which:

     - Understand multiple major languages, including English, Japanese,
       Korean, Chinese, Arabic, and more.

     - Are designed to be lightweight and fast for mobile and edge devices,
       and can be used in the cloud where latency and compute costs matter.

     - Support streaming transcription to reduce latency on real-time
       applications.

     - Are trained from scratch on a large, unique dataset of audio data,
       allowing our team to quickly train custom models for jargon or dialects.

     - Are available under permissive licenses, with English fully MIT
       licensed and other languages under a non-commercial agreement.

   You'll most likely want to use the specific bindings for your language of
   choice, since this is a low-level C API to the underlying implementation.
   This is the interface that those bindings all use though, so if you're
   interested in porting to a new environment or language, the inline notes
   here may be useful.

   Here's an example of how to use the transcriber:
   ```c
   #include "moonshine-c-api.h"

   int main(int argc, char *argv[]) {
     int32_t transcriber_handle = moonshine_load_transcriber_from_files(
       "path/to/models", MOONSHINE_MODEL_ARCH_BASE, NULL, 0,
       MOONSHINE_HEADER_VERSION);
     if (transcriber_handle < 0) {
       fprintf(stderr, "Failed to load transcriber\n");
       return 1;
     }

     float audio_data[32000] = {};
     size_t audio_length = 32000;
     int32_t sample_rate = 16000;
     transcript_t *transcript = NULL;
     int32_t error = moonshine_transcribe_without_streaming(transcriber_handle,
   audio_data, audio_length, sample_rate, 0, &transcript); if (error != 0) {
       fprintf(stderr, "Failed to transcribe\n");
       return 1;
     }
     for (size_t i = 0; i < transcript->line_count; i++) {
       printf( "Line %zu at %f seconds: %s\n", i, transcript->lines[i].start,
         transcript->lines[i].text);
     }
     moonshine_free_transcriber(transcriber_handle);
     return 0;
   }

   All API calls are thread-safe, so you can call them from multiple threads
   concurrently. Calculations on a single transcriber will be serialized
   however, so latency will be affected for calls from other threads while
   the transcriber is busy.
   ```
*/

#if defined(ANDROID)
#include <android/asset_manager.h>
#endif
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define MOONSHINE_EXPORT __declspec(dllexport)
#else
#define MOONSHINE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------ CONSTANTS -------------------------------- */

/* What version of the Moonshine library the header file is associated with.
   You should pass this version to moonshine_load_transcriber so that newer
   versions of the library can emulate any older behavior that has changed.
   The format is MAJOR * 10000 + MINOR * 100 + PATCH.
   For example, version 2.0.0 would be 20000.
   For example, version 2.3.7 would be 20307.                                */
#define MOONSHINE_HEADER_VERSION (20000)

/* Supported model architectures.                                            */
#define MOONSHINE_MODEL_ARCH_TINY (0)
#define MOONSHINE_MODEL_ARCH_BASE (1)
#define MOONSHINE_MODEL_ARCH_TINY_STREAMING (2)
#define MOONSHINE_MODEL_ARCH_BASE_STREAMING (3)
#define MOONSHINE_MODEL_ARCH_SMALL_STREAMING (4)
#define MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING (5)

/* Error codes.                                                            */
#define MOONSHINE_ERROR_NONE (0)
#define MOONSHINE_ERROR_UNKNOWN (-1)
#define MOONSHINE_ERROR_INVALID_HANDLE (-2)
#define MOONSHINE_ERROR_INVALID_ARGUMENT (-3)

/* Flags.                                                                */
#define MOONSHINE_FLAG_FORCE_UPDATE (1 << 0)
/* Apply alphanumeric-spelling fusion to every completed line in the
   returned transcript. The transcriber must have been constructed with
   a spelling model (either ``spelling_model_path`` in
   moonshine_load_transcriber_from_files or a non-null
   ``spelling_model_data`` buffer in
   moonshine_load_transcriber_from_memory) for this flag to have any
   effect; if no spelling model is loaded, the flag is ignored.

   When fusion fires for a line, the line's ``text`` field is *replaced*
   with the resolved single character (e.g. ``"a"`` or ``"$"``). Speech
   that does not resolve to a character is left unchanged so command
   words like "stop" / "clear" / "delete" can still be classified by
   higher-level Python code. */
#define MOONSHINE_FLAG_SPELLING_MODE (1 << 1)

/* --------------------------- DATA STRUCTURES ----------------------------- */

/* Values passed to moonshine_load_transcriber,
   moonshine_create_text_to_speech_synthesizer or
   moonshine_create_graph_to_phonemizer at creation time that control
   the behavior of the transcriber. A typical use case would be to specify
   model configuration options like layer names that vary by language. The
   value is a string. You don't normally need to care about these, this is just
   for advanced customizations.                                              */
struct moonshine_option_t {
  const char *name;
  const char *value;
};

/* All transcription calls return a list of "lines". These line objects
represent a piece of speech, something like a sentence or phrase. For
non-streaming calls, you get back a finalized list of these lines, with all
their states set to “complete”. Each streaming call returns a similar list, but
if there isn’t a pause at the end of the current audio - if the user still
seems to be speaking but cut off - the final line will be marked as being
incomplete.

All memory referenced by the line objects is owned by the transcriber and is
valid until the next call to that transcriber, or until the transcriber is
freed.

The audio data is 16KHz float PCM, between -1.0 and 1.0.

To make the streaming results easier to work with we offer some guarantees:

 - Lines are never removed from the results, only added.

 - Only the last line in the list may potentially be incomplete.

 - If speech is detected by the VAD, but no transcription can be produced, the
   line will be an empty string, "".

 - Line indexes can be used as stable references when repeatedly calling
   streaming transcription. This means a client can remember the length of the
   last results returned, and when it calls again it can figure out the updates
   by iterating the results starting at that line index.

 - The line id is a stable identifier for the line. This is set to a 64-bit
   randomly-generated number, with the goal of minimizing the chances of a
   collision. Currently these IDs are in ascending order in any one transcript,
   but this is not guaranteed and should not be relied on.

 - When speaker identification is enabled (the opt-in ``identify_speakers``
   option), each line carries an array of speaker spans describing who was
   talking during which parts of the line, including UTF-8 character ranges
   into the line text. Word timestamps are enabled automatically in this mode.
   Speaker IDs are 64-bit
   randomly-generated numbers that are stable for a given speaker within a
   stream, and speaker indices count speakers in order of first appearance.
   Unlike the text and timing of a line, speaker spans for recent audio are
   *mutable*: streaming diarization re-clusters a sliding window
   (``diarization_cluster_window_sec``, default 120s) as more speech arrives;
   assignments for older audio are frozen. The ``have_speakers_changed`` flag is
   set on a line whenever its spans changed since the previous call.

See the stream transcription examples below for more details on what this
means in practice.
*/

/* A single word with timing information.
   Only populated when word_timestamps option is enabled. */
struct transcript_word_t {
  /* UTF-8-encoded word text. */
  const char *text;
  /* Start time in seconds (absolute, from start of audio/stream). */
  float start;
  /* End time in seconds. */
  float end;
  /* Model confidence score, 0.0 to 1.0. */
  float confidence;
};

/* One contiguous span of speech within a line attributed to a single
   speaker. Only populated when the identify_speakers option is enabled.
   Spans can be revised on any transcription call, even for lines that are
   already complete; see the have_speakers_changed flag on
   transcript_line_t. Character ranges use UTF-8 byte offsets into the line
   text; word_timestamps are enabled automatically when identify_speakers is
   on. */
struct speaker_span_t {
  /* Time offset from the start of the array or stream in seconds. */
  float start_time;
  /* Length of the span in seconds. */
  float duration;
  /* Stable identifier for the speaker within this stream. */
  uint64_t speaker_id;
  /* The order the speaker first appeared in the transcript, starting at 0. */
  uint32_t speaker_index;
  /* UTF-8 byte offset into the line's text where this span begins (inclusive).
     Only meaningful when identify_speakers is enabled; word_timestamps are
     turned on automatically in that case. Both zero when unknown. */
  uint64_t start_char;
  /* UTF-8 byte offset into the line's text where this span ends (exclusive).
     Both zero when unknown. */
  uint64_t end_char;
};

/* Information about a single “line” of a transcript. */
struct transcript_line_t {
  /* UTF-8-encoded transcription. */
  const char *text;
  /* The audio data for the current phrase. */
  const float *audio_data;
  /* The number of elements in the audio data array. */
  size_t audio_data_count;
  /* Time offset from the start of the array or stream in seconds.  */
  float start_time;
  /* How long the segment currently is in seconds. */
  float duration;
  /* Stable identifier for the line. */
  uint64_t id;
  /* Streaming-only: Zero means the speaker hasn't finished talking in this
   * segment, non-zero means they have. */
  int8_t is_complete;
  /* Streaming-only: Whether the line has been updated since the previous call
   * to transcribe_stream_chunk. */
  int8_t is_updated;
  /* Streaming-only: Whether the line was newly added since the previous call to
   * transcribe_stream_chunk. */
  int8_t is_new;
  /* Streaming-only: Whether the text of the line has changed since the previous
   * call to transcribe_stream_chunk. */
  int8_t has_text_changed;
  /* Whether the speaker spans of the line have changed since the previous
   * call to transcribe_stream_chunk. Unlike the other change flags, this can
   * fire for lines that are already complete, since diarization refines
   * speaker assignments retroactively as more audio arrives. */
  int8_t have_speakers_changed;
  /* Speaker spans covering this line, ordered by start time and clipped to
   * the line's time range. NULL unless the identify_speakers option is
   * enabled and speech has been attributed to a speaker. */
  const struct speaker_span_t *speaker_spans;
  /* Number of entries in the speaker_spans array. */
  uint64_t speaker_span_count;
  /* Streaming-only: The latency of the last transcription in milliseconds. */
  uint32_t last_transcription_latency_ms;
  /* Word-level timestamps. NULL if word_timestamps option is not enabled. */
  const struct transcript_word_t *words;
  /* Number of words in the words array. 0 if not enabled. */
  uint64_t word_count;
};

/* An entire transcription of an audio data array or stream.                 */
struct transcript_t {
  struct transcript_line_t *lines; /* All lines of the transcript. */
  uint64_t line_count;             /* Number of lines in the transcript.      */
};

/* ------------------------------ FUNCTIONS -------------------------------- */

/* Returns the loaded moonshine library version. This may be different from
   the header version if a newer shared library is loaded.
*/
MOONSHINE_EXPORT int32_t moonshine_get_version(void);

/* Converts an error code number returned from an API call into a
   human-readable string. */
MOONSHINE_EXPORT const char *moonshine_error_to_string(int32_t error);

/* Frees a buffer that a moonshine_* function documented as "allocated with
   malloc; release with free" returned to the caller. This covers, for
   example, ``out_audio_data`` from moonshine_text_to_speech /
   moonshine_phonemes_to_speech, the JSON / comma-separated strings from
   moonshine_get_tts_dependencies / moonshine_get_g2p_dependencies /
   moonshine_get_tts_voices, and ``out_phonemes`` from
   moonshine_text_to_phonemes.

   Always use this instead of the C runtime ``free`` directly. On Windows the
   library and its host (e.g. a Python binding) can be linked against
   different C runtimes with independent heaps, so freeing a library-allocated
   pointer with the host's ``free`` corrupts the heap. Routing the free back
   through the library guarantees the allocation and deallocation happen in
   the same runtime. Safe to call on NULL. */
MOONSHINE_EXPORT void moonshine_free_buffer(void *ptr);

/* Converts a transcript_t struct into a human-readable string for debugging
 * purposes. The string is owned by the library, and is valid until the next
 * call to moonshine_transcript_to_string. */
MOONSHINE_EXPORT const char *moonshine_transcript_to_string(
    const struct transcript_t *transcript);

/* Loads models from the file system, using `path` as the root directory. The
   implementation expects the following files to be present in the directory:
   - encoder_model.ort
   - decoder_model_merged.ort
   - tokenizer.bin
   The .ort files are quantized activation ONNX models that have been converted
   to ORT format using the onnxruntime tools. The simplest way to obtain these
   files is to run the `scripts/download-moonshine-model.py` script, for
   example `python scripts/download-moonshine-model.py --model-type base
   --model-language en`.
   The source weights are available on the Hugging Face Model Hub at
   https://huggingface.co/UsefulSensors/, and the download and conversion to
   ONNX script is available in this repository at
   `scripts/convert-moonshine-model.sh`.
   The tokenizer.bin contains the token to character mapping for the model,
   in a compact binary format. The `scripts/json-to-bin-vocab.py` can be used
   to convert common tokenizer.json files to tokenizer.bin files.

   The `model_arch` parameter is used to select the model architecture, for
   example MOONSHINE_MODEL_ARCH_BASE or MOONSHINE_MODEL_ARCH_TINY_STREAMING.

   The `options` parameter is used to set any custom options for the
   transcriber. Recognized options include ``log_ort_run`` (bool),
   ``ort_providers`` (comma-separated execution provider names such as
   ``CoreML,CPU`` on macOS or ``NNAPI,CPU`` on Android; default is CPU-only),
   and ``coreml_cache_dir`` (directory for CoreML compiled model cache).
   Pass ``identify_speakers`` (bool, default false) to enable speaker
   diarization: each line then carries a ``speaker_spans`` array describing
   who spoke when, including UTF-8 character ranges into the line text.
   This also enables word timestamps automatically. This runs the cpp-annote
   diarization pipeline (a port of
   pyannote community-1) inline inside transcription calls, which adds
   significant compute, and re-clustering cost grows with session length unless
   bounded by ``diarization_cluster_window_sec``.
   ``diarization_cluster_cadence`` (float seconds, default 2.0) sets the
   minimum interval between re-clustering passes - raise it to reduce cost on
   long sessions - ``diarization_analyze_cadence`` (float seconds,
   default 0 = model default of 1.0) sets the interval between
   segmentation/embedding model runs, and ``diarization_cluster_window_sec``
   (float seconds, default 120.0) limits how much audio history VBx
   re-clustering considers on each refresh (0 = unlimited full history).
   Pass ``"spelling_model_path"`` with a path to a
   spelling-CNN ``.ort`` file (e.g.
   ``https://download.moonshine.ai/model/spelling-en/spelling_cnn.ort``)
   to enable alphanumeric spelling fusion via
   ``MOONSHINE_FLAG_SPELLING_MODE``; if not set, the spelling model is
   not loaded and the flag is a no-op.

   The `options_count` parameter is the number of options in the options array.

   The `moonshine_version` parameter should be set to MOONSHINE_HEADER_VERSION
   to ensure that if a newer version of the library is loaded, it emulates the
   behavior of the older version to ensure compatibility.

   The return value is a handle to a transcriber, which can be used to identify
   the transcriber in subsequent calls. If there was an error, a negative value
   is returned. This code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_load_transcriber_from_files(
    const char *path, uint32_t model_arch,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version);

/* Loads models from memory. The `encoder_model_data`, `decoder_model_data` and
   `tokenizer_data` parameters are the data arrays for the models in binary
   format, and are expected to be in the same format as the files disk.

   `spelling_model_data` and `spelling_model_data_size` are an optional
   in-memory ``.ort`` payload for the alphanumeric spelling-CNN. Pass
   ``NULL`` and ``0`` if you don't want spelling fusion. When provided,
   the buffer must outlive the transcriber (it is *not* copied) and the
   transcriber will run spelling fusion whenever
   ``MOONSHINE_FLAG_SPELLING_MODE`` is passed to
   ``moonshine_transcribe_stream`` or
   ``moonshine_transcribe_without_streaming``.

   All of the other parameters are the same as for
   moonshine_load_transcriber_from_files.                                    */
MOONSHINE_EXPORT int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    const uint8_t *spelling_model_data, size_t spelling_model_data_size,
    uint32_t model_arch, const struct moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version);

/* Releases all resources used by the transcriber. Subsequent transcriber
   creation calls may reuse this transcriber's ID, so ensure you remove
   all references to it in your client code after freeing it.*/
MOONSHINE_EXPORT void moonshine_free_transcriber(int32_t transcriber_handle);

/* Given an array of PCM audio data, identifies sections of speech and
   transcribes them into text. This is the call to use if you're analyzing audio
   from a file or other static source where you have all the audio data at once.
   If you are transcribing audio from a live microphone or other real-time
   source, you should use the streaming API instead, since it offers lower
   latency for those use cases.

   `transcriber_handle` should be a handle to a transcriber returned by
    moonshine_load_transcriber_from_files or
    moonshine_load_transcriber_from_memory.

   `audio_data` should be a pointer to an array of PCM audio data, between -1.0
    and 1.0, at a sample rate of `sample_rate` Hz. Internally the library uses
    16,000 Hz, so to avoid resampling you should capture audio at this rate if
    possible.

   `audio_length` should be the number of samples in the audio data array.

   `sample_rate` should be the sample rate of the audio data, in Hz.
   `flags` should be a bitwise OR of flags. Currently the only supported flag
   is MOONSHINE_FLAG_SPELLING_MODE, which applies alphanumeric-spelling fusion
   to completed lines (requires the transcriber to have been loaded with a
   spelling model; otherwise the flag is a no-op). Pass zero for the default
   behavior.

   `out_transcript` should be a pointer to a pointer to a transcript_t struct.
   The transcript_t struct will be populated with the transcript data, which
   consists of a list of lines, each with text, audio data, and timestamps.
   This data is owned by the transcriber and is valid until the next call to
   that transcriber, or until the transcriber is freed.

   The return value is zero on success, or a non-zero error code on failure.
   The error code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_transcribe_without_streaming(
    int32_t transcriber_handle, float *audio_data, uint64_t audio_length,
    int32_t sample_rate, uint32_t flags, struct transcript_t **out_transcript);

/* Streaming allows the library to incrementally return updated results as
   new audio data becomes available in real-time. This approach allows us to
   produce results with lower latency than non-streaming approaches, by
   reusing calculations done on earlier audio data.

   The `transcriber_handle` should be a handle to a transcriber returned by
   moonshine_load_transcriber_from_files or
   moonshine_load_transcriber_from_memory. A single transcriber can have
   multiple streams associated with it, and each stream can be used to
   transcribe a separate audio stream.

   The `flags` should be a bitwise OR of flags. None are currently supported so
   this should always be zero.

   The return value is a handle to a stream, which can be used to identify the
   stream in subsequent calls. If there was an error, a negative value is
   returned. The error code can be converted to a human-readable string using
   moonshine_error_to_string.

   Below is some pseudocode showing an example of how to use streaming. In a
   real application you'll want to check the return value of the functions and
   handle errors appropriately. You can see a more complete example in the
   moonshine-test-v2.cpp file.

   ```c
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        "path/to/models", MOONSHINE_MODEL_ARCH_BASE, NULL, 0,
        MOONSHINE_HEADER_VERSION);
    int32_t stream_handle = moonshine_create_stream(transcriber_handle, 0);
    moonshine_start_stream(transcriber_handle, stream_handle);

    float* latest_audio_data;
    size_t latest_audio_data_length;
    while (get_audio_from_microphone(&latest_audio_data,
      &latest_audio_data_length)) {
      moonshine_transcribe_add_audio_to_stream(transcriber_handle,
        stream_handle, latest_audio_data, latest_audio_data_length,
       microphone_sample_rate, 0);
      if (time_since_last_transcription < min_time_between_transcriptions) {
        continue;
      }
      transcript_t *partial_transcript = NULL;
      moonshine_transcribe_stream(transcriber_handle,
        stream_handle, 0, &partial_transcript);
      print_transcript(out_transcript);
    }
    moonshine_stop_stream(transcriber_handle, stream_handle);

    transcript_t *final_transcript = NULL;
    moonshine_transcribe_stream(transcriber_handle, stream_handle, 0,
      &final_transcript);
    print_transcript(final_transcript);

    moonshine_free_stream(transcriber_handle, stream_handle);
    moonshine_free_transcriber(transcriber_handle);
    ```

   The transcripts that are returned consist of a list of lines, each with
   text, audio data, timestamp, duration, and other metadata. This metadata
   includes an `is_updated` flag, which is set to 1 if the line has been updated
   since the last call to moonshine_transcribe_stream. You can use this as a
   "dirty flag" to determine how to update your UI in a minimal way, touching
   only the elements that have changed. Updated lines only appear at the end of
   the list of lines, and once the `is_complete` flag is set to 1 for a line,
   its text and timing will never change again.

   The one exception is speaker information: when the `identify_speakers`
   option is enabled, speaker spans for recent audio can be revised on any call
   to moonshine_transcribe_stream, since diarization re-clusters a sliding
   window of recent speech. Older assignments are frozen. Watch the
   `have_speakers_changed` flag to detect these revisions.
*/

/* Creates a stream. This function returns a handle to the stream, which can be
   used to identify the stream in subsequent calls. If there was an error, a
   negative value is returned. The error code can be converted to a
   human-readable string using moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_create_stream(int32_t transcriber_handle,
                                                 uint32_t flags);

/* Releases the resources used by a stream.
   Subsequent stream creation calls may reuse this stream's ID, so ensure you
   remove all references to it in your client code after freeing it.*/
MOONSHINE_EXPORT int32_t moonshine_free_stream(int32_t transcriber_handle,
                                               int32_t stream_handle);

/* Starts a stream. This should be called before any calls to
   moonshine_transcribe_stream_chunk. Start/stop are supported because there may
   sometimes be a discontinuity in the audio input, for example when the user
   mutes their input, so we need a way to start fresh after a break like this.
   This function returns zero on success, or a non-zero error code on failure.
   The error code can be converted to a human-readable string using
   moonshine_error_to_string.
 */
MOONSHINE_EXPORT int32_t moonshine_start_stream(int32_t transcriber_handle,
                                                int32_t stream_handle);

/* Stops a stream. This function returns zero on success, or a non-zero error
   code on failure. The error code can be converted to a human-readable string
   using moonshine_error_to_string.
 */
MOONSHINE_EXPORT int32_t moonshine_stop_stream(int32_t transcriber_handle,
                                               int32_t stream_handle);

/* Call this when new audio data becomes available from your microphone or other
   audio source. This function will add the audio data to the stream's buffer,
   but it will not transcribe it or do any other processing, so this should be
   safe to call frequently even from time-critical threads. The size of the
   input audio doesn't have any impact on performance, so you should call this
   with whatever the natural chunk size is for your audio source. It is up to
   you to call moonshine_transcribe_stream when you want an updated transcript,
   the frequency of which should be determined by your application's latency and
   compute budgets.

   `transcriber_handle` should be a handle to a transcriber returned by
   moonshine_load_transcriber_from_files or
   moonshine_load_transcriber_from_memory.

   `stream_handle` should be a handle to a stream returned by
   moonshine_create_stream.

   `new_audio_data` should be a pointer to an array of PCM audio data, between
   -1.0 and 1.0, at a sample rate of `sample_rate` Hz. `audio_length` should be
   the number of samples in the audio data array.

   `sample_rate` should be the sample rate of the audio data, in Hz.

   `flags` should be a bitwise OR of flags. None are currently supported so
   this should always be zero.

   The return value is zero on success, or a non-zero error code on failure.
   The error code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_transcribe_add_audio_to_stream(
    int32_t transcriber_handle, int32_t stream_handle,
    const float *new_audio_data, uint64_t audio_length, int32_t sample_rate,
    uint32_t flags);

/* Analyzes all the audio data in the stream and returns an updated transcript
   of all the speech segments found. By default this function will only perform
   full analysis on the audio data if there has been more than 200ms of new
   samples since the last complete analysis. This is to ensure that too-frequent
   calls to this function don't result in poor performance. This can be
   overridden by setting the MOONSHINE_FLAG_FORCE_UPDATE flag.

   `transcriber_handle` should be a handle to a transcriber returned by
   moonshine_load_transcriber_from_files or
   moonshine_load_transcriber_from_memory.

   `stream_handle` should be a handle to a stream returned by
   moonshine_create_stream.

   `flags` should be a bitwise OR of flags. Currently the only supported flag is
   MOONSHINE_FLAG_FORCE_UPDATE, which ignores the time-based caching logic to
   ensure the stream is fully analyzed by the models.

   `out_transcript` should be a pointer to a pointer to a transcript_t struct.
   The transcript_t struct will be populated with the transcript data, which
   consists of a list of lines, each with text, audio data, and timestamps.
   This data is owned by the transcriber and is valid until the next call to
   that transcriber, or until the transcriber is freed.

   The return value is zero on success, or a non-zero error code on failure.
   The error code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_transcribe_stream(
    int32_t transcriber_handle, int32_t stream_handle, uint32_t flags,
    struct transcript_t **out_transcript);

/* ------------------------------ INTENT RECOGNIZER ------------------------- */

/* Supported embedding model architectures for intent recognition.           */
#define MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M (0)

/* Maximum number of intent matches returned by moonshine_get_closest_intents.
 */
#define MOONSHINE_INTENT_MAX_MATCHES (6)

/* One ranked intent match from moonshine_get_closest_intents. */
struct moonshine_intent_match_t {
  char *canonical_phrase;
  float similarity;
};

/* Creates an intent recognizer from files on disk.

   `model_path` should be the path to the directory containing the embedding
   model files (ONNX model and tokenizer.bin).

   `model_arch` should be one of the MOONSHINE_EMBEDDING_MODEL_ARCH_* constants.
   Currently only MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M is supported.

   `model_variant` specifies which model variant to load: "fp32", "fp16", "q8",
   "q4", or "q4f16". Pass NULL to use the default "q4" variant.

   Similarity filtering is done per call in moonshine_get_closest_intents via
   `tolerance_threshold`, not at construction time.

   Returns a non-negative handle on success, or a negative error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t moonshine_create_intent_recognizer(
    const char *model_path, uint32_t model_arch, const char *model_variant);

/* Frees an intent recognizer and all its resources. */
MOONSHINE_EXPORT void moonshine_free_intent_recognizer(
    int32_t intent_recognizer_handle);

/* Registers a canonical intent phrase (no callback).

   `embedding` is an optional pointer to an array of floats of size
   `embedding_size`. If `embedding` is NULL, the embedding is calculated for the
   canonical phrase.
   `priority` affects how intents are ranked. If a higher priority intent is
   within the tolerance threshold, it will be ranked above lower priority
   intents, even if their similarity is higher.

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_register_intent(
    int32_t intent_recognizer_handle, const char *canonical_phrase,
    float *embedding, uint64_t embedding_size, int32_t priority);

/* Unregisters an intent by its canonical phrase.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_unregister_intent(
    int32_t intent_recognizer_handle, const char *canonical_phrase);

/* Synchronously ranks registered intents against `utterance`.

   `tolerance_threshold` is the minimum similarity (0.0–1.0, inclusive) for a
   candidate to appear in the results.

   On success, returns MOONSHINE_ERROR_NONE, sets `*out_count` to the number
   of matches (0 to MOONSHINE_INTENT_MAX_MATCHES), and sets `*out_matches` to a
   heap-allocated array sorted by descending similarity. Each
   `canonical_phrase` is a separate heap allocation. When `*out_count` is zero,
   `*out_matches` is set to NULL.

   On failure, returns a non-zero error code and sets `*out_matches` to NULL and
   `*out_count` to zero.

   Release results with moonshine_free_intent_matches.
*/
MOONSHINE_EXPORT int32_t moonshine_get_closest_intents(
    int32_t intent_recognizer_handle, const char *utterance,
    float tolerance_threshold, struct moonshine_intent_match_t **out_matches,
    uint64_t *out_count);

/* Frees an array returned by moonshine_get_closest_intents (safe on NULL /
   zero count). */
MOONSHINE_EXPORT void moonshine_free_intent_matches(
    struct moonshine_intent_match_t *matches, uint64_t count);

/* Gets the number of registered intents.
   Returns the count on success (>= 0), or a negative error code on failure.
*/
MOONSHINE_EXPORT int32_t
moonshine_get_intent_count(int32_t intent_recognizer_handle);

/* Clears all registered intents.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t
moonshine_clear_intents(int32_t intent_recognizer_handle);

/* Calculates the intent embedding for a given sentence.

   On success, ``*out_embedding`` is set to a heap-allocated array of floats and
   ``*out_embedding_size`` is set to the number of elements. Release the array
   with ``moonshine_free_intent_embedding``.

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_calculate_intent_embedding(
    int32_t intent_recognizer_handle, const char *sentence,
    float **out_embedding, uint64_t *out_embedding_size,
    const char *model_name);

/* Frees an intent embedding returned by moonshine_calculate_intent_embedding.
 */
MOONSHINE_EXPORT void moonshine_free_intent_embedding(float *embedding);

/* Calculates the cosine similarity between two embedding vectors.

   Both ``embedding_a`` and ``embedding_b`` must have ``embedding_size``
   elements.  The result is written to ``*out_similarity`` and is in the
   range [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite).

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_calculate_embedding_distance(
    int32_t intent_recognizer_handle, const float *embedding_a,
    const float *embedding_b, uint64_t embedding_size, float *out_similarity);

/* ------------------------------ TEXT TO SPEECH ------------------------- */

/* Creates a text to speech synthesizer from files on disk.
   Returns a non-negative handle on success, or a non-zero error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.
   Pass option ``voice`` as ``kokoro_<id>`` or ``piper_<stem>`` to select the
   vocoder, or as a bare Kokoro id / Piper stem when using the default auto
   choice (and other TTS paths via ``moonshine_option_t`` as documented for
   ``MoonshineTTSOptions``). ``engine`` / ``vocoder_engine`` options are
   ignored.

   ZipVoice (zero-shot voice cloning) is selected with ``voice`` =
   ``zipvoice_<id>`` for a built-in VCTK reference voice (e.g.
   ``zipvoice_american_female``, ``zipvoice_indian_male``), or a bare
   ``zipvoice`` together with a caller-supplied reference clip via
   ``moonshine_create_tts_synthesizer_from_memory`` (key
   ``zipvoice/clone_audio``). ZipVoice model assets
   (``zipvoice/text_encoder.ort``, ``zipvoice/fm_decoder.ort``,
   ``zipvoice/vocoder.ort``, ``zipvoice/tokens.txt``,
   ``zipvoice/model.json``) are resolved under ``g2p_root`` or supplied in
   memory. English only for now.
*/
MOONSHINE_EXPORT int32_t moonshine_create_tts_synthesizer_from_files(
    const char *language, const char **filenames, uint64_t filenames_count,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version);

/* Creates a text to speech synthesizer from memory.
   Returns a non-negative handle on success, or a non-zero error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.

   ``filenames[i]`` is the canonical ``MoonshineTTSOptions::files`` key (e.g.
   ``kokoro/model.onnx``, ``kokoro/config.json``,
   ``kokoro/voices/af_heart.kokorovoice``,
   ``piper/onnx``, ``piper/onnx.json``, ``zipvoice/text_encoder.ort``,
   ``zipvoice/fm_decoder.ort``, ``zipvoice/vocoder.ort``,
   ``zipvoice/tokens.txt``, ``zipvoice/model.json``). For ZipVoice a
   caller-supplied reference clip is passed as key ``zipvoice/clone_audio``
   (raw little-endian float32 mono PCM); set ``zipvoice_clone_sample_rate`` and,
   optionally, ``zipvoice_clone_transcript``. When the transcript is omitted,
   pass ``zipvoice_asr_transcriber_handle=<handle>`` (an existing transcriber
   from ``moonshine_create_transcriber_*``) to auto-transcribe the clip with
   Moonshine ASR. When ``memory[i]`` is non-NULL and
   ``memory_sizes[i]`` > 0, that buffer is used as the asset bytes; the library
   does not copy it—keep the buffers valid until
   ``moonshine_free_tts_synthesizer``. When ``memory[i]`` is NULL or
   ``memory_sizes[i]`` is zero, the key string is also used as a path relative
   to ``g2p_options.g2p_root`` (from ``options``), same as path-only map
   entries.

   Other ``options`` are parsed like
   ``moonshine_create_tts_synthesizer_from_files``.
*/
MOONSHINE_EXPORT int32_t moonshine_create_tts_synthesizer_from_memory(
    const char *language, const char **filenames,
    const uint64_t filenames_count, const uint8_t **memory,
    const uint64_t *memory_sizes, const struct moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version);

/* Releases the resources used by a text to speech synthesizer.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT void moonshine_free_tts_synthesizer(
    int32_t tts_synthesizer_handle);

/* Returns G2P-only canonical asset keys for one or more languages.
   ``languages`` is comma-separated CLI tags (same as ``moonshine_create_*``
   ``language``); an empty string (or NULL) means all known languages (union of
   keys).
   ``options`` / ``options_count``: same ``moonshine_option_t`` entries as
   grapheme phonemizer / G2P
   (``g2p_root``, ``spanish_narrow_obstruents``, ``oov_onnx_override``, …).
   TTS-only keys
   (``voice``, deprecated ``vocoder_engine`` / ``engine``, Piper/Kokoro paths)
   are ignored here. Non-empty values for in-memory override keys add those
   canonical key names to the list. On success, writes a comma-separated list to
   ``*out_dependencies_json`` and returns
   ``MOONSHINE_ERROR_NONE``. The buffer is allocated with ``malloc``; release
   with ``free``. On failure (e.g. unknown language token), logs and returns a
   non-zero error code and sets
   ``*out_dependencies_json`` to NULL.
*/
MOONSHINE_EXPORT int32_t moonshine_get_g2p_dependencies(
    const char *languages, const struct moonshine_option_t *options,
    uint64_t options_count, char **out_dependencies_json);

/* Returns merged G2P + TTS vocoder canonical asset keys as a JSON array of
   strings (flat list).
   ``languages`` is comma-separated; empty or NULL means all known languages.
   ``options`` / ``options_count``: same entries as
   ``moonshine_create_tts_synthesizer_from_files``
   (``voice`` with optional ``kokoro_`` / ``piper_`` prefix, ``g2p_root``,
   ``piper_onnx``,
   ``kokoro_model``, …; ``vocoder_engine`` / ``engine`` are ignored). Vocoder
   keys follow Kokoro vs Piper selection and the requested ``voice`` like
   ``MoonshineTTS``. On success,
   ``*out_dependencies_json`` is a NUL-terminated JSON array; free with
   ``free``.
*/
MOONSHINE_EXPORT int32_t moonshine_get_tts_dependencies(
    const char *languages, const struct moonshine_option_t *options,
    uint64_t options_count, char **out_dependencies_json);

/* Returns known TTS voices for the requested languages with availability state.
   ``languages`` is comma-separated; empty or NULL means all registered catalog
   languages (same tag set as G2P dependencies) that have a resolved TTS layout.
   ``options`` / ``options_count``: same entries as
   ``moonshine_create_tts_synthesizer_from_files``
   (``voice`` prefix selects vocoder for listing; ``vocoder_engine`` /
   ``engine`` are ignored; Piper/Kokoro path overrides). For accurate ``found``
   / ``missing``, set an asset root with
   ``g2p_root`` or the aliases ``path_root``, ``tts_root``, or ``model_root``
   (see
   ``MoonshineTTSOptions::parse_options``). If none are set, the implementation
   uses the process current working directory. Language bindings typically
   default this to their download/cache directory. The ``voice`` option does not
   filter the list.

   On success, ``*out_voices_json`` is a NUL-terminated JSON object mapping each
   language tag to a JSON array of objects ``{"id":"<voice>","state":"found"}``
   or ``{"id":"<voice>","state":"missing"}``. Voice ids are prefixed with
   ``kokoro_`` or ``piper_``. Kokoro uses the upstream Kokoro-82M voice id
   catalog plus any extra ``*.kokorovoice`` in the bundle; Piper lists the
   language default ONNX stem plus any ``*.onnx`` in the resolved voices
   directory. ``found`` means the asset is on disk or supplied via the in-memory
   file map like ``MoonshineTTS``. Free with ``free``.
*/
MOONSHINE_EXPORT int32_t moonshine_get_tts_voices(
    const char *languages, const struct moonshine_option_t *options,
    uint64_t options_count, char **out_voices_json);

/* ------------------------------ MODEL DOWNLOAD MANIFESTS ----------------- */

/* Returns the download manifest for a speech-to-text transcription model as a
   JSON object. This lets language bindings and applications fetch exactly the
   files a model needs from the CDN (https://download.moonshine.ai) without
   hardcoding the file layout, then load the model from the resulting
   directory with moonshine_load_transcriber_from_files.

   ``language`` is a language code (for example ``"en"``) or English name (for
   example ``"English"``); it must not be empty.

   ``options`` / ``options_count`` recognize:
     - ``model_arch``: one of the MOONSHINE_MODEL_ARCH_* constants as a decimal
       string. When omitted, the default (first) model for the language is
       used.
     - ``include_spelling`` (bool): when true and a spelling model is published
       for the language, its files are appended as an extra group. Defaults to
       false.
   Other options are ignored.

   On success, writes a NUL-terminated JSON object to
   ``*out_dependencies_json`` and returns ``MOONSHINE_ERROR_NONE``. The shape
   is:
     ``{"groups":[{"base_url":"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en","files":["encoder_model.ort","decoder_model_merged.ort","tokenizer.bin","decoder_with_attention.ort"]}]}``
   Download each file from ``base_url + "/" + file``. A model is a single
   group, plus an optional second group for the spelling model (which uses a
   different ``base_url``). The buffer is allocated with ``malloc``; release it
   with ``free``. On failure (empty/unknown language, or an unknown
   language+arch pair) returns a non-zero error code and sets
   ``*out_dependencies_json`` to NULL. */
MOONSHINE_EXPORT int32_t moonshine_get_stt_dependencies(
    const char *language, const struct moonshine_option_t *options,
    uint64_t options_count, char **out_dependencies_json);

/* Returns the download manifest for an intent-recognition embedding model as a
   JSON object with the same shape as moonshine_get_stt_dependencies. Load the
   downloaded directory with moonshine_create_intent_recognizer.

   ``model_name`` is an embedding model id (for example
   ``"embeddinggemma-300m"``); pass NULL or an empty string to use the default
   model.

   ``options`` / ``options_count`` recognize ``variant`` (aliases:
   ``model_variant``): one of ``"q4"``, ``"q8"``, ``"fp16"``, ``"fp32"``, or
   ``"q4f16"``. When omitted, the model's default variant is used. Other
   options are ignored. The manifest includes the model's external-data sidecar
   (``model*.onnx_data``) alongside the ``.onnx`` file and ``tokenizer.bin``.

   On success, writes a NUL-terminated JSON object to
   ``*out_dependencies_json`` (single group) and returns
   ``MOONSHINE_ERROR_NONE``; free with ``free``. On failure (unknown model or
   variant) returns a non-zero error code and sets ``*out_dependencies_json``
   to NULL. */
MOONSHINE_EXPORT int32_t moonshine_get_intent_dependencies(
    const char *model_name, const struct moonshine_option_t *options,
    uint64_t options_count, char **out_dependencies_json);

/* Synthesizes text to speech.
   ``options`` / ``options_count``: optional per-call overrides using the same
   ``name`` / ``value`` convention as the synthesizer constructor. Currently
   only
   ``speed`` is honored for the duration of this call (Kokoro ONNX input and
   Piper length scale); other entries are ignored. Pass NULL / 0 to use the
   synthesizer default speed from construction.

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_text_to_speech(
    int32_t tts_synthesizer_handle, const char *text,
    const struct moonshine_option_t *options, uint64_t options_count,
    float **out_audio_data, uint64_t *out_audio_data_size,
    int32_t *out_sample_rate);

/* Synthesizes speech directly from International Phonetic Alphabet (IPA)
   phonemes, skipping the grapheme-to-phoneme conversion that
   ``moonshine_text_to_speech`` performs internally. ``phonemes`` should be an
   IPA string in the same format produced by ``moonshine_text_to_phonemes`` (a
   grapheme-to-phonemizer created for the matching language). This lets callers
   inspect or edit the phonemes between the text-to-phonemes and
   phonemes-to-speech steps (e.g. to fix pronunciation of a name). The
   phonemes are normalized to the active vocoder's phoneme inventory before
   synthesis, so passing the raw ``moonshine_text_to_phonemes`` output for the
   same language yields audio equivalent to ``moonshine_text_to_speech`` on the
   original text.

   ``options`` / ``options_count`` behave exactly like
   ``moonshine_text_to_speech``: only ``speed`` is honored for the duration of
   the call; pass NULL / 0 to use the synthesizer defaults.

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_phonemes_to_speech(
    int32_t tts_synthesizer_handle, const char *phonemes,
    const struct moonshine_option_t *options, uint64_t options_count,
    float **out_audio_data, uint64_t *out_audio_data_size,
    int32_t *out_sample_rate);

/* Creates a grapheme to phonemizer from files on disk.
   Returns a non-negative handle on success, or a negative error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.

   Lexicons and bundled ONNX assets are resolved under ``g2p_root`` (or the
   process current working directory when ``g2p_root`` / ``model_root`` is
   unset) using the same canonical relative keys as
   ``MoonshineG2POptions::files`` in the C++ API (for example
   ``en_us/dict_filtered_heteronyms.tsv``,
   ``zh_hans/roberta_chinese_base_upos_onnx/meta.json``,
   ``zh_hans/roberta_chinese_base_upos_onnx/model.onnx``,
   ``en_us/g2p-config.json``, ``en_us/oov/model.onnx``,
   ``en_us/oov/onnx-config.json``). Japanese and Arabic tok-POS / diacritizer
   bundles use the same pattern under ``ja/...`` and
   ``ar_msa/...``. Korean rule G2P uses ``ko/dict.tsv`` only. If an ONNX
   model uses external data files (e.g. ``model.onnx.data``), those must sit
   beside the ``.onnx`` on disk so the runtime can open them.
*/
MOONSHINE_EXPORT int32_t moonshine_create_grapheme_to_phonemizer_from_files(
    const char *language, const char **filenames, uint64_t filenames_count,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version);

/* Creates a grapheme to phonemizer from memory.
   Returns a non-negative handle on success, or a negative error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.

   ``filenames[i]`` is the canonical ``MoonshineG2POptions::files`` key.
   When ``memory[i]`` is non-NULL and ``memory_sizes[i]`` > 0, that buffer is
   used as the asset bytes (not copied—keep valid until the phonemizer is
   freed). When ``memory[i]`` is NULL or size zero, the key is also used as a
   path relative to ``g2p_root``, like path-only map entries.

   Register every file the engine needs: language lexicon ``dict.tsv`` paths,
   English ``g2p-config.json`` and OOV ONNX keys under ``en_us/oov/``, and
   for ONNX bundles the ``meta.json``, ``vocab.txt``, ``tokenizer_config.json``,
   and ``model.onnx`` keys under the bundle directory key. English OOV overrides
   use ``oov_onnx_override`` for the ``.onnx`` bytes and ``oov_onnx_config`` for
   the merged JSON config UTF-8 text. Models split across ``model.onnx`` plus
   external weight files must be supplied as a single self-contained ``.onnx``
   buffer (or remain on disk via the path fallback) so the runtime does not
   need a sidecar ``.data`` file.
*/
MOONSHINE_EXPORT int32_t moonshine_create_grapheme_to_phonemizer_from_memory(
    const char *language, const char **filenames,
    const uint64_t filenames_count, const uint8_t **memory,
    const uint64_t *memory_sizes, const struct moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version);

/* Releases the resources used by a grapheme to phonemizer.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT void moonshine_free_grapheme_to_phonemizer(
    int32_t grapheme_to_phonemizer_handle);

/* Converts a text into the equivalent International Phonetic Alphabet (IPA)
   phonemes. Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_text_to_phonemes(
    int32_t grapheme_to_phonemizer_handle, const char *text,
    const struct moonshine_option_t *options, uint64_t options_count,
    const char **out_phonemes, uint64_t *out_phonemes_count);

#ifdef __cplusplus
}
#endif

#endif
