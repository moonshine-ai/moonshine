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

/* --------------------------- DATA STRUCTURES ----------------------------- */

/* Values passed to moonshine_load_transcriber at creation time that control
   the behavior of the transcriber. A typical use case would be to specify
   model configuration options like layer names that vary by language. The
   value is a string. You don't normally need to care about these, this is just
   for advanced customizations.                                              */
struct transcriber_option_t {
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

 - The speaker ID is another 64-bit randomly-generated number, used to identify
   the calculated speaker of the line, for diarization purposes. This is not
   available until the line has accumulated enough audio data to be confident
   in the speaker identification, or if the line is complete.

See the stream transcription examples below for more details on what this
means in practice.
*/

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
  /* Whether a speaker ID has been calculated for the line. */
  int8_t has_speaker_id;
  /* The speaker ID for the line. */
  uint64_t speaker_id;
  /* What order the speaker appeared in the current transcript. */
  uint32_t speaker_index;
  /* Streaming-only: The latency of the last transcription in milliseconds. */
  uint32_t last_transcription_latency_ms;
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
   transcriber.

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
    const struct transcriber_option_t *options, uint64_t options_count,
    int32_t moonshine_version);

/* Loads models from memory. The `encoder_model_data`, `decoder_model_data` and
   `tokenizer_data` parameters are the data arrays for the models in binary
   format, and are expected to be in the same format as the files disk.
   All of the other parameters are the same as for
   moonshine_load_transcriber_from_files.                                    */
MOONSHINE_EXPORT int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    uint32_t model_arch, const struct transcriber_option_t *options,
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
   `flags` should be a bitwise OR of any of the following flags:
   - MOONSHINE_FLAG_DISABLE_VAD: Disable voice activity detection, so all audio
     is processed as if it were speech.

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
   the list of lines, and once the `is_complete` flag is set to 1 for a line, it
   will never be updated again.
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

/* Callback function type for intent handlers. The callback receives:
   - user_data: The user data pointer passed to moonshine_register_intent.
   - trigger_phrase: The trigger phrase that matched.
   - utterance: The utterance that was recognized.
   - similarity: The similarity score between 0 and 1.
*/
typedef void (*moonshine_intent_callback)(void *user_data,
                                          const char *trigger_phrase,
                                          const char *utterance,
                                          float similarity);

/* Creates an intent recognizer from files on disk.

   `model_path` should be the path to the directory containing the embedding
   model files (ONNX model and tokenizer.bin).

   `model_arch` should be one of the MOONSHINE_EMBEDDING_MODEL_ARCH_* constants.
   Currently only MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M is supported.

   `model_variant` specifies which model variant to load: "fp32", "fp16", "q8",
   "q4", or "q4f16". Pass NULL to use the default "q4" variant.

   `threshold` is the minimum similarity score required to trigger an intent
   (default 0.7, range 0.0-1.0).

   Returns a non-negative handle on success, or a negative error code on
   failure. The error code can be converted to a human-readable string using
   moonshine_error_to_string.
*/
MOONSHINE_EXPORT int32_t
moonshine_create_intent_recognizer(const char *model_path, uint32_t model_arch,
                                   const char *model_variant, float threshold);

/* Frees an intent recognizer and all its resources. */
MOONSHINE_EXPORT void moonshine_free_intent_recognizer(
    int32_t intent_recognizer_handle);

/* Registers an intent with a trigger phrase and callback. When an utterance
   is processed that is similar enough to the trigger phrase (above the
   threshold), the callback will be invoked.

   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_register_intent(
    int32_t intent_recognizer_handle, const char *trigger_phrase,
    moonshine_intent_callback callback, void *user_data);

/* Unregisters an intent by its trigger phrase.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_unregister_intent(
    int32_t intent_recognizer_handle, const char *trigger_phrase);

/* Processes an utterance and invokes the callback of the most similar intent
   if the similarity exceeds the threshold.
   Returns 1 if an intent was recognized, 0 if not, or a negative error code.
*/
MOONSHINE_EXPORT int32_t moonshine_process_utterance(
    int32_t intent_recognizer_handle, const char *utterance);

/* Sets the similarity threshold for the intent recognizer.
   Returns zero on success, or a non-zero error code on failure.
*/
MOONSHINE_EXPORT int32_t moonshine_set_intent_threshold(
    int32_t intent_recognizer_handle, float threshold);

/* Gets the current similarity threshold for the intent recognizer.
   Returns the threshold on success (>= 0), or a negative error code on failure.
*/
MOONSHINE_EXPORT float moonshine_get_intent_threshold(
    int32_t intent_recognizer_handle);

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

#ifdef __cplusplus
}
#endif

#endif
