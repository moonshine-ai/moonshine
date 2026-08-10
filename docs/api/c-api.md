# C API

The C API in [`core/moonshine-c-api.h`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-c-api.h) is the portable ABI that every Moonshine language binding is built on. Python, Swift, Java/Android, JavaScript/WASM, and the C++ wrapper all call into this layer so they share one implementation of speech to text, text to speech, embeddings, and related helpers. Prefer those higher-level bindings for application code; use the C API directly when you are writing a new binding, embedding Moonshine in a C or C++ project, or need the lowest-level control over handles, buffers, and model loading.

All API calls are thread-safe. Work on a single transcriber is serialized, so concurrent calls that share one handle will queue behind each other. Handles are opaque `int32_t` values: non-negative on success, negative error codes on failure (convert with `moonshine_error_to_string()`). Pass `MOONSHINE_HEADER_VERSION` into loaders so newer shared libraries can emulate the header you compiled against. Buffers documented as allocated with `malloc` must be released with `moonshine_free_buffer()`, not the process `free`, so Windows hosts linked against a different CRT stay safe.

- [Constants](#constants)
    - [Version](#version)
    - [Model architectures](#model-architectures)
    - [Error codes](#error-codes)
    - [Flags](#flags)
- [Data Structures](#data-structures)
    - [`moonshine_option_t`](#moonshine_option_t)
    - [`transcript_word_t`](#transcript_word_t)
    - [`speaker_span_t`](#speaker_span_t)
    - [`transcript_line_t`](#transcript_line_t)
    - [`transcript_t`](#transcript_t)
    - [`moonshine_speech_clip_t`](#moonshine_speech_clip_t)
    - [TTS voice catalog entries](#tts-voice-catalog-entries)
- [Utilities](#utilities)
    - [`moonshine_get_version()`](#moonshine_get_version)
    - [`moonshine_error_to_string()`](#moonshine_error_to_string)
    - [`moonshine_free_buffer()`](#moonshine_free_buffer)
    - [`moonshine_transcript_to_string()`](#moonshine_transcript_to_string)
- [Speech to Text](#speech-to-text)
    - [`moonshine_load_transcriber_from_files()`](#moonshine_load_transcriber_from_files)
    - [`moonshine_load_transcriber_from_memory_files()`](#moonshine_load_transcriber_from_memory_files)
    - [`moonshine_load_transcriber_from_memory()`](#moonshine_load_transcriber_from_memory)
    - [`moonshine_free_transcriber()`](#moonshine_free_transcriber)
    - [`moonshine_transcriber_set_keyterms()`](#moonshine_transcriber_set_keyterms)
    - [`moonshine_transcribe_without_streaming()`](#moonshine_transcribe_without_streaming)
- [Streaming Speech to Text](#streaming-speech-to-text)
    - [`moonshine_create_stream()`](#moonshine_create_stream)
    - [`moonshine_free_stream()`](#moonshine_free_stream)
    - [`moonshine_start_stream()`](#moonshine_start_stream)
    - [`moonshine_stop_stream()`](#moonshine_stop_stream)
    - [`moonshine_transcribe_add_audio_to_stream()`](#moonshine_transcribe_add_audio_to_stream)
    - [`moonshine_transcribe_stream()`](#moonshine_transcribe_stream)
- [Embeddings](#embeddings)
    - [`moonshine_create_embedding_model()`](#moonshine_create_embedding_model)
    - [`moonshine_create_embedding_model_from_memory()`](#moonshine_create_embedding_model_from_memory)
    - [`moonshine_free_embedding_model()`](#moonshine_free_embedding_model)
    - [`moonshine_calculate_embedding()`](#moonshine_calculate_embedding)
    - [`moonshine_free_embedding()`](#moonshine_free_embedding)
    - [`moonshine_calculate_embedding_distance()`](#moonshine_calculate_embedding_distance)
- [Speech Clips](#speech-clips)
    - [`moonshine_extract_speech_clip()`](#moonshine_extract_speech_clip)
- [Text to Speech](#text-to-speech)
    - [`moonshine_create_tts_synthesizer_from_files()`](#moonshine_create_tts_synthesizer_from_files)
    - [`moonshine_create_tts_synthesizer_from_memory()`](#moonshine_create_tts_synthesizer_from_memory)
    - [`moonshine_free_tts_synthesizer()`](#moonshine_free_tts_synthesizer)
    - [`moonshine_text_to_speech()`](#moonshine_text_to_speech)
    - [`moonshine_phonemes_to_speech()`](#moonshine_phonemes_to_speech)
    - [`moonshine_get_tts_dependencies()`](#moonshine_get_tts_dependencies)
    - [`moonshine_get_tts_voices()`](#moonshine_get_tts_voices)
- [Grapheme to Phonemes](#grapheme-to-phonemes)
    - [`moonshine_create_grapheme_to_phonemizer_from_files()`](#moonshine_create_grapheme_to_phonemizer_from_files)
    - [`moonshine_create_grapheme_to_phonemizer_from_memory()`](#moonshine_create_grapheme_to_phonemizer_from_memory)
    - [`moonshine_free_grapheme_to_phonemizer()`](#moonshine_free_grapheme_to_phonemizer)
    - [`moonshine_text_to_phonemes()`](#moonshine_text_to_phonemes)
    - [`moonshine_get_g2p_dependencies()`](#moonshine_get_g2p_dependencies)
- [Model Download Manifests](#model-download-manifests)
    - [`moonshine_get_stt_dependencies()`](#moonshine_get_stt_dependencies)
    - [`moonshine_get_embedding_dependencies()`](#moonshine_get_embedding_dependencies)
    - [`moonshine_get_diarization_dependencies()`](#moonshine_get_diarization_dependencies)
    - [`moonshine_get_stt_catalog()`](#moonshine_get_stt_catalog)
    - [`moonshine_get_embedding_catalog()`](#moonshine_get_embedding_catalog)

## Constants

### Version

| Constant | Value | Meaning |
| --- | --- | --- |
| `MOONSHINE_HEADER_VERSION` | `30000` (3.0.0) | Pass this to loader functions so a newer library can emulate this header. Format is `MAJOR * 10000 + MINOR * 100 + PATCH`. |
| `MOONSHINE_FROM_MEMORY_REMOVED_VERSION` | `30000` | Callers passing this version or newer are refused by the deprecated `moonshine_load_transcriber_from_memory()`. Use `moonshine_load_transcriber_from_memory_files()` instead. |

### Model architectures

| Constant | Value |
| --- | --- |
| `MOONSHINE_MODEL_ARCH_TINY` | `0` |
| `MOONSHINE_MODEL_ARCH_BASE` | `1` |
| `MOONSHINE_MODEL_ARCH_TINY_STREAMING` | `2` |
| `MOONSHINE_MODEL_ARCH_BASE_STREAMING` | `3` |
| `MOONSHINE_MODEL_ARCH_SMALL_STREAMING` | `4` |
| `MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING` | `5` |
| `MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M` | `0` |

### Error codes

| Constant | Value |
| --- | --- |
| `MOONSHINE_ERROR_NONE` | `0` |
| `MOONSHINE_ERROR_UNKNOWN` | `-1` |
| `MOONSHINE_ERROR_INVALID_HANDLE` | `-2` |
| `MOONSHINE_ERROR_INVALID_ARGUMENT` | `-3` |

Loader functions that return a handle also use negative values as errors; convert any non-success code with `moonshine_error_to_string()`.

### Flags

| Constant | Meaning |
| --- | --- |
| `MOONSHINE_FLAG_FORCE_UPDATE` | Ignore the ~200ms caching window in `moonshine_transcribe_stream()` and force a full analysis. |
| `MOONSHINE_FLAG_SPELLING_MODE` | Apply alphanumeric-spelling fusion to completed lines. Requires a spelling model at load time; otherwise ignored. When fusion fires, the line `text` is replaced with the resolved character (for example `"a"` or `"$"`). |

## Data Structures

### `moonshine_option_t`

Name/value string pairs passed to loaders and some synthesis calls. See [Options](options.md) for the full catalog of keys.

| Member | Type | Meaning |
| --- | --- | --- |
| `name` | `const char *` | Option key. |
| `value` | `const char *` | Option value (always a string, even for numbers and booleans). |

### `transcript_word_t`

A single word with timing information. Empty / unused unless the `word_timestamps` option is enabled (`identify_speakers` turns that on for you). Each entry has the word itself, its start and end times in seconds, and a confidence value.

| Member | Type | Meaning |
| --- | --- | --- |
| `text` | `const char *` | UTF-8-encoded word text. |
| `start` | `float` | Start time in seconds from the start of the audio/stream. |
| `end` | `float` | End time in seconds. |
| `confidence` | `float` | Model confidence score, 0.0 to 1.0. |

### `speaker_span_t`

One contiguous span of speech within a line attributed to a single speaker. Only populated when the opt-in `identify_speakers` option is enabled (which also turns on word timestamps automatically, since they are needed to map spans onto the line text). Spans are ordered by start time and clipped to the line's time range.

Be aware that speaker spans are *mutable*: streaming diarization re-clusters a sliding window of recent speech (`diarization_cluster_window_sec`, 120 seconds by default) as more audio arrives, so spans within that window can move, merge, split, or change speaker on any transcription call — even on lines that are already complete. Assignments for audio older than the window are frozen. Watch `have_speakers_changed` on [`transcript_line_t`](#transcript_line_t) (or the `LineSpeakersChanged` event in higher-level bindings) to catch revisions. Character ranges are UTF-8 byte offsets into the line text; both are zero when the span could not be aligned to words yet.

| Member | Type | Meaning |
| --- | --- | --- |
| `start_time` | `float` | Time offset from the start of the array or stream, in seconds. |
| `duration` | `float` | Length of the span in seconds. |
| `speaker_id` | `uint64_t` | Unique-ish identifier that is stable for a given speaker within a stream, designed for storage or keeping track of speakers over time. |
| `speaker_index` | `uint32_t` | Order the speaker first appeared in the transcript, starting at 0 — handy for default names like "Speaker 1". |
| `start_char` | `uint64_t` | UTF-8 byte offset into the line text where this span begins (inclusive); both char fields zero when unknown. |
| `end_char` | `uint64_t` | UTF-8 byte offset into the line text where this span ends (exclusive). Slice with `text[start_char:end_char]` in bindings that expose strings that way. |

### `transcript_line_t`

Represents a single "line" or speech segment in a transcript: timing, speaker, text, and state such as whether the speech is ongoing or done. If you're building an application that involves transcription, this structure has all of the information available about each line of speech. Be aware that each line can be updated multiple times with new text and other information as the user keeps speaking.

Memory referenced by a line is owned by the transcriber and remains valid until the next call on that transcriber, or until the transcriber is freed. Audio data is 16 kHz float PCM in [-1.0, 1.0] — useful for further processing (for example to drive a visual indicator or to feed into a specialized model after the line is complete).

Higher-level bindings map changes on these lines into events (`LineStarted`, `LineUpdated`, `LineTextChanged`, `LineSpeakersChanged`, `LineCompleted`); see [Transcription Event Flow](../using/transcription.md#transcription-event-flow). Prefer those callbacks over polling the dirty flags below, because the library updates the transcript internally as you add audio chunks.

| Member | Type | Meaning |
| --- | --- | --- |
| `text` | `const char *` | UTF-8-encoded text extracted from this segment's audio. |
| `audio_data` | `const float *` | Raw audio for the current phrase, as 16 kHz mono PCM in [-1.0, 1.0]. |
| `audio_data_count` | `size_t` | Number of samples in `audio_data`. |
| `start_time` | `float` | Time offset from the start of the audio array or stream, in seconds — when the utterance was first detected. |
| `duration` | `float` | How long the segment currently is, in seconds. |
| `id` | `uint64_t` | Stable, collision-resistant line identifier (a randomly generated 64-bit number) for storage and for tracking a line as it changes over time. It stays the same from the line's first appearance onwards. Ids happen to ascend within a transcript today, but that is not guaranteed and should not be relied on. |
| `is_complete` | `int8_t` | Streaming: false until the segment has been completed, then true for the remainder of the line's lifetime. |
| `is_updated` | `int8_t` | Streaming: true if any information about the line changed since the previous `moonshine_transcribe_stream()` call. Do not poll this as your primary change signal — use events/listeners in bindings. |
| `is_new` | `int8_t` | Streaming: true if the line was added by the last update call. |
| `has_text_changed` | `int8_t` | Streaming: true if the line's text was modified by the last update. If set, `is_updated` is always set too; if only duration or audio changed, `is_updated` can be true while this is false. |
| `have_speakers_changed` | `int8_t` | True if the line's speaker spans were revised by the last update. Unlike the other change flags, this can be set for lines that are already complete, since diarization keeps refining assignments for recent audio. Only relevant when `identify_speakers` is enabled. |
| `speaker_spans` | `const struct speaker_span_t *` | Speaker spans covering this line, ordered by start time and clipped to the line's time range. NULL unless `identify_speakers` is enabled and speech has been attributed to a speaker. See [`speaker_span_t`](#speaker_span_t). |
| `speaker_span_count` | `uint64_t` | Number of speaker spans. |
| `last_transcription_latency_ms` | `uint32_t` | Streaming: milliseconds between the library deciding speech had ended and the final transcript for that line being ready. Useful for measuring end-of-phrase responsiveness; see [Benchmarks](../using/benchmarks.md). |
| `words` | `const struct transcript_word_t *` | Per-word timings, or NULL if the `word_timestamps` option is not enabled. See [`transcript_word_t`](#transcript_word_t). |
| `word_count` | `uint64_t` | Number of entries in `words`; zero when word timestamps are not enabled. |

Streaming guarantees: lines are never removed, only added; only the last line may be incomplete; empty text `""` means speech was detected but no transcription was produced; line indexes are stable across streaming calls; once `is_complete` is set, text and timing do not change again (speaker spans for recent audio are the exception when diarization is on — assignments older than `diarization_cluster_window_sec` are frozen).

### `transcript_t`

An entire transcription of an audio buffer or stream: a list of [`transcript_line_t`](#transcript_line_t) values. The transcript is reset whenever a stream is started fresh (or a new one-shot transcription runs), so if you need to retain information from it, make explicit copies. Most applications work through streaming updates and binding-level event callbacks rather than retaining this structure long-term.

| Member | Type | Meaning |
| --- | --- | --- |
| `lines` | `struct transcript_line_t *` | All lines of the transcript. |
| `line_count` | `uint64_t` | Number of lines. |

### `moonshine_speech_clip_t`

A short window of mostly-speech audio returned by `moonshine_extract_speech_clip()`, typically used as a ZipVoice cloning reference.

| Member | Type | Meaning |
| --- | --- | --- |
| `audio_data` | `float *` | 16 kHz mono PCM; NULL unless `is_complete` is non-zero. Allocated with malloc; release with `moonshine_free_buffer()`. |
| `audio_length` | `uint64_t` | Sample count. |
| `start_time` | `float` | Where the window starts in the input recording, in seconds. |
| `speech_duration` | `float` | How much of the window is speech, in seconds. |
| `is_complete` | `int32_t` | Non-zero once a window with enough speech was found. |
| `transcript` | `char *` | UTF-8 transcript when clone ASR refined a complete clip; otherwise NULL. Release with `moonshine_free_buffer()`. |

### TTS voice catalog entries

`moonshine_get_tts_voices()` returns JSON rather than a C struct. Each voice row has the shape used by higher-level helpers such as `get_tts_voice_catalog()` / `list_tts_voices()`:

**Voice entry** (`{"id":"...","state":"..."}`):

- `id`: The voice identifier string (often with a `kokoro_` or `piper_` prefix to pin the vocoder).
- `state`: Either `"found"` (assets present under the resolved asset root) or `"missing"` (listed in the catalog but not on disk yet).

**By availability** (the dictionary shape bindings expose from `list_tts_voices()`):

- `present`: Sorted list of voice ids that are already available under the asset root used for the query.
- `downloadable`: Sorted list of catalog voice ids that are not on disk yet but can be fetched (for example when a high-level `TextToSpeech().load()` downloads assets).

## Utilities

Versioning, error strings, transcript debugging, and buffer ownership helpers used across the rest of the API.

### `moonshine_get_version()`

Returns the loaded moonshine library version. This may be different from the header version if a newer shared library is loaded.

```c
int32_t moonshine_get_version(void);
```

**Returns:** The loaded library version as an integer (`MAJOR * 10000 + MINOR * 100 + PATCH`). May differ from `MOONSHINE_HEADER_VERSION` if a newer shared library is loaded.

### `moonshine_error_to_string()`

Converts an error code number returned from an API call into a human-readable string.

```c
const char *moonshine_error_to_string(
    int32_t error
);
```

| Argument | Description |
| --- | --- |
| `error` | Error code returned from an API call (for example a negative handle or `MOONSHINE_ERROR_*` value). |

**Returns:** A human-readable string for the error code.

### `moonshine_free_buffer()`

Frees a buffer that a `moonshine_*` function returned to the caller as a heap allocation. This covers, for example, `out_audio_data` from `moonshine_text_to_speech()` / `moonshine_phonemes_to_speech()`, the JSON / comma-separated strings from `moonshine_get_tts_dependencies()` / `moonshine_get_g2p_dependencies()` / `moonshine_get_tts_voices()`, and `out_phonemes` from `moonshine_text_to_phonemes()`.

Always use this instead of the C runtime `free` directly. On Windows the library and its host (e.g. a Python binding) can be linked against different C runtimes with independent heaps, so freeing a library-allocated pointer with the host's `free` corrupts the heap. Routing the free back through the library guarantees the allocation and deallocation happen in the same runtime. Safe to call on NULL.

```c
void moonshine_free_buffer(
    void *ptr
);
```

| Argument | Description |
| --- | --- |
| `ptr` | Pointer previously returned by a Moonshine function as a malloc-allocated buffer. Safe to pass `NULL`. |

**Returns:** Nothing.

### `moonshine_transcript_to_string()`

Converts a transcript_t struct into a human-readable string for debugging purposes. The string is owned by the library, and is valid until the next call to `moonshine_transcript_to_string()`.

```c
const char *moonshine_transcript_to_string(
    const struct transcript_t *transcript
);
```

| Argument | Description |
| --- | --- |
| `transcript` | Transcript to format. |

**Returns:** A human-readable string describing the transcript. The string is owned by the library and stays valid until the next call to `moonshine_transcript_to_string()`.

## Speech to Text

Load a transcriber from files or memory, run one-shot transcription, set key terms, and release resources.

### `moonshine_load_transcriber_from_files()`

Loads models from the file system, using `path` as the root directory. A non-streaming model directory is expected to contain:

- `encoder_model.ort`
- `decoder_model_merged.ort`
- `tokenizer.bin`

The `.ort` files are quantized-activation ONNX models converted to ORT format with the onnxruntime tools. The simplest way to obtain them is to run `python scripts/download-moonshine-model.py --model-type base --model-language en`. The source weights are on the [Hugging Face Model Hub](https://huggingface.co/moonshine-ai/), and the download and conversion script is `scripts/convert-moonshine-model.sh` in this repository. `tokenizer.bin` holds the token-to-character mapping in a compact binary format; `scripts/json-to-bin-vocab.py` converts a common `tokenizer.json` into one.

The `options` parameter is a list of [`moonshine_option_t`](#moonshine_option_t) entries. See [Options → Speech to Text](options.md#speech-to-text) (and [shared options](options.md#shared-options)) for every recognized key.

```c
int32_t moonshine_load_transcriber_from_files(
    const char *path,
    uint32_t model_arch,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `path` | Root directory holding the model files listed above. |
| `model_arch` | Model architecture to load, for example `MOONSHINE_MODEL_ARCH_BASE` or `MOONSHINE_MODEL_ARCH_TINY_STREAMING`. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Speech to Text options](options.md#speech-to-text). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative handle identifying the transcriber in subsequent calls, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_load_transcriber_from_memory_files()`

Loads a transcriber from a set of in-memory model assets keyed by their canonical filename. This is the in-memory counterpart that reaches full parity with `moonshine_load_transcriber_from_files()`: unlike `moonshine_load_transcriber_from_memory()` (which only accepts a fixed encoder/decoder/tokenizer/spelling set and rejects streaming models), this entry point accepts whatever files the chosen architecture needs, resolved by name.

`filenames[i]` is the canonical filename as it would appear on disk under a model directory. Recognized keys depend on `model_arch`:

- Non-streaming (TINY, BASE): `encoder_model.ort`, `decoder_model_merged.ort`, and `tokenizer.bin` are all required, plus the optional word-timestamp decoder `decoder_with_attention.ort` (or the two-pass `alignment_model.ort`) when the `word_timestamps` option is set.
- Streaming (`*_STREAMING`): `frontend.ort`, `encoder.ort`, `adapter.ort`, `cross_kv.ort`, `decoder_kv.ort`, `streaming_config.json`, and `tokenizer.bin` are all required, plus the optional `decoder_kv_with_attention.ort` when `word_timestamps` is set.
- Either kind also accepts `spelling_cnn.ort`, and the two diarization models `segmentation.ort` and `embedding.ort`, which are required when the `identify_speakers` option is set. Fetch those two with `moonshine_get_diarization_dependencies()`.

Unrecognized keys are rejected with `MOONSHINE_ERROR_INVALID_ARGUMENT`, and missing required keys cause the load to fail. The recognized set is the union of the names above across every architecture, plus `spelling_cnn_meta.json`, so passing an asset this architecture or option set has no use for is fine — handing over a whole downloaded model directory works. A misspelled name is reported against the key you passed rather than surfacing later as a missing-asset failure.

When `memory[i]` is non-NULL and `memory_sizes[i]` > 0, that buffer is used as the asset bytes. The library does not copy the model buffers (the ONNX Runtime sessions read them directly), so the buffers must outlive the transcriber, exactly as for `moonshine_load_transcriber_from_memory()`. When `memory[i]` is NULL or `memory_sizes[i]` is zero, `filenames[i]` is also used as a filesystem path (relative to the current working directory unless absolute), so callers can mix in-memory and on-disk assets.

All other parameters behave as in the other transcriber loaders.

```c
int32_t moonshine_load_transcriber_from_memory_files(
    const char **filenames,
    const uint8_t **memory,
    const uint64_t *memory_sizes,
    uint64_t file_count,
    uint32_t model_arch,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `filenames` | Canonical asset filenames, one per entry, as they would appear on disk under a model directory. A name outside the recognized set fails the call. |
| `memory` | Asset bytes for each filename, or `NULL` for that entry to load it from disk instead. Buffers are not copied and must outlive the transcriber. |
| `memory_sizes` | Byte count for each entry in `memory`; zero also means "load this one from disk". |
| `file_count` | Number of entries in `filenames`, `memory`, and `memory_sizes`. |
| `model_arch` | Model architecture to load, which determines the set of required filenames. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Speech to Text options](options.md#speech-to-text). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative transcriber handle on success, or a negative error code on failure — `MOONSHINE_ERROR_INVALID_ARGUMENT` when a filename key is not recognized.

### `moonshine_load_transcriber_from_memory()`

**DEPRECATED** Use `moonshine_load_transcriber_from_memory_files()` instead. This function is deprecated and will be removed in a future version.

Callers that pass a `moonshine_version` of MOONSHINE_FROM_MEMORY_REMOVED_VERSION or newer are refused: the call logs an explanation and returns MOONSHINE_ERROR_INVALID_ARGUMENT without loading anything. Only clients built against an earlier header, which pass that earlier version here, can still use it.

Loads models from memory. The `encoder_model_data`, `decoder_model_data` and `tokenizer_data` parameters are the data arrays for the models in binary format, and are expected to be in the same format as the files disk.

`spelling_model_data` and `spelling_model_data_size` are an optional in-memory `.ort` payload for the alphanumeric spelling-CNN. Pass `NULL` and `0` if you don't want spelling fusion. When provided, the buffer must outlive the transcriber (it is *not* copied) and the transcriber will run spelling fusion whenever `MOONSHINE_FLAG_SPELLING_MODE` is passed to `moonshine_transcribe_stream()` or `moonshine_transcribe_without_streaming()`.

All of the other parameters are the same as for `moonshine_load_transcriber_from_files()`.

```c
int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data,
    size_t encoder_model_data_size,
    const uint8_t *decoder_model_data,
    size_t decoder_model_data_size,
    const uint8_t *tokenizer_data,
    size_t tokenizer_data_size,
    const uint8_t *spelling_model_data,
    size_t spelling_model_data_size,
    uint32_t model_arch,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `encoder_model_data` | Encoder model bytes, in the same format as the file on disk. |
| `encoder_model_data_size` | Byte count of `encoder_model_data`. |
| `decoder_model_data` | Decoder model bytes, in the same format as the file on disk. |
| `decoder_model_data_size` | Byte count of `decoder_model_data`. |
| `tokenizer_data` | Tokenizer bytes, in the same format as `tokenizer.bin` on disk. |
| `tokenizer_data_size` | Byte count of `tokenizer_data`. |
| `spelling_model_data` | Optional in-memory `.ort` payload for the alphanumeric spelling-CNN; pass `NULL` to go without spelling fusion. Not copied, so it must outlive the transcriber. |
| `spelling_model_data_size` | Byte count of `spelling_model_data`; pass `0` when there is none. |
| `model_arch` | Model architecture to load. Streaming architectures are rejected by this entry point. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Speech to Text options](options.md#speech-to-text). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Must be older than `MOONSHINE_FROM_MEMORY_REMOVED_VERSION`; this call refuses anything newer. |

**Returns:** A non-negative transcriber handle on success, or a negative error code on failure — including `MOONSHINE_ERROR_INVALID_ARGUMENT` when `moonshine_version` is `MOONSHINE_FROM_MEMORY_REMOVED_VERSION` or newer.

### `moonshine_free_transcriber()`

Releases all resources used by the transcriber. Subsequent transcriber creation calls may reuse this transcriber's ID, so ensure you remove all references to it in your client code after freeing it.

```c
void moonshine_free_transcriber(
    int32_t transcriber_handle
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. |

**Returns:** Nothing.

### `moonshine_transcriber_set_keyterms()`

Replaces the contextual-biasing key terms on an existing transcriber, so a caller can follow whatever context the user is in - the contact list on screen, the vocabulary of the document being dictated into - without reloading the model. `keyterms` is a comma-separated list using the same syntax as the `keyterms` load option; pass NULL or an empty string to turn biasing off.

Safe to call between transcribe calls on a live stream. Takes effect on the next transcribe call: it does not retroactively change text already emitted.

```c
int32_t moonshine_transcriber_set_keyterms(
    int32_t transcriber_handle,
    const char *keyterms
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. |
| `keyterms` | Comma-separated list using the same syntax as the `keyterms` load option. Pass `NULL` or an empty string to turn biasing off. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code if the handle is invalid or the loaded model is not a streaming architecture (only those decode through a path that can apply the bias).

### `moonshine_transcribe_without_streaming()`

Given an array of PCM audio data, identifies sections of speech and transcribes them into text. This is the call to use if you're analyzing audio from a file or other static source where you have all the audio data at once. If you are transcribing audio from a live microphone or other real-time source, you should use the streaming API instead, since it offers lower latency for those use cases.

```c
int32_t moonshine_transcribe_without_streaming(
    int32_t transcriber_handle,
    float *audio_data,
    uint64_t audio_length,
    int32_t sample_rate,
    uint32_t flags,
    struct transcript_t **out_transcript
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. |
| `audio_data` | Array of mono PCM audio data between -1.0 and 1.0, at `sample_rate` Hz. The library works at 16,000 Hz internally, so capture at that rate to avoid resampling. |
| `audio_length` | Number of samples in `audio_data`. |
| `sample_rate` | Sample rate of the audio data, in Hz. |
| `flags` | Bitwise OR of flags. The only supported flag is `MOONSHINE_FLAG_SPELLING_MODE`, which applies alphanumeric-spelling fusion to completed lines; it is a no-op unless the transcriber was loaded with a spelling model. Pass zero for the default behavior. |
| `out_transcript` | Receives a pointer to a [`transcript_t`](#transcript_t): a list of lines with text, audio data, and timestamps. The data is owned by the transcriber and stays valid until the next call on that transcriber, or until it is freed. |

**Returns:** Zero on success, or a non-zero error code on failure. Convert the code with `moonshine_error_to_string()`.

## Streaming Speech to Text

Create streams on a transcriber, feed live audio, and pull incremental transcripts with lower latency than one-shot transcription.

Streaming reuses work on earlier audio so results can update as new samples arrive. A single transcriber can own multiple streams. Feed audio with `moonshine_transcribe_add_audio_to_stream()` as often as your capture thread needs; call `moonshine_transcribe_stream()` on your own schedule for UI updates.

```c
int32_t transcriber_handle = moonshine_load_transcriber_from_files(
    "path/to/models", MOONSHINE_MODEL_ARCH_BASE, NULL, 0,
    MOONSHINE_HEADER_VERSION);
int32_t stream_handle = moonshine_create_stream(transcriber_handle, 0);
moonshine_start_stream(transcriber_handle, stream_handle);

float *latest_audio_data;
size_t latest_audio_data_length;
while (get_audio_from_microphone(&latest_audio_data, &latest_audio_data_length)) {
  moonshine_transcribe_add_audio_to_stream(
      transcriber_handle, stream_handle, latest_audio_data,
      latest_audio_data_length, microphone_sample_rate, 0);
  transcript_t *partial_transcript = NULL;
  moonshine_transcribe_stream(
      transcriber_handle, stream_handle, 0, &partial_transcript);
  print_transcript(partial_transcript);
}
moonshine_stop_stream(transcriber_handle, stream_handle);

transcript_t *final_transcript = NULL;
moonshine_transcribe_stream(
    transcriber_handle, stream_handle, 0, &final_transcript);
print_transcript(final_transcript);

moonshine_free_stream(transcriber_handle, stream_handle);
moonshine_free_transcriber(transcriber_handle);
```

Returned transcripts are lists of lines with text, audio, timestamps, and flags such as `is_updated`. Use those flags as dirty markers for minimal UI updates. Updated lines appear at the end of the list; once `is_complete` is set, text and timing do not change again.

The exception is speaker information: when `identify_speakers` is enabled, speaker spans for recent audio can be revised on later `moonshine_transcribe_stream()` calls. Watch `have_speakers_changed` for those revisions.

### `moonshine_create_stream()`

Creates a stream on an existing transcriber. The returned handle identifies the stream in subsequent calls.

```c
int32_t moonshine_create_stream(
    int32_t transcriber_handle,
    uint32_t flags
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. A single transcriber can own multiple streams, each transcribing a separate audio source. |
| `flags` | Bitwise OR of flags. None are currently supported, so pass zero. |

**Returns:** A non-negative stream handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_free_stream()`

Releases the resources used by a stream. Subsequent stream creation calls may reuse this stream's ID, so ensure you remove all references to it in your client code after freeing it.

```c
int32_t moonshine_free_stream(
    int32_t transcriber_handle,
    int32_t stream_handle
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle to the transcriber that owns the stream. |
| `stream_handle` | Handle returned by `moonshine_create_stream()`. |

**Returns:** Zero on success, or a non-zero error code on failure.

### `moonshine_start_stream()`

Starts a stream. Call this before any calls to `moonshine_transcribe_stream()`. Start and stop exist because the audio input can be discontinuous — when the user mutes their microphone, for example — so there needs to be a way to start fresh after a break.

```c
int32_t moonshine_start_stream(
    int32_t transcriber_handle,
    int32_t stream_handle
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle to the transcriber that owns the stream. |
| `stream_handle` | Handle returned by `moonshine_create_stream()`. |

**Returns:** Zero on success, or a non-zero error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_stop_stream()`

Stops a stream.

```c
int32_t moonshine_stop_stream(
    int32_t transcriber_handle,
    int32_t stream_handle
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle to the transcriber that owns the stream. |
| `stream_handle` | Handle returned by `moonshine_create_stream()`. |

**Returns:** Zero on success, or a non-zero error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_transcribe_add_audio_to_stream()`

Call this when new audio data becomes available from your microphone or other audio source. This function will add the audio data to the stream's buffer, but it will not transcribe it or do any other processing, so this should be safe to call frequently even from time-critical threads. The size of the input audio doesn't have any impact on performance, so you should call this with whatever the natural chunk size is for your audio source. It is up to you to call `moonshine_transcribe_stream()` when you want an updated transcript, the frequency of which should be determined by your application's latency and compute budgets.

```c
int32_t moonshine_transcribe_add_audio_to_stream(
    int32_t transcriber_handle,
    int32_t stream_handle,
    const float *new_audio_data,
    uint64_t audio_length,
    int32_t sample_rate,
    uint32_t flags
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. |
| `stream_handle` | Handle returned by `moonshine_create_stream()`. |
| `new_audio_data` | Array of mono PCM audio data between -1.0 and 1.0, at `sample_rate` Hz. |
| `audio_length` | Number of samples in `new_audio_data`. |
| `sample_rate` | Sample rate of the audio data, in Hz. |
| `flags` | Bitwise OR of flags. None are currently supported, so pass zero. |

**Returns:** Zero on success, or a non-zero error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_transcribe_stream()`

Analyzes all the audio data in the stream and returns an updated transcript of all the speech segments found. By default this function will only perform full analysis on the audio data if there has been more than 200ms of new samples since the last complete analysis. This is to ensure that too-frequent calls to this function don't result in poor performance. This can be overridden by setting the MOONSHINE_FLAG_FORCE_UPDATE flag.

```c
int32_t moonshine_transcribe_stream(
    int32_t transcriber_handle,
    int32_t stream_handle,
    uint32_t flags,
    struct transcript_t **out_transcript
);
```

| Argument | Description |
| --- | --- |
| `transcriber_handle` | Handle returned by a `moonshine_load_transcriber_*` function. |
| `stream_handle` | Handle returned by `moonshine_create_stream()`. |
| `flags` | Bitwise OR of flags. The only supported flag is `MOONSHINE_FLAG_FORCE_UPDATE`, which ignores the time-based caching logic so the stream is fully analyzed by the models. |
| `out_transcript` | Receives a pointer to a [`transcript_t`](#transcript_t): a list of lines with text, audio data, and timestamps. The data is owned by the transcriber and stays valid until the next call on that transcriber, or until it is freed. |

**Returns:** Zero on success, or a non-zero error code on failure. Convert the code with `moonshine_error_to_string()`.

## Embeddings

Load a text embedding model, embed sentences, compare vectors, and free results.

### `moonshine_create_embedding_model()`

Creates an embedding model from files on disk.

`model_variant` specifies which model variant to load: "fp32", "fp16", "q8", "q4", or "q4f16". Pass NULL to use the default "q4" variant.

```c
int32_t moonshine_create_embedding_model(
    const char *model_path,
    uint32_t model_arch,
    const char *model_variant
);
```

| Argument | Description |
| --- | --- |
| `model_path` | Path to the directory containing the embedding model files (the `.ort` model and `tokenizer.bin`). |
| `model_arch` | One of the `MOONSHINE_EMBEDDING_MODEL_ARCH_*` constants. Currently only `MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M` is supported. |
| `model_variant` | Which variant to load: `"fp32"`, `"fp16"`, `"q8"`, `"q4"`, or `"q4f16"`. Pass `NULL` for the default `"q4"`. |

**Returns:** A non-negative embedding model handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_create_embedding_model_from_memory()`

Creates an embedding model from in-memory model buffers.

This mirrors `moonshine_load_transcriber_from_memory_files()` and `moonshine_create_tts_synthesizer_from_memory()`: `filenames[i]` is the canonical asset filename (as listed by `moonshine_get_embedding_dependencies()`, e.g. `model_q4.ort` and `tokenizer.bin`) and `memory[i]` / `memory_sizes[i]` are the corresponding bytes. The embedding model must be a single self-contained all-in-one `.ort` file (no external-data sidecar); the tokenizer is `tokenizer.bin`. The library copies the bytes it needs, so the buffers only need to remain valid for the duration of this call.

```c
int32_t moonshine_create_embedding_model_from_memory(
    uint32_t model_arch,
    const char *model_variant,
    const char **filenames,
    uint64_t filenames_count,
    const uint8_t **memory,
    const uint64_t *memory_sizes,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `model_arch` | One of the `MOONSHINE_EMBEDDING_MODEL_ARCH_*` constants. |
| `model_variant` | Which variant to load (`"fp32"`, `"fp16"`, `"q8"`, `"q4"`, `"q4f16"`; `NULL` defaults to `"q4"`). Only used to pick the model file when the filename keys leave it ambiguous. |
| `filenames` | Canonical asset filenames, as listed by `moonshine_get_embedding_dependencies()` (for example `model_q4.ort` and `tokenizer.bin`). |
| `filenames_count` | Number of entries in `filenames`, `memory`, and `memory_sizes`. |
| `memory` | Asset bytes for each filename. Copied by the library, so the buffers only need to stay valid for this call. |
| `memory_sizes` | Byte count for each entry in `memory`. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Embeddings options](options.md#embeddings). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative embedding model handle on success, or a negative error code on failure.

### `moonshine_free_embedding_model()`

Frees an embedding model and all its resources.

```c
void moonshine_free_embedding_model(
    int32_t embedding_model_handle
);
```

| Argument | Description |
| --- | --- |
| `embedding_model_handle` | Handle returned by `moonshine_create_embedding_model()` or `moonshine_create_embedding_model_from_memory()`. |

**Returns:** Nothing.

### `moonshine_calculate_embedding()`

Calculates the embedding for a given sentence.

On success, `*out_embedding` is set to a heap-allocated array of floats and `*out_embedding_size` is set to the number of elements. Release the array with `moonshine_free_embedding()`.

```c
int32_t moonshine_calculate_embedding(
    int32_t embedding_model_handle,
    const char *sentence,
    float **out_embedding,
    uint64_t *out_embedding_size,
    const char *model_name
);
```

| Argument | Description |
| --- | --- |
| `embedding_model_handle` | Handle returned by a `moonshine_create_embedding_model*` function. |
| `sentence` | UTF-8 text to embed. |
| `out_embedding` | Receives a heap-allocated array of floats. Release it with `moonshine_free_embedding()`. |
| `out_embedding_size` | Receives the number of elements in `out_embedding`. |
| `model_name` | Embedding model id used to select the prompt template, or `NULL` for the default. |

**Returns:** Zero on success, or a non-zero error code on failure.

### `moonshine_free_embedding()`

Frees an embedding returned by `moonshine_calculate_embedding()`.

```c
void moonshine_free_embedding(
    float *embedding
);
```

| Argument | Description |
| --- | --- |
| `embedding` | Pointer returned by `moonshine_calculate_embedding()` via `out_embedding`. |

**Returns:** Nothing.

### `moonshine_calculate_embedding_distance()`

Calculates the cosine similarity between two embedding vectors.

Both `embedding_a` and `embedding_b` must have `embedding_size` elements. The result is written to `*out_similarity` and is in the range [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite).

```c
int32_t moonshine_calculate_embedding_distance(
    int32_t embedding_model_handle,
    const float *embedding_a,
    const float *embedding_b,
    uint64_t embedding_size,
    float *out_similarity
);
```

| Argument | Description |
| --- | --- |
| `embedding_model_handle` | Handle returned by a `moonshine_create_embedding_model*` function. |
| `embedding_a` | First vector, with `embedding_size` elements. |
| `embedding_b` | Second vector, with `embedding_size` elements. |
| `embedding_size` | Number of elements in each vector. |
| `out_similarity` | Receives the cosine similarity, in the range [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite). |

**Returns:** Zero on success, or a non-zero error code on failure.

## Speech Clips

Extract a short, mostly-speech window from a recording for zero-shot voice cloning.

### `moonshine_extract_speech_clip()`

Finds the best short window of speech in a recording, for use as the reference clip in zero-shot voice cloning.

Extract is VAD-only and stays cheap enough for the capture loop. When ZipVoice is later created without `zipvoice_clone_transcript`, the owned clone ASR (from `g2p_root/clone_asr/` or `clone_asr/...` memory keys) refines the clip and fills the transcript once — see `moonshine_get_tts_dependencies()`.

The returned clip is always 16 kHz mono regardless of `sample_rate`.

Recognised `options` are listed under [Speech clip extract](options.md#speech-clip-extract).

```c
int32_t moonshine_extract_speech_clip(
    const float *audio_data,
    uint64_t audio_length,
    int32_t sample_rate,
    int32_t tts_synthesizer_handle,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    struct moonshine_speech_clip_t *out_clip
);
```

| Argument | Description |
| --- | --- |
| `audio_data` | Recording to search, as mono PCM between -1.0 and 1.0 at `sample_rate` Hz. |
| `audio_length` | Number of samples in `audio_data`. |
| `sample_rate` | Sample rate of the input audio, in Hz. The returned clip is always 16 kHz mono regardless. |
| `tts_synthesizer_handle` | A valid synthesizer from one of the `moonshine_create_tts_synthesizer_*` functions. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Speech clip extract](options.md#speech-clip-extract). |
| `options_count` | Number of entries in `options`. |
| `out_clip` | Receives the chosen window as a [`moonshine_speech_clip_t`](#moonshine_speech_clip_t). |

**Returns:** Zero on success, or a non-zero error code on failure. Success does not guarantee a clip: when no window holds at least `minimum_speech_seconds` of speech, `out_clip->is_complete` is zero and no audio is returned, so record more and call again.

## Text to Speech

Create a synthesizer, list voices and download dependencies, and synthesize from text or IPA phonemes.

### `moonshine_create_tts_synthesizer_from_files()`

Creates a text to speech synthesizer from files on disk. Pass construction options such as `voice` through the `options` array; see [Options → Text to Speech](options.md#text-to-speech). ZipVoice cloning details are on that page and under [`moonshine_create_tts_synthesizer_from_memory()`](#moonshine_create_tts_synthesizer_from_memory).

```c
int32_t moonshine_create_tts_synthesizer_from_files(
    const char *language,
    const char **filenames,
    uint64_t filenames_count,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `language` | Language tag for the voice, for example `"en_us"`. |
| `filenames` | Canonical asset keys to load from disk, resolved relative to `g2p_root`. |
| `filenames_count` | Number of entries in `filenames`. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Text to Speech options](options.md#text-to-speech). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative synthesizer handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_create_tts_synthesizer_from_memory()`

Creates a text to speech synthesizer from memory.

`filenames[i]` is the canonical `MoonshineTTSOptions::files` key (e.g. `kokoro/model.ort`, `kokoro/config.json`, `kokoro/voices/af_heart.kokorovoice`, `piper/onnx`, `piper/onnx.json`, `zipvoice/text_encoder.ort`, `zipvoice/fm_decoder.ort`, `zipvoice/vocoder.ort`, `zipvoice/tokens.txt`, `zipvoice/model.json`). For ZipVoice a caller-supplied reference clip is passed as key `zipvoice/clone_audio` (raw little-endian float32 mono PCM); set `zipvoice_clone_sample_rate` and, optionally, `zipvoice_clone_transcript`. When the transcript is omitted, supply `clone_asr/<stt-filename>` keys (from the ZipVoice TTS dependency `clone_asr` group) so the library can refine and auto-transcribe the clip with its owned ASR. When `memory[i]` is non-NULL and `memory_sizes[i]` > 0, that buffer is used as the asset bytes; the library does not copy it—keep the buffers valid until `moonshine_free_tts_synthesizer()`. When `memory[i]` is NULL or `memory_sizes[i]` is zero, the key string is also used as a path relative to `g2p_options.g2p_root` (from `options`), same as path-only map entries.

Other `options` are parsed like `moonshine_create_tts_synthesizer_from_files()`.

```c
int32_t moonshine_create_tts_synthesizer_from_memory(
    const char *language,
    const char **filenames,
    const uint64_t filenames_count,
    const uint8_t **memory,
    const uint64_t *memory_sizes,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `language` | Language tag for the voice, for example `"en_us"`. |
| `filenames` | Canonical `MoonshineTTSOptions::files` keys, as listed above. |
| `filenames_count` | Number of entries in `filenames`, `memory`, and `memory_sizes`. |
| `memory` | Asset bytes for each key, or `NULL` for that entry to load it from disk instead. Buffers are not copied and must stay valid until `moonshine_free_tts_synthesizer()`. |
| `memory_sizes` | Byte count for each entry in `memory`; zero also means "load this one from disk". |
| `options` | Same as [`moonshine_create_tts_synthesizer_from_files()`](#moonshine_create_tts_synthesizer_from_files); see [TTS options](options.md#text-to-speech). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative synthesizer handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_free_tts_synthesizer()`

Releases the resources used by a text to speech synthesizer.

```c
void moonshine_free_tts_synthesizer(
    int32_t tts_synthesizer_handle
);
```

| Argument | Description |
| --- | --- |
| `tts_synthesizer_handle` | Handle returned by a `moonshine_create_tts_synthesizer_*` function. |

**Returns:** Nothing.

### `moonshine_text_to_speech()`

Synthesizes text to speech. `options` / `options_count` are optional per-call overrides; currently only [`speed`](options.md#text-to-speech) is honored for the call duration. Pass NULL / 0 to use constructor defaults.

```c
int32_t moonshine_text_to_speech(
    int32_t tts_synthesizer_handle,
    const char *text,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    float **out_audio_data,
    uint64_t *out_audio_data_size,
    int32_t *out_sample_rate
);
```

| Argument | Description |
| --- | --- |
| `tts_synthesizer_handle` | Handle returned by a `moonshine_create_tts_synthesizer_*` function. |
| `text` | UTF-8 text to speak. |
| `options` | Optional per-call overrides; only [`speed`](options.md#text-to-speech) is honored. Pass `NULL` for the synthesizer's construction defaults. |
| `options_count` | Number of entries in `options`; pass `0` with a `NULL` array. |
| `out_audio_data` | Receives the synthesized mono PCM samples. Allocated with malloc; release with `moonshine_free_buffer()`. |
| `out_audio_data_size` | Receives the number of samples in `out_audio_data`. |
| `out_sample_rate` | Receives the sample rate of the synthesized audio, in Hz. |

**Returns:** Zero on success, or a non-zero error code on failure.

### `moonshine_phonemes_to_speech()`

Synthesizes speech directly from International Phonetic Alphabet (IPA) phonemes, skipping the grapheme-to-phoneme conversion that `moonshine_text_to_speech()` performs internally. `phonemes` should be an IPA string in the same format produced by `moonshine_text_to_phonemes()` (a grapheme-to-phonemizer created for the matching language). This lets callers inspect or edit the phonemes between the text-to-phonemes and phonemes-to-speech steps (e.g. to fix pronunciation of a name). The phonemes are normalized to the active vocoder's phoneme inventory before synthesis, so passing the raw `moonshine_text_to_phonemes()` output for the same language yields audio equivalent to `moonshine_text_to_speech()` on the original text.

`options` / `options_count` behave like `moonshine_text_to_speech()`: only [`speed`](options.md#text-to-speech) is honored for the call; pass NULL / 0 for defaults.

```c
int32_t moonshine_phonemes_to_speech(
    int32_t tts_synthesizer_handle,
    const char *phonemes,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    float **out_audio_data,
    uint64_t *out_audio_data_size,
    int32_t *out_sample_rate
);
```

| Argument | Description |
| --- | --- |
| `tts_synthesizer_handle` | Handle returned by a `moonshine_create_tts_synthesizer_*` function. |
| `phonemes` | IPA string in the same format produced by `moonshine_text_to_phonemes()` for the matching language. |
| `options` | Optional per-call overrides; only [`speed`](options.md#text-to-speech) is honored. Pass `NULL` for the synthesizer's construction defaults. |
| `options_count` | Number of entries in `options`; pass `0` with a `NULL` array. |
| `out_audio_data` | Receives the synthesized mono PCM samples. Allocated with malloc; release with `moonshine_free_buffer()`. |
| `out_audio_data_size` | Receives the number of samples in `out_audio_data`. |
| `out_sample_rate` | Receives the sample rate of the synthesized audio, in Hz. |

**Returns:** Zero on success, or a non-zero error code on failure.

### `moonshine_get_tts_dependencies()`

Returns merged G2P + TTS vocoder download dependencies as a JSON object with a `groups` array (same shape as `moonshine_get_stt_dependencies()`). Each group is `{ "base_url", "files": [{name,url,size,checksum,checksum_type}] }`. `languages` is comma-separated; empty or NULL means all known languages. `options` / `options_count`: same [TTS options](options.md#text-to-speech) as synthesizer create (`voice`, `g2p_root`, and related).

When `voice` selects ZipVoice, an additional group with `"role":"clone_asr"` lists the catalog-default STT for the language (including the attention decoder for word timestamps). Local `name`s are prefixed `clone_asr/`; `url`s point at the STT CDN. Bindings should download those files under `g2p_root/clone_asr/` (or pass `clone_asr/...` memory keys on create).

On success, `*out_dependencies_json` is a NUL-terminated JSON object; release with `moonshine_free_buffer()`.

```c
int32_t moonshine_get_tts_dependencies(
    const char *languages,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    char **out_dependencies_json
);
```

| Argument | Description |
| --- | --- |
| `languages` | Comma-separated language tags. Empty or `NULL` means all known languages. |
| `options` | Same entries as synthesizer creation; see [Text to Speech options](options.md#text-to-speech). |
| `options_count` | Number of entries in `options`. |
| `out_dependencies_json` | Receives a NUL-terminated JSON object with a `groups` array. Release with `moonshine_free_buffer()`. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure.

### `moonshine_get_tts_voices()`

Returns known TTS voices for the requested languages with availability state. `languages` is comma-separated; empty or NULL means all registered catalog languages (same tag set as G2P dependencies) that have a resolved TTS layout. `options` / `options_count`: [TTS options](options.md#text-to-speech) used for listing (`voice` selects vocoder catalog; set `g2p_root` / aliases for accurate found/missing). The `voice` option does not filter the returned list.

On success, `*out_voices_json` is a NUL-terminated JSON object mapping each language tag to a JSON array of objects `{"id":"<voice>","state":"found"}` or `{"id":"<voice>","state":"missing"}`. Voice ids are prefixed with `kokoro_` or `piper_`. Kokoro uses the upstream Kokoro-82M voice id catalog plus any extra `*.kokorovoice` in the bundle; Piper lists the language default voice stem plus every voice in the resolved voices directory, in either shipped form (`<stem>.ort`, or the split `<stem>.model.ort` plus `<stem>.weights.ort` pair). `found` means the asset is on disk or supplied via the in-memory file map like `MoonshineTTS`. Free with `moonshine_free_buffer()`.

```c
int32_t moonshine_get_tts_voices(
    const char *languages,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    char **out_voices_json
);
```

| Argument | Description |
| --- | --- |
| `languages` | Comma-separated language tags. Empty or `NULL` means every registered catalog language that has a resolved TTS layout. |
| `options` | Same entries as synthesizer creation; see [Text to Speech options](options.md#text-to-speech). Set `g2p_root` (or the `path_root` / `tts_root` / `model_root` aliases) so `found` / `missing` is accurate. |
| `options_count` | Number of entries in `options`. |
| `out_voices_json` | Receives a NUL-terminated JSON object mapping each language tag to an array of [voice entries](#tts-voice-catalog-entries). Release with `moonshine_free_buffer()`. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure.

## Grapheme to Phonemes

Create a G2P engine, convert text to IPA, and list G2P asset dependencies.

### `moonshine_create_grapheme_to_phonemizer_from_files()`

Creates a grapheme to phonemizer from files on disk.

Lexicons and bundled ONNX assets are resolved under `g2p_root` (or the process current working directory when `g2p_root` / `model_root` is unset) using the same canonical relative keys as `MoonshineG2POptions::files` in the C++ API (for example `en_us/dict_filtered_heteronyms.tsv`, `zh_hans/roberta_chinese_base_upos_onnx/meta.json`, `zh_hans/roberta_chinese_base_upos_onnx/model.model.ort`, `en_us/g2p-config.json`, `en_us/oov/model.ort`, `en_us/oov/onnx-config.json`). Japanese and Arabic tok-POS / diacritizer bundles use the same pattern under `ja/...` and `ar_msa/...`. Korean rule G2P uses `ko/dict.tsv` only. Models that ship as a split ORT pair need both `<stem>.model.ort` and `<stem>.weights.ort` present.

Every model is ORT-format. Moonshine cannot load a `.onnx`: the wasm and mobile runtimes are minimal ONNX Runtime builds with no ONNX parser compiled in. Convert one with `scripts/convert-models-to-ort.py`.

```c
int32_t moonshine_create_grapheme_to_phonemizer_from_files(
    const char *language,
    const char **filenames,
    uint64_t filenames_count,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `language` | Language tag to phonemize for, for example `"en_us"`. |
| `filenames` | Canonical `MoonshineG2POptions::files` keys to load from disk, resolved relative to `g2p_root`. |
| `filenames_count` | Number of entries in `filenames`. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Grapheme to Phonemes options](options.md#grapheme-to-phonemes). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative phonemizer handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_create_grapheme_to_phonemizer_from_memory()`

Creates a grapheme to phonemizer from memory.

`filenames[i]` is the canonical `MoonshineG2POptions::files` key. When `memory[i]` is non-NULL and `memory_sizes[i]` > 0, that buffer is used as the asset bytes (not copied—keep valid until the phonemizer is freed). When `memory[i]` is NULL or size zero, the key is also used as a path relative to `g2p_root`, like path-only map entries.

Register every file the engine needs: language lexicon `dict.tsv` paths, English `g2p-config.json` and the OOV model keys under `en_us/oov/`, and for model bundles the `meta.json`, `vocab.txt`, `tokenizer_config.json`, and `model.ort` keys under the bundle directory key (or both halves of a split pair). English OOV overrides use `oov_onnx_override` for the model bytes and `oov_onnx_config` for the merged JSON config UTF-8 text; those key names predate the move to ORT and are kept for compatibility, but the bytes must be ORT-format.

Every model buffer must be a self-contained ORT model. Moonshine cannot load a `.onnx`, and there is no support for a sidecar weights file: convert with `scripts/convert-models-to-ort.py`.

```c
int32_t moonshine_create_grapheme_to_phonemizer_from_memory(
    const char *language,
    const char **filenames,
    const uint64_t filenames_count,
    const uint8_t **memory,
    const uint64_t *memory_sizes,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    int32_t moonshine_version
);
```

| Argument | Description |
| --- | --- |
| `language` | Language tag to phonemize for, for example `"en_us"`. |
| `filenames` | Canonical `MoonshineG2POptions::files` keys, as described above. |
| `filenames_count` | Number of entries in `filenames`, `memory`, and `memory_sizes`. |
| `memory` | Asset bytes for each key, or `NULL` for that entry to load it from disk instead. Buffers are not copied and must stay valid until the phonemizer is freed. |
| `memory_sizes` | Byte count for each entry in `memory`; zero also means "load this one from disk". |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Grapheme to Phonemes options](options.md#grapheme-to-phonemes). |
| `options_count` | Number of entries in `options`. |
| `moonshine_version` | Set to `MOONSHINE_HEADER_VERSION` so a newer library emulates the behavior of the header you compiled against. |

**Returns:** A non-negative phonemizer handle on success, or a negative error code on failure. Convert the code with `moonshine_error_to_string()`.

### `moonshine_free_grapheme_to_phonemizer()`

Releases the resources used by a grapheme to phonemizer.

```c
void moonshine_free_grapheme_to_phonemizer(
    int32_t grapheme_to_phonemizer_handle
);
```

| Argument | Description |
| --- | --- |
| `grapheme_to_phonemizer_handle` | Handle returned by a `moonshine_create_grapheme_to_phonemizer_*` function. |

**Returns:** Nothing.

### `moonshine_text_to_phonemes()`

Converts a text into the equivalent International Phonetic Alphabet (IPA) phonemes.

```c
int32_t moonshine_text_to_phonemes(
    int32_t grapheme_to_phonemizer_handle,
    const char *text,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    const char **out_phonemes,
    uint64_t *out_phonemes_count
);
```

| Argument | Description |
| --- | --- |
| `grapheme_to_phonemizer_handle` | Handle returned by a `moonshine_create_grapheme_to_phonemizer_*` function. |
| `text` | UTF-8 text to convert. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Grapheme to Phonemes options](options.md#grapheme-to-phonemes). |
| `options_count` | Number of entries in `options`. |
| `out_phonemes` | Receives the IPA string. Allocated with malloc; release with `moonshine_free_buffer()`. |
| `out_phonemes_count` | Receives the length of the phoneme string. |

**Returns:** Zero on success, or a non-zero error code on failure.

### `moonshine_get_g2p_dependencies()`

Returns G2P-only canonical asset keys for one or more languages. `languages` is comma-separated CLI tags (same as `moonshine_create_*` `language`); an empty string (or NULL) means all known languages (union of keys). `options` / `options_count`: [G2P options](options.md#grapheme-to-phonemes). TTS-only keys are ignored. Non-empty in-memory override values add those canonical key names to the list. On success, writes a comma-separated list to `*out_dependencies_json` and returns `MOONSHINE_ERROR_NONE`. The buffer is allocated with `malloc`; release with `moonshine_free_buffer()`. On failure (e.g. unknown language token), logs and returns a non-zero error code and sets `*out_dependencies_json` to NULL.

```c
int32_t moonshine_get_g2p_dependencies(
    const char *languages,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    char **out_dependencies_json
);
```

| Argument | Description |
| --- | --- |
| `languages` | Comma-separated CLI language tags, the same ones the `moonshine_create_*` functions take. An empty string or `NULL` means all known languages. |
| `options` | Array of [`moonshine_option_t`](#moonshine_option_t). See [Grapheme to Phonemes options](options.md#grapheme-to-phonemes); TTS-only keys are ignored. |
| `options_count` | Number of entries in `options`. |
| `out_dependencies_json` | Receives a comma-separated list of canonical asset keys, despite the parameter name. Release with `moonshine_free_buffer()`. Set to `NULL` on failure. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure (for example an unknown language token).

## Model Download Manifests

Ask the library which CDN files a model needs, and enumerate published STT and embedding catalogs, without hardcoding layouts in bindings.

### `moonshine_get_stt_dependencies()`

Returns the download manifest for a speech-to-text transcription model as a JSON object. This lets language bindings and applications fetch exactly the files a model needs from the CDN (https://download.moonshine.ai) without hardcoding the file layout, then load the model from the resulting directory with `moonshine_load_transcriber_from_files()`.

`options` / `options_count` may match the load-time list; only keys that change the file set are honored — see [Download manifests](options.md#download-manifests).

On success, writes a NUL-terminated JSON object to `*out_dependencies_json` and returns `MOONSHINE_ERROR_NONE`. The shape is: `{"groups":[{"base_url":"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en","files":[{"name":"encoder_model.ort","url":"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/encoder_model.ort","size":12345,"checksum":"abc==","checksum_type":"crc32c"}, ...]}]}` Each entry in `files` is an object with `name` (canonical filename), `url` (fully-qualified download URL, i.e. `base_url + "/" + name`), `size` (bytes, or null when unknown), `checksum` (base64 digest, or ""), and `checksum_type` (e.g. "crc32c", or ""). A model is a single group, plus an optional second group for the spelling model (which uses a different `base_url`). The buffer is allocated with `malloc`; release it with `moonshine_free_buffer()`. On failure (empty/unknown language, or an unknown language+arch pair) returns a non-zero error code and sets `*out_dependencies_json` to NULL.

```c
int32_t moonshine_get_stt_dependencies(
    const char *language,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    char **out_dependencies_json
);
```

| Argument | Description |
| --- | --- |
| `language` | A language code (for example `"en"`) or English name (for example `"English"`). Must not be empty. |
| `options` | See [Download manifests](options.md#download-manifests). |
| `options_count` | Number of entries in `options`. |
| `out_dependencies_json` | Receives a NUL-terminated JSON manifest in the shape described above. Release with `moonshine_free_buffer()`. Set to `NULL` on failure. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure (an empty or unknown language, or an unknown language and architecture pair).

### `moonshine_get_embedding_dependencies()`

Returns the download manifest for an embedding model as a JSON object with the same shape as `moonshine_get_stt_dependencies()`. Load the downloaded directory with `moonshine_create_embedding_model()`.

`options` / `options_count` accept [`variant` / `model_variant`](options.md#embeddings). The manifest lists the single all-in-one model file (`model_<variant>.ort`) and `tokenizer.bin`.

On success, writes a NUL-terminated JSON object to `*out_dependencies_json` (single group, same file-object shape as `moonshine_get_stt_dependencies()`) and returns `MOONSHINE_ERROR_NONE`; release with `moonshine_free_buffer()`. On failure (unknown model or variant) returns a non-zero error code and sets `*out_dependencies_json` to NULL.

```c
int32_t moonshine_get_embedding_dependencies(
    const char *model_name,
    const struct moonshine_option_t *options,
    uint64_t options_count,
    char **out_dependencies_json
);
```

| Argument | Description |
| --- | --- |
| `model_name` | An embedding model id (for example `"embeddinggemma-300m"`). Pass `NULL` or an empty string for the default model. |
| `options` | Recognizes [`variant` / `model_variant`](options.md#embeddings); other keys are ignored. |
| `options_count` | Number of entries in `options`. |
| `out_dependencies_json` | Receives a NUL-terminated JSON manifest with a single group. Release with `moonshine_free_buffer()`. Set to `NULL` on failure. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure (an unknown model or variant).

### `moonshine_get_diarization_dependencies()`

Returns the download manifest for the speaker diarization models as a JSON object with the same shape as `moonshine_get_stt_dependencies()`. Fetch these whenever you intend to pass `identify_speakers=true` to a transcriber, and point the transcriber at them with the `diarization_model_dir` option (or supply them as `segmentation.ort` / `embedding.ort` entries to `moonshine_load_transcriber_from_memory_files()`).

There is one set of diarization models and it has no variants, so this takes no arguments beyond the output pointer. The manifest is a single group of two files totalling about 8.2 MB.

These models were compiled into the library before version 26.8; a transcriber built with `identify_speakers=true` and no diarization models now fails to load rather than falling back. See docs/diarization-models.md.

The buffer is allocated with `malloc`; release it with `moonshine_free_buffer()`. Returns `MOONSHINE_ERROR_NONE` on success.

```c
int32_t moonshine_get_diarization_dependencies(
    char **out_dependencies_json
);
```

| Argument | Description |
| --- | --- |
| `out_dependencies_json` | On success, set to a NUL-terminated JSON object. Release with `moonshine_free_buffer()`. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure.

### `moonshine_get_stt_catalog()`

Returns the full speech-to-text model catalog as a JSON object, so bindings can build language/model pickers and resolve defaults without their own copy of the tables. The shape is: `{"languages":[{"code":"en","english_name":"English","models":[{"model_arch":9,"download_url":"https://...","is_default":true}, ...]}, ...]}` The buffer is allocated with `malloc`; release it with `moonshine_free_buffer()`. Returns `MOONSHINE_ERROR_NONE` on success.

```c
int32_t moonshine_get_stt_catalog(
    char **out_catalog_json
);
```

| Argument | Description |
| --- | --- |
| `out_catalog_json` | On success, set to a NUL-terminated JSON catalog object. Release with `moonshine_free_buffer()`. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure.

### `moonshine_get_embedding_catalog()`

Returns the full text embedding model catalog as a JSON object. The shape is: `{"models":[{"name":"embeddinggemma-300m","english_name":"Embedding Gemma 300M","download_url":"https://...","variants":["q4", ...],"default_variant":"q4"}]}` The buffer is allocated with `malloc`; release it with `moonshine_free_buffer()`. Returns `MOONSHINE_ERROR_NONE` on success.

```c
int32_t moonshine_get_embedding_catalog(
    char **out_catalog_json
);
```

| Argument | Description |
| --- | --- |
| `out_catalog_json` | On success, set to a NUL-terminated JSON catalog object. Release with `moonshine_free_buffer()`. |

**Returns:** `MOONSHINE_ERROR_NONE` on success, or a non-zero error code on failure.
