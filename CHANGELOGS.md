# Changelog

All notable user-facing changes to Moonshine Voice are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Speaker attribution can be corrected by hand in the meeting notes web example. A name marks where its speaker starts: drag it into the words above or below to move that boundary, Alt-drag to start a second speaker partway through a turn, Alt-click to add a speaker diarization never separated out, and drag a name past its own last word to remove it. Corrections are pinned, so the next diarization pass does not undo them.
- Two speakers given the same name in the meeting notes web example are treated as one person: adjacent turns merge under a single name, and renaming one renames all. Only hand-typed names count, so speakers the page failed to tell apart are never merged on their own.
- The meeting notes web example asks before closing a tab holding a transcript that has not been copied or exported since it last changed. Nothing is uploaded, so the tab is the only copy.
- Key terms can be estimated from a passage of text — a document, an agenda, the last few messages in a thread — instead of listed by hand. Pass `context` at load time, or replace it on a running transcriber with `set_context()` in Python, `setContext()` in Swift, Java and JavaScript, or `moonshine_transcriber_set_context()` in C. Candidates are picked with the model's own tokenizer, ranked by frequency and capped at 200 by default; multi-word and alphanumeric terms still need `keyterms`.
- Runtime key-term biasing for streaming speech-to-text. Pass `keyterms` (a comma-separated list) to steer the decoder towards jargon, product names, or proper nouns with no retraining, tune the strength with `keyterm_boost` (default 2.0), and replace terms mid-stream with `set_keyterms()` in Python, `setKeyterms()` in Swift, Java and JavaScript, or `moonshine_transcriber_set_keyterms` in C. With a hundred terms live the default removes an eighth to a sixth of the errors on those words for a tenth of a point on everything else, and costs about a millisecond of end-of-phrase latency on an iPad (A16). See [Domain Customization](README.md#domain-customization) for the measured accuracy curves at longer list lengths.
- Android's `MicTranscriber` takes native transcriber options through a chained `options()` setter, so an app can start listening with key terms, VAD tuning or speaker identification already in place. The other bindings already had this; on Android anything the setters did not cover meant building a `Transcriber` by hand and giving up the microphone plumbing. Options are read as the model loads, so calling it afterwards throws instead of quietly doing nothing — use `setKeyterms()` to change terms on a model that is already running.

### Changed

- Encoding text to tokens no longer scans the whole vocabulary for every subword: a 10,000-term key-term list took about 55 seconds to compile on an iPad and now takes around a second. Transcription output is unaffected, since it only ever decodes.
- `moonshine_load_transcriber_from_memory_files()` rejects an unrecognized filename key with `MOONSHINE_ERROR_INVALID_ARGUMENT`, naming the key, instead of dropping it silently and then reporting the required file as missing. Keys from every architecture are accepted, so handing over a whole downloaded model directory still works.

### Fixed

- Builds of the core library no longer write into the source tree. Each of its subprojects — `ort-utils`, `moonshine-utils`, `bin-tokenizer`, `moonshine-tts` and the rest — used a fixed directory like `core/ort-utils/build`, which every configuration shared: an Android ABI, an iOS slice, a wasm build and the host all wrote the same archives, so whichever ran last left objects the next one tried to link, failing with `is incompatible with aarch64linux` or `is neither Wasm object file nor LLVM bitcode`. They now build inside whichever build tree is configuring them, which makes concurrent builds for different targets safe and means clearing that one directory is a full clean. Anything that reached into the old locations for a built artifact should look under the build directory instead, as in `core/build/bin-tokenizer/bin-tokenizer-test`.
- The native library is now built optimized everywhere. The pip wheels, published binary archives and Android debug variant were all built with no CMake build type, so everything outside the prebuilt ONNX Runtime ran at debug speed — on a Pixel 10a this made contextual biasing look about three times as expensive as it is. Streaming latency on the Pixel improved by 8-15%, and the README's measured latencies have been refreshed (Linux x86 and Raspberry Pi 5 await hardware to re-measure on).

## [0.1.1] - August 6th, 2026

### Added

- Speculative decoding for streaming speech-to-text, on by default. Streaming re-decodes verify the previous hypothesis and continue from the first mismatch instead of restarting from BOS, cutting live latency on several platforms (for example Medium Streaming on MacBook Pro from ~103ms to ~74ms). Disable with `use_speculative_decoding=false`.
- `AgentFlow.otherwise()` — a callback for speech that matched no trigger and no waiting prompt, so dictation-style UIs can take free-form lines without treating them as failed commands.
- Cross-platform voice cloning API: `cloning()`, `clone_from()`, and `start_cloning()` / `VoiceClone` for file, PCM, or live microphone capture, aligned across Python, JavaScript, Swift, and Java.
- AgentFlow and voice-cloning examples on Android and iOS/Swift, plus dictation and meeting note taker web examples.
- Language tabs on the web example code snippets.
- Hugging Face mirror of every downloadable asset at [moonshine-ai/moonshine-voice-assets](https://huggingface.co/moonshine-ai/moonshine-voice-assets).

### Changed

- **Breaking:** `DialogFlow` is renamed to `AgentFlow` everywhere (APIs, packages, and example apps).
- **Breaking:** High-level client APIs are regularized across Python, JavaScript, Swift, and Java around construct → configure with chainable setters → `load()`. In particular, Python `MicTranscriber` and `TextToSpeech` no longer download and open models inside the constructor.
- **Breaking:** The C++ binding (`moonshine-cpp.h`) follows the same higher-level shape for transcription and TTS.
- **Breaking:** Only OnnxRuntime flatbuffer models (`.ort`) are accepted. Supplying `.onnx` (or ONNX external-data sidecars) fails with a clear migration error. Convert with `python scripts/convert-models-to-ort.py`. See [docs/ort-only-models.md](docs/ort-only-models.md).
- **Breaking:** Speaker-diarization models are no longer embedded in the library. With `identify_speakers`, they download on first use like other models (~8 MB off every mobile binary; Android arm64 install ~24.6 MB → ~16.4 MB, iOS linked binary ~30.6 MB → ~22.4 MB). See [docs/diarization-models.md](docs/diarization-models.md).
- Downloadable assets are served from `download.moonshine.ai` (and the Hugging Face mirror) instead of the old GCP-hosted buckets; large model blobs are no longer kept in git LFS in this repo.
- Smaller WebAssembly downloads after ORT-only packaging and cleanup of unused model files.
- Improved quantization accuracy for shipped models.
- Streaming transcription backs off the inference interval when it falls behind realtime, to reduce piled-up work.

### Removed

- Remaining public references to the old Intent API (use `AgentFlow` phrase matching instead).
- Embedded diarization model payloads from the core library binary.

### Fixed

- Windows MSVC path handling for OnnxRuntime sessions and file opens.
- Clearer errors when deprecated or unsupported model formats are supplied.
