# Changelog

All notable user-facing changes to Moonshine Voice are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Please keep the bullets high level, and no more than about 200 characters.

## [0.1.3]

### Added

- `decode_incomplete_lines` (default true). Set false to encode as audio arrives but wait until the line is complete before decoding.

### Changed

- Streaming speaker diarization analyzes at most one segmentation window per audio append (Stop still drains the rest) and skips embedding inference on silent speaker classes.
- Meeting Notes waits until a line is complete before decoding, and writes each finished line on its own line in the document.

### Fixed

- Meeting Notes playback no longer clicks from dropped capture frames, resampler phase jumps, or mixing a second copy of the meeting that the microphone heard.

## [0.1.2] - August 13th, 2026

### Added

- Runtime domain customization for streaming speech-to-text: pass `keyterms` to bias decoding towards jargon, or `context` to find the terms in a passage of text. See [Domain Customization](docs/models/domain-customization.md).
- Documentation has been reorganized in `mkdocs` style, with one file per section rather than everything in one large README.md. These docs are also available at [moonshine.readthedocs.io](https://moonshine.readthedocs.io).

### Changed

- `moonshine_load_transcriber_from_memory_files()` rejects an unrecognized filename key with `MOONSHINE_ERROR_INVALID_ARGUMENT`, naming it, instead of dropping it silently and reporting the file as missing.

### Fixed

- Meeting Notes no longer freezes for seconds when returning to its tab during a long recording: capture audio is batched and the main thread yields while catching up.
- Core library builds no longer write into the source tree, where targets clobbered each other's archives, and the wheels, archives and Android debug variant are now optimized rather than debug builds (8-15% faster streaming on a Pixel 10a).

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
