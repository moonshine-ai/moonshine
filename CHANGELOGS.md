# Changelog

All notable user-facing changes to Moonshine Voice are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
