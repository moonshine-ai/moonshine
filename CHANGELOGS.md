# Changelog

All notable user-facing changes to Moonshine Voice are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Speaker attribution can be corrected by hand in the meeting notes web example. A speaker's name is the boundary it sits on: drag it into the words above or below to move where that speaker starts, cutting a line in two if the join falls partway through one; Alt-drag to leave a second name partway through a turn; Alt-click the words to start a speaker diarization never separated out at all; and drag a name past its own last word to be rid of it. Hand-made corrections are pinned, so the next diarization pass over that window does not undo them.
- Two speakers given the same name in the meeting notes web example are read as one person: where their turns run together, they are laid out as a single turn under a single name, and renaming it renames all of them. Diarization splits one voice in two often enough — someone turns away from the microphone, or rejoins on a worse line — and typing the same name over both is the whole correction. Only names typed by hand count, so two speakers the page has failed to tell apart are never quietly merged on the strength of a made-up "Speaker 3".
- The meeting notes web example asks before closing a tab holding a transcript nobody has copied or exported since it last changed. Nothing is uploaded, so the tab is the only copy there is.
- Runtime key-term biasing for streaming speech-to-text. Pass `keyterms` (a comma-separated list) to steer the decoder towards jargon, product names, or proper nouns with no retraining, and tune the strength with `keyterm_boost` (default 2.0). Terms can be replaced while audio is streaming, so an app can follow whatever context the user is in: `Transcriber.set_keyterms()` in Python, `setKeyterms()` in Swift, Java and JavaScript, and `moonshine_transcriber_set_keyterms` in the C API. On a biasing test set built from LibriSpeech test-clean, with a realistic hundred terms live per utterance, the default removes an eighth to a sixth of the errors on the listed words for about a tenth of a point on everything else, and all three streaming models agree on where to set it. A 100-term list adds about a millisecond to end-of-phrase latency on a physical iPad (A16), so it needs no extra latency budget. Longer lists cost accuracy rather than time: a hundred terms nobody says cost a third of a point of general WER, ten thousand cost a point and a half. See [Domain Customization](README.md#domain-customization) for the measured curves, `scripts/make-keyterm-testset.py` to build a test set out of any ASR corpus, `scripts/eval-keyterm-biasing.py` to sweep the boost on it, and `scripts/eval-librispeech.py --keyterms-file` for the effect on general accuracy.

### Changed

- Encoding text to tokens no longer scans the whole vocabulary for every subword, which made installing a large key-term list slow: 10,000 terms took about 55 seconds to compile on an iPad and now takes around a second. Transcription output is unaffected, since it only ever decodes.

### Fixed

- The native library is now built optimized everywhere. CMake adds no optimization flag at all unless a build type is chosen, and the pip wheels, the published binary archives and the Android debug variant were all configured without one, so they shipped code compiled at debug speed. ONNX Runtime is prebuilt and was never affected, so model inference kept its speed and the gap went unnoticed; what was slow was everything around it, and the more work a feature does outside the model the more it lost. On a Pixel 10a this made contextual biasing look about three times as expensive as it is. Streaming latency on the Pixel improved by 8-15% from the fix alone, and the README's measured latencies have been refreshed (the Linux x86 and Raspberry Pi 5 columns still await hardware to re-measure on).

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
