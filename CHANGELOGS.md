# Changelog

All notable user-facing changes to Moonshine Voice are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Please keep the bullets high level, and no more than about 200 characters.

## [0.1.5]

### Added

- Streaming frontend graphs can load as `frontend.model.ort` plus `frontend.weights.ort`, keeping int8 weights on disk. English streaming models on `quantized_26_08_21` cut the frontend download by about 75%.
- `moonshine-voice[finetune]` is an alias of `[lora]`; `moonshine-voice finetune` runs the same trainer.
- Built-in `--dataset uwb_atcc` (real VHF, research/NC), `--sites encoder|both`, `--adapt full`, and `--eval-dataset atco2`.
- Streaming text to speech: push text as it arrives with `stream()` / `push_text` / `flush`, and take audio back chunk by chunk or through an `on_chunk` handler.
- Streaming starts speaking part way through a sentence rather than waiting for the whole one, so the first audio arrives sooner on long replies. Piper's sub-sentence chunks are sample-exact.
- Cancelling a streamed reply now reports a cancelled status to whoever is pulling chunks, so an interrupted reply is distinguishable from one that ran out of text.
- `AgentFlow.say_stream()` and `Dialog.say_stream()` speak a language model's reply as it is generated, instead of waiting for the whole thing.
- `moonshine_tts_split_utterances()` exposes the sentence splitter that streaming and `say()` share.
- `EmbeddingModel` is a public low-level type in Python, JavaScript, Swift, and Java, for embedding text and scoring similarity without adopting AgentFlow.
- Japanese streaming speech-to-text in small and tiny sizes, and small streaming is now what `"ja"` selects by default. The older non-streaming Japanese models stay available by architecture.

### Changed

- Streaming speech-to-text models open faster by reusing mapped `.ort` bytes at session create instead of copying them, matching the non-streaming path.
- Importing `moonshine-voice` no longer loads `requests`, so a CLI call against a cached model starts sooner.
- Text to speech now splits sentences with one shared implementation that keeps `Dr. Smith` and `J. R. R. Tolkien` whole and understands `。！？؟।` terminators, replacing four different naive rules.
- The `fp32`, `fp16`, and `q4f16` embedding model variants are no longer supported. Passing them fails with a "no longer supported" error; use `q4` (the default) or `q8`.
- Text to speech holds the audio device open across queued utterances, so consecutive `say()` calls no longer leave a gap or a click between them.
- Kokoro text to speech is 4x faster on short sentences and 2x on long ones on a Raspberry Pi 4, a closer match to the reference voice, and a 10 MB smaller download. It uses about 85 MB more memory.
- Kokoro now ships as two stages, `kokoro/prosody.*` and `kokoro/decoder.*`, and no whole-utterance model. They render the same audio, so the download halves. In-memory callers should pass all four keys.
- Every Piper voice now ships as `<stem>.upstream.*` plus `<stem>.generator.*`, the same total size and the same audio, which is what lets a reply start playing before it is synthesized.
- Streamed speech is levelled from a per-voice measurement instead of being left unnormalized, so it no longer arrives several decibels quieter than `say()` and the same across voices.
- Domain customization docs and the fine-tune Colab no longer call ATCOSIM radio. Phraseology stays the default walkthrough; VHF is a separate command. The example lives under `examples/python/finetune/`.

### Fixed

- `moonshine_get_stt_dependencies` now says whether the language is unknown or the architecture is unpublished, and lists that language's architectures (GitHub issue #214).
- Streaming no longer logs `Memory is empty` or drops hypotheses when short chunks arrive faster than encoder lookahead, including on medium-streaming (GitHub issue #218).
- AgentFlow no longer downloads the q4 embedding model and then tries to open the fp32 file, which crashed `load()` (GitHub issue #210).
- Destroying a file-backed Transcriber now unmaps every `.ort` it opened, so creating and closing one in a loop no longer retains tens of megabytes per instance.
- Kokoro text to speech failed to load in WebAssembly after the faster model landed, because the runtime was built without the quantized operators it needs.
- A native error in WebAssembly now reports its message instead of a heap address.

## [0.1.3]

### Added

- `decode_incomplete_lines` (default true). Set false to encode as audio arrives but wait until the line is complete before decoding.
- Optional `moonshine-voice[lora]` extra trains a decoder-only LoRA adapter on your audio (ATCOSIM example included). Default inference installs are unchanged.

### Changed

- The LoRA Colab notebook calls the same `fit_adapter` and ATCOSIM helpers as `python -m moonshine_voice.lora` instead of inlining the trainer.
- Streaming speaker diarization analyzes at most one segmentation window per audio append (Stop still drains the rest) and skips embedding inference on silent speaker classes.
- Meeting Notes waits until a line is complete before decoding, and writes each finished line on its own line in the document.

### Fixed

- Meeting Notes playback no longer clicks from dropped capture frames, resampler phase jumps, or mixing a second copy of the meeting that the microphone heard.
- C API streaming comments no longer refer to an undeclared `out_transcript`, a missing `moonshine-test-v2.cpp`, or the old `transcribe_stream_chunk` name.
- `transcribe_stream` after `stop_stream` now transcribes leftover audio, so a final transcript no longer requires earlier partial updates.

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
