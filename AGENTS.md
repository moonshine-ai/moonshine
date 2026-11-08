# Contributor notes for agents

This file is for people (and coding agents) working **in this repository**. App developers integrating the published library should use [`.agents/skills/moonshine-voice/SKILL.md`](.agents/skills/moonshine-voice/SKILL.md) instead — do not mix those audiences.

## Branching

`main` only ever contains released code. Development happens on a `dev-v<version>` candidate branch. Pull requests target that candidate, not `main`. See [docs/release-process.md](docs/release-process.md). Do not start or publish a release unless the user explicitly asked.

## Building and tests

Large model and TTS binaries are not in git. Before `scripts/test-core.sh` or offline TTS work:

```bash
scripts/fetch-voice-assets.sh all
```

A few compile-time embeds and ONNX Runtime prebuilts still use Git LFS. If a compile fails with something like `'version' does not name a type` on an `*_embedded.cpp` file, run `git lfs pull`.

Useful scripts (prefer these over ad-hoc cmake/pytest invocations):

- `scripts/test-core.sh` — C++ core build and tests
- `scripts/test-python.sh` — Python bindings
- `scripts/test-wasm.sh` — WebAssembly bindings
- `scripts/test-docs.sh` — documentation snippet tests
- `scripts/format-core.sh` — Google-style clang-format for first-party `core/`
- `scripts/check-banned-constructs.sh` — C++ construct gates

C++ policy lives in [core/STYLE_GUIDE.md](core/STYLE_GUIDE.md): C++20, RAII, no new owning `new`/`delete` or `reinterpret_cast` outside the baseline allow-list, no unsafe C string functions. Do not format or lint `core/third-party/` or `core/cpp-annote/`.

## Streaming model

- Every field of `MoonshineStreamingState` is per-stream and must be cleared in `reset()`; `scripts/check-streaming-state-reset.sh` (a ctest) fails otherwise. A field that survives reset leaks one stream's audio into the next (issues #229, #231).
- `decode_full` output is defined as greedy decoding: every emitted token is the argmax given the tokens before it. Speculative verification and realignment only change how many decoder calls that takes, never the tokens. Keep the control loop in `core/speculative-realign.h` free of ONNX Runtime so `streaming-model-test` can drive it with a scripted decoder.
- Before changing `encode`, `compute_cross_kv`, or `decode_full`, run `core/build/streaming-latency-bench` from `core/` on tiny and medium (`-m` twice to compare a stateless and a cached `encoder.ort`). Report mean/p90, decoder calls, divergence rate, and the transcript-difference count against the baseline row; the dynamically quantized graphs make a few percent of updates differ between any two numerically distinct paths, so compare WER on a corpus (`--transcripts` writes one line per update for scoring) rather than demand identical text.
- Every published streaming pin from `quantized_26_11_07` on carries the cached encoder graph (`lora --export --encoder-cache`); older pins are stateless and still load. The runtime tells them apart by the encoder's input count and reads the cache width from the graph. `test-assets/tiny-streaming-en-stateless` (fetched by `scripts/fetch-voice-assets.sh`) is the stateless reference `streaming-model-test` compares against.
- Checkpoints store sliding windows in two conventions; `inclusive_windows()` in `lora/export.py` normalises them. Any new export must be checked against the published stateless encoder's memory and scored on in-language audio before it is pinned.
- Publishing a model is three steps, in order: `scripts/upload-streaming-model.sh` to the CDN, point the catalog at the new dated directory and run `scripts/generate-model-file-metadata.py`, then `scripts/mirror-model-to-hf.sh` so the archive mirror stays complete.

## Public API

Language bindings follow construct → chainable setters → `load()`. Constructors are cheap and cannot fail. Do not put downloads or model opens in a constructor. The high-level types are `MicTranscriber`, `AgentFlow`, and `TextToSpeech`; `Transcriber` is the low-level PCM path and `EmbeddingModel` is the low-level text-embedding path. `DialogFlow` and the Intent API are gone.

Only OnnxRuntime flatbuffer models (`.ort`) are accepted. Do not add `.onnx` loading paths.

User-facing changes go in [CHANGELOGS.md](CHANGELOGS.md): Keep a Changelog, high-level bullets, no more than about 200 characters each.

## Layout

- `core/` — C++ engine and C API (`moonshine-c-api.h`)
- `language-bindings/` — Python, WASM, Swift, Android
- `docs/` — mkdocs source, published at https://moonshine-voice.readthedocs.io
- `examples/` — per-platform sample apps (the fine-tune notebook is `examples/python/finetune/`; the trainer is `moonshine_voice.lora` / `moonshine-voice finetune`)
- `micro/` — tiny on-device models, separate from the main library
