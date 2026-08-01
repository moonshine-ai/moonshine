# Moonshine loads ORT-format models only

Moonshine used to accept a model in either format: ONNX (`.onnx`) or the
OnnxRuntime flatbuffer encoding (`.ort`). It now accepts only the latter,
everywhere, on every platform. This is a breaking change for anyone supplying
their own model files.

## Why

The WebAssembly, iOS and Android builds link a *minimal* ONNX Runtime, cut down
to the operators our models actually use. A minimal build has no ONNX parser
compiled into it at all, so on those platforms a `.onnx` could never have been
read, whatever the calling code did. See
[the wasm README](../wasm/README.md#the-minimal-build-and-what-it-costs-you)
for what that buys.

That left two formats supported on desktop and one on mobile, which is the
worst arrangement available: code that worked on a developer's Mac failed in
the browser or on a phone, and the failure surfaced as an opaque ONNX Runtime
parse error a long way from the cause. Rejecting `.onnx` on every platform
means one behaviour everywhere, and you find out at your desk.

## What changed

Models we ship were converted some time ago and need nothing from you. What
changes is model files you supply yourself:

- The `piper_onnx` / `piper_model_onnx` options and the `piper/onnx` in-memory
  key, for a custom Piper voice.
- The `oov_onnx_override` option and key, for a custom English OOV model.
- Embedding model directories downloaded before the `.ort` migration, which
  hold a `model_<variant>.onnx` plus a `.onnx_data` sidecar.
- Any `.onnx` sitting beside a model we resolve by name, which used to be
  picked up when no `.ort` was present.

The option and key names keep their old spelling so existing calling code
compiles and runs unchanged. Only the file format they accept has narrowed.

Support for ONNX external data (`model.onnx.data`, `model.onnx_data`) is gone
with it. An `.ort` is self-contained, so a model that needed a sidecar has to
be converted into one file.

## Migrating

Convert each model once and point at the result:

```bash
python scripts/convert-models-to-ort.py path/to/model.onnx
```

For an embedding model directory, re-download it rather than converting by
hand; the published directories have shipped `.ort` for a while.

A `.onnx` now fails with a message naming the file and this command, rather
than a parse error. If you see one at startup, that is what it is telling you.

## What did not change

- Piper's config sidecar keeps its upstream `<voice>.onnx.json` name whatever
  format the model ships in. It is JSON, not a model, and is unaffected.
- Bundle directories keep names like
  `zh_hans/roberta_chinese_base_upos_onnx/`. They are directory names.
- You can still request a voice by its upstream `<voice>.onnx` name; that is a
  voice identifier, not a filename, and it resolves to the `.ort` on disk.
