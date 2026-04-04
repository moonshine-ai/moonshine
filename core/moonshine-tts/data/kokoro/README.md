# Kokoro — bundled TTS for `moonshine_tts`

This directory is a **Kokoro ONNX bundle** layout (`kokoro/` under your asset root or current working directory). It must contain at least:

| Path | Role |
|------|------|
| `config.json` | Model config (includes phoneme `vocab` for ONNX). |
| `model.onnx` | Kokoro-82M acoustic ONNX (this repo often ships the **8-bit–quantized** build from [onnx-community/Kokoro-82M-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-ONNX) `onnx/model_quantized.onnx`, patched for C++; or a **local FP32** export from `scripts/download_kokoro_onnx.py`). |
| `voices/*.kokorovoice` | Style tensors for ONNX inference (C++ cannot load Hugging Face `voices/*.pt` pickles). |

Optional in a **build tree** (not required under `data` if you only ship C++): `kokoro-v1_0.pth` (PyTorch weights, for re-exporting ONNX), `voices/*.pt` (source for `.kokorovoice`), `onnx_export_meta.json` (written by the download/export script).

## Provenance

| Asset | Source |
|--------|--------|
| **Weights, config, native voices (`voices/*.pt`)** | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Hugging Face (see upstream **VOICES.md** for voice IDs and locales). |
| **`model.onnx`** | Either: **(A)** `scripts/fetch_hf_kokoro_quantized_onnx.py` — downloads [onnx-community/Kokoro-82M-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-ONNX) `onnx/model_quantized.onnx` (~92 MiB) and renames input `style` → `ref_s` for `MoonshineTTS`; or **(B)** `scripts/download_kokoro_onnx.py` — local FP32 export via **`kokoro`** (`KModel`, `KModelForONNX`) with **`disable_complex=True`**. |
| **`*.kokorovoice`** | Produced by `scripts/export_kokoro_voice_for_cpp.py` from each `voices/*.pt` tensor pack. Format: magic `KVO1`, little-endian `uint32` rows/cols, row-major `float32` data (after squeezing singleton dims to shape `[N, 256]`). |

`MoonshineTTS` passes **`speed`** as float32 `[1]` or double scalar depending on the graph (detected at load time; ONNX Runtime requires `GetInputTypeInfo(i).GetONNXType() == ONNX_TYPE_TENSOR` before reading the element type).

Python TTS in the parent monorepo (`speak.py`) can use the same HF bundle or PyTorch weights; the C++ path is **ONNX + `.kokorovoice` only**.

### Optional: onnx-shrink-ray weight packing

The same **onnx-shrink-ray** `quantize_weights` path used for Arabic BERT (int8 weight storage + dequant,
`float_quantization=False`) **does not preserve numeric parity** for Kokoro: `download_kokoro_onnx.py --verify`
drops from correlation ~0.997 (FP32) to ~0.1 after shrink, so **the bundled `data/kokoro/model.onnx`
stays FP32**. For experiments you can run:

```bash
python scripts/download_kokoro_onnx.py --out-dir data/kokoro --only-shrink
# or after export:  --shrink-weights
```

Expect a smaller file (~80 MiB vs ~310 MiB) but **validate audio** before shipping; ORT dynamic MatMul/Gemm
quantization (`--experimental-int8`) is also known to break prosody (see script docstring).

## Dependencies (rebuild)

```bash
pip install kokoro torch onnx onnxruntime onnxruntime-extensions huggingface_hub
# optional weight-pack experiments: pip install onnx-shrink-ray onnx-graphsurgeon
```

Versions drift over time; if export fails, align with the `kokoro` release compatible with the checkpoint (see HF model card).

## Install prebuilt quantized ONNX (smaller bundle)

From the repo root (requires `pip install huggingface_hub onnx`):

```bash
python scripts/fetch_hf_kokoro_quantized_onnx.py --backup
```

`--backup` saves any existing `model.onnx` as `model.onnx.fp32.bak`. To restore FP32 after experimenting: `cp model.onnx.fp32.bak model.onnx`.

## Rebuild everything (recommended flow)

From the **repository root**:

1. **Download weights, config, voice `.pt` files, and export ONNX + `.kokorovoice`** into a staging directory (example: `models/kokoro`):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir models/kokoro --verify
   ```

   `--verify` runs a numeric parity check (PyTorch vs ONNX) and is **optional** but recommended after toolchain upgrades.

   To fetch only some voices (faster):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir models/kokoro --voices af_heart,jf_alpha
   ```

   To write **directly** into this bundle (overwrites same paths):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir data/kokoro --verify
   ```

   Use `--skip-kokorovoice-export` if you only want ONNX/config/`.pt` without regenerating `.kokorovoice`.

2. **Install into `data/kokoro`** if you built under `models/kokoro`:

   ```bash
   mkdir -p data/kokoro/voices
   cp -a models/kokoro/config.json models/kokoro/model.onnx data/kokoro/
   cp -a models/kokoro/voices/*.kokorovoice data/kokoro/voices/
   ```

## Rebuild only `.kokorovoice` files

If you already have `voices/*.pt` (from HF or a previous download) but need to refresh C++ sidecars:

```bash
python scripts/export_kokoro_voice_for_cpp.py --voices-dir data/kokoro/voices
# or
python scripts/export_kokoro_voice_for_cpp.py --voices-dir models/kokoro/voices
```

Single file:

```bash
python scripts/export_kokoro_voice_for_cpp.py \
  models/kokoro/voices/af_heart.pt \
  data/kokoro/voices/af_heart.kokorovoice
```

## Rebuild only `model.onnx` (weights unchanged)

```bash
python scripts/download_kokoro_onnx.py --out-dir models/kokoro --skip-download
```

Then copy `model.onnx` (and updated `onnx_export_meta.json` if present) into `data/kokoro/` as needed.

## Sanity checks after a rebuild

1. **Voice export (needs PyTorch + one `*.pt`):**

   ```bash
   python scripts/export_kokoro_voice_for_cpp.py \
     models/kokoro/voices/af_heart.pt /tmp/af_heart.kokorovoice
   python3 -c "import struct; d=open('/tmp/af_heart.kokorovoice','rb').read(12); assert d[:4]==b'KVO1'; print('ok', struct.unpack('<II', d[4:12]))"
   ```

   Expect: `ok (510, 256)` (rows/cols may match upstream; second dimension is style size).

2. **End-to-end TTS (needs built `moonshine_tts` target → `moonshine-tts` binary, ONNX, voices, and `data` G2P assets):**

   ```bash
   cmake --build build --target moonshine_tts
   build/moonshine-tts --lang en_us -o /tmp/kokoro_smoke.wav --text "Hello"
   ```

   Expect: success message with non-zero sample count at 24000 Hz.

These checks were executed successfully against this repository (export from `models/kokoro/voices/af_heart.pt`, header `KVO1` + shape `(510, 256)`, and `moonshine-tts` WAV output).
