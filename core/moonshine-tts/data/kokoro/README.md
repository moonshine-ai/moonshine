# Kokoro — bundled TTS for `moonshine_tts`

This directory is a **Kokoro ONNX bundle** layout (`kokoro/` under your asset root or current working directory). It must contain at least:

| Path | Role |
|------|------|
| `config.json` | Model config (includes phoneme `vocab` for ONNX). |
| `prosody.model.ort` + `prosody.weights.ort` | Kokoro-82M's first half, as a split ORT pair (see [What ships](#what-ships)): phonemes to frame-rate features, run once per utterance. |
| `decoder.model.ort` + `decoder.weights.ort` | Its second half: a range of those frames to audio, run once per streamed chunk, or over every frame at once for a whole utterance. |
| `voices/*.kokorovoice` | Style tensors for ONNX inference (C++ cannot load Hugging Face `voices/*.pt` pickles). |

The two stages are what let streaming cut below a sentence, so the first audio of a long sentence arrives before all of it has been decoded. Run back to back they reproduce the whole-utterance graph sample for sample, which is why that graph is no longer published: carrying it as well would double the download for nothing. Generate the stages with `python scripts/split-kokoro-stages.py`.

A whole-utterance `model.model.ort` + `model.weights.ort` pair, or a single `model.ort`, is still loaded when present and used in preference to the stages. That is what a caller pointing `kokoro_model` at their own export gets, and what an install predating the stages keeps working from.

## What ships

The bundled model is the upstream **`model_uint8`** build with its remaining
float32 weights packed to int8 and then split into a graph/weights pair:

1. [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
   `onnx/model_uint8.onnx` (177 MB). Mixed precision: the convolutions stay
   float, which is what keeps it intelligible.
2. `onnx-shrink-ray` `integer_weights`, **per-channel**, over the 72% of that
   file that is still float32 → 83 MB `.onnx`.
3. `python scripts/convert-models-to-ort.py --force-split core/moonshine-tts/data/kokoro`
   → `model.model.ort` + `model.weights.ort`, 83 MB together. Splitting is what
   makes the size win survive conversion to ORT format, which otherwise folds
   the dequantize chain back into float32 weights. See
   `core/moonshine-tts/src/split-weights.h`. `--force-split` is required
   because the automatic test reads Kokoro's weight-byte mix and predicts a
   loss; measurement says otherwise, and the flag's help text says why.

Against the dynamically quantized 93 MB `model.ort` this replaces, on a
Raspberry Pi 4 at 4 threads: 4.0x faster on a 7-token sentence, 2.1x on a
130-token passage, quicker to load, the same 2.2% word error rate, and mel
distance against the float32 reference of 2.77/2.69 rather than 10.99/5.80.
It costs about 85 MB more peak resident memory (506 MB against 422 MB),
because the weights are dequantized to float32 once at load and stay that way.

Per-channel scaling is what buys that fidelity, and it is not optional:
per-tensor measures the same 83 MB at the same speed but is 3.4x worse on mel
distance (0.3622 against 0.1055 measured on the float32 base). Kokoro's
iSTFTNet generator uses weight normalization throughout, which spreads channel
magnitudes by construction — the same condition that made per-channel
load-bearing for the STT frontend.

> **`onnx-shrink-ray` needs a fix first.** Its per-channel path divides by a
> channel's value range, and Kokoro's PL-BERT word embeddings contain a
> channel whose range is zero. That yields a NaN zero point, and one NaN
> weight makes every output of the model NaN — audibly silence, and 36% word
> error if you measure it. Substituting a range of 1 leaves such a channel
> exactly reconstructible. Not yet fixed upstream in
> [usefulsensors/onnx_shrink_ray](https://github.com/usefulsensors/onnx_shrink_ray).

Optional in a **build tree** (not required under `data` if you only ship C++): `kokoro-v1_0.pth` (PyTorch weights, for re-exporting ONNX), `voices/*.pt` (source for `.kokorovoice`), `onnx_export_meta.json` (written by the download/export script).

## Provenance

| Asset | Source |
|--------|--------|
| **Weights, config, native voices (`voices/*.pt`)** | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Hugging Face (see upstream **VOICES.md** for voice IDs and locales). |
| **`model.onnx`** | Either: **(A)** `scripts/fetch_hf_kokoro_quantized_onnx.py` — downloads [onnx-community/Kokoro-82M-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-ONNX) `onnx/model_quantized.onnx` (~92 MiB) and renames input `style` → `ref_s` for `MoonshineTTS`; or **(B)** `scripts/download_kokoro_onnx.py` — local FP32 export via **`kokoro`** (`KModel`, `KModelForONNX`) with **`disable_complex=True`**. |
| **`*.kokorovoice`** | Produced by `scripts/export_kokoro_voice_for_cpp.py` from each `voices/*.pt` tensor pack. Format: magic `KVO1`, little-endian `uint32` rows/cols, row-major `float32` data (after squeezing singleton dims to shape `[N, 256]`). |

`MoonshineTTS` passes **`speed`** as float32 `[1]` or double scalar depending on the graph (detected at load time; ONNX Runtime requires `GetInputTypeInfo(i).GetONNXType() == ONNX_TYPE_TENSOR` before reading the element type).

Python TTS in the parent monorepo (`speak.py`) can use the same HF bundle or PyTorch weights; the C++ path is **ONNX + `.kokorovoice` only**.

### Judging a quantized Kokoro build

The same **onnx-shrink-ray** `quantize_weights` path used for Arabic BERT (int8 weight storage + dequant,
`float_quantization=False`) is what the shipped build uses, but judging it needs care.

Correlation and mel distance against the FP32 output rank these builds
differently from how a listener does, and the disagreement is not small.
Restricting static quantization to convolutions improves mel distance from
0.841 to 0.725 and drives duration drift to zero, while taking word error rate
from 18.3% to 24.0%. Round-trip the synthesized audio through `Transcriber`
and count words before believing a spectral metric.

Measure on the runtime you ship, too. Under the ONNX Runtime this library
bundles, the dynamically quantized build this replaced scores mel 10.99/5.80
on a Pi, while a newer pip `onnxruntime` on a Mac makes the same file look
close to the reference. Quantized operators are exactly where releases differ,
so a Mac reading can rank two builds the opposite way from the target.

For experiments you can run:

```bash
python scripts/download_kokoro_onnx.py --out-dir data/kokoro --only-shrink
# or after export:  --shrink-weights
```

Expect a smaller file (~80 MiB vs ~310 MiB) but **validate audio** before shipping; ORT dynamic MatMul/Gemm
quantization (`--experimental-int8`) is a separate trap: it puts a convolution-heavy graph on
`ConvInteger`, which has no optimized kernel, so the result is slower than FP32 on Arm as well as
less accurate.

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
   cp -a models/kokoro/config.json models/kokoro/model.model.ort \
     models/kokoro/model.weights.ort data/kokoro/
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
