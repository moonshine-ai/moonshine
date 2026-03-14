# Word Timestamps: Overhead Analysis & Fix

## Current Results (Two-Pass — WRONG approach)

The current C++ implementation uses a **separate alignment ONNX model** run after transcription. This is the two-pass approach we already proved is inferior:

| Audio | Duration | Without | With | Overhead | % |
|-------|----------|---------|------|----------|---|
| beckett.wav | 10s | 145ms | 184ms | +39ms | +27% |
| two_cities_16k.wav | 44s | 800ms | 990ms | +190ms | +24% |

This ~25% overhead is unacceptable. It defeats the purpose of Moonshine's low-latency design.

## Why This Happened

The ONNX `.ort` decoder models don't expose cross-attention weights as outputs. So instead of collecting attention during the decode loop (single-pass), we exported a separate "alignment model" and run it after transcription — effectively running the decoder twice.

## What We Proved in Python

The Python prototype (`word_timestamps_v2.py`) demonstrated the correct single-pass approach:

| Audio | Duration | Transcribe Only | Single-Pass (v2) | Overhead | % |
|-------|----------|-----------------|-------------------|----------|---|
| beckett.wav | 10s | 181ms | 185ms | +4ms | +2% |
| Genesis 1:2 | 13s | 353ms | 362ms | +9ms | +3% |

**+2-3% overhead** because the attention weights are already computed during decoding — we just copy them out.

## The Fix

Re-export the decoder ONNX model to include cross-attention weights as additional outputs. Then the C++ decode loop collects attention at each step (like the Python v2 does), and no second model pass is needed.

### Why this can't be done externally

The HuggingFace Moonshine decoder uses `DynamicCache` and `EncoderDecoderCache` classes with dynamic Python control flow that `torch.onnx.export` cannot trace. The original `.ort` models were exported using Moonshine's internal `optimum`-based export pipeline, which handles the KV cache merging with ONNX `If` operators.

To export a decoder with attention outputs, the Moonshine team would need to modify their export pipeline to:
1. Pass `output_attentions=True` to the decoder during export tracing
2. Add the cross-attention weight tensors as additional ONNX graph outputs
3. Re-export the `decoder_model_merged.ort` with these additional outputs

This is a one-line change in their export code, but it requires access to their internal tooling.

### What we can do now

The current implementation uses a **teacher-forced alignment model** (`alignment_model.ort`) that runs after transcription. This adds ~25% overhead because it's a full decoder pass.

### What 3% would look like

If the main decoder outputs attention weights at each step:
1. During the autoregressive decode loop in `MoonshineModel::transcribe()`, read the additional attention output tensors
2. After decoding completes, run DTW on the accumulated attention matrix
3. Delete the separate `alignment_model.ort` — no longer needed
4. The only overhead is copying out the attention tensors (already computed by the decoder) — ~0.2ms per step

Expected result: **+2-3% overhead**, matching the Python v2 benchmark.

### Recommendation for the Moonshine team

Add `cross_attentions` as optional ONNX outputs to the decoder export, gated behind a model variant or build flag. The decoder already computes these weights internally — they just need to be surfaced.
