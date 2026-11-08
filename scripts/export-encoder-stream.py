#!/usr/bin/env python3
"""
Export ``encoder_stream.onnx``: the streaming encoder with per-layer caches.

``encoder.ort`` is run on every update over a window of ``16 * depth`` frames of
left context plus the new frames plus the lookahead, to publish only the new
frames. Every encoder layer attends over a fixed sliding window and the encoder
has no positional encoding, so a layer only needs a short history of its own
input to produce its next rows. This graph keeps that history between calls and
runs each layer only over the frames that changed.

Inputs:

    features  [1, N, encoder_dim]  feature frames not yet seen by the graph
    k_cache   [L, 1, C, heads * head_dim]  keys of the last rows of each layer input
    v_cache   [L, 1, C, heads * head_dim]  values of the same rows
    x_cache   [L, 1, F, encoder_dim]  layer inputs still waiting on lookahead
    frames    [1] int64            feature frames seen before this call
    flush     [1] int64            1 at the end of a line: emit every frame
                                   without waiting for lookahead

Outputs ``encoded`` (the newly final encoder rows, the same rows ``encoder.ort``
publishes) and the three caches for the next call. C is the longest window
(past + future) and F the longest lookahead. Caches are right-aligned; how many
of their rows are valid follows from ``frames``, so a zeroed cache with
``frames = 0`` is the start of a stream. Outputs after ``flush = 1`` are not a
valid state to continue from.

Results match ``encoder.onnx`` up to float rounding. After quantization with
scripts/quantize-streaming-model.sh they are close but not identical to
``encoder.ort``, because dynamic quantization picks one activation scale per
matmul input, and the input is now a few new rows instead of the whole window.

Usage, from a fresh virtual environment (python3 on PATH must be the one with
these packages, because the quantize script runs ``python3 -m ...``):
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==5.22.2 onnx==1.19.1 onnxruntime==1.23.2 \\
        onnx-shrink-ray==0.0.11
    python scripts/export-encoder-stream.py \\
        --model moonshine-ai/moonshine-streaming-small --output-dir small/
    scripts/quantize-streaming-model.sh small/ encoder_stream
    cp small/encoder_stream.ort <streaming model directory>/

Those are the versions this was checked with. The quantized file depends on
the onnx, onnxruntime and onnx-shrink-ray versions: other versions give a
slightly different file whose output rows also differ slightly.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Same opset as the other streaming graphs (see moonshine_voice/lora/export.py).
OPSET = 18


class StreamingEncoder(nn.Module):
    """Runs the encoder layers over new rows only, carrying per-layer caches."""

    def __init__(self, encoder, windows):
        super().__init__()
        self.layers, self.final_norm = encoder.layers, encoder.final_norm
        self.windows = windows
        attn = encoder.layers[0].self_attn
        self.heads = attn.config.num_attention_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.cache_rows = max(past + future for past, future in windows)
        self.pending_rows = max(future for _, future in windows)

    def attend(self, layer, x_query, keys, values, mask):
        """One encoder layer for ``x_query`` against precomputed keys and values.

        Mirrors MoonshineStreamingEncoderLayer and its eager attention, but takes
        the keys and values separately so they can cover more rows than the
        queries do.
        """
        attn = layer.self_attn
        rows = x_query.shape[1]
        query = attn.q_proj(layer.input_layernorm(x_query))
        query = query.view(1, rows, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        values = values.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.matmul(query, keys.transpose(2, 3)) * self.scaling
        weights = torch.softmax(weights + mask, dim=-1, dtype=torch.float32)
        out = torch.matmul(weights, values).transpose(1, 2).reshape(1, rows, -1)
        hidden = x_query + attn.o_proj(out)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    @staticmethod
    def positions(rows: Tensor, start: Tensor) -> Tensor:
        """``start + arange(rows.shape[1])``, kept dynamic in the traced graph."""
        ones = torch.ones_like(rows[0, :, 0], dtype=torch.long)
        return torch.cumsum(ones, 0) - 1 + start

    def forward(self, features: Tensor, k_cache: Tensor, v_cache: Tensor,
                x_cache: Tensor, frames: Tensor, flush: Tensor):
        zero = torch.zeros_like(frames[0])
        seen = frames[0]
        holding = 1 - flush[0].clamp(0, 1)
        x_new = features
        delay = 0
        k_out, v_out, x_out = [], [], []
        for index, (layer, (past, future)) in enumerate(zip(self.layers, self.windows)):
            # Rows are counted from the start of the stream. Before this call the
            # layer had received `end_prev` input rows and produced `done_prev`
            # output rows; it holds keys and values for rows [start, end_prev)
            # and its inputs for rows [done_prev, end_prev).
            end_prev = torch.maximum(seen - delay, zero)
            done_prev = torch.maximum(seen - delay - future, zero)
            start = torch.maximum(done_prev - past, zero)
            end_new = end_prev + torch.ones_like(x_new[0, :, 0], dtype=torch.long).sum()
            done_new = torch.maximum(end_new - future * holding, zero)

            attn = layer.self_attn
            normed = layer.input_layernorm(x_new)
            keys = torch.cat(
                [k_cache[index][:, self.cache_rows - (end_prev - start):],
                 attn.k_proj(normed)], dim=1)
            values = torch.cat(
                [v_cache[index][:, self.cache_rows - (end_prev - start):],
                 attn.v_proj(normed)], dim=1)
            pending = torch.cat(
                [x_cache[index][:, self.pending_rows - (end_prev - done_prev):],
                 x_new], dim=1)
            x_query = pending[:, :done_new - done_prev]

            # `past` and `future` are inclusive, as in the exported encoder.onnx.
            distance = (self.positions(x_query, done_prev).unsqueeze(1)
                        - self.positions(keys, start).unsqueeze(0))
            allowed = (distance >= -future) & (distance <= past)
            mask = torch.zeros_like(distance, dtype=features.dtype)
            mask = mask.masked_fill(~allowed, torch.finfo(features.dtype).min)
            mask = mask.unsqueeze(0).unsqueeze(0)
            x_next = self.attend(layer, x_query, keys, values, mask)

            keep_from = torch.maximum(done_new - past, zero) - start
            k_out.append(self.right_align(keys[:, keep_from:], self.cache_rows))
            v_out.append(self.right_align(values[:, keep_from:], self.cache_rows))
            x_out.append(self.right_align(pending[:, done_new - done_prev:],
                                          self.pending_rows))
            x_new = x_next
            delay += future

        return (self.final_norm(x_new), torch.stack(k_out), torch.stack(v_out),
                torch.stack(x_out))

    @staticmethod
    def right_align(rows: Tensor, size: int) -> Tensor:
        padded = torch.cat([torch.zeros(1, size, rows.shape[2], dtype=rows.dtype),
                            rows], dim=1)
        return padded[:, -size:]


def main():
    parser = argparse.ArgumentParser(
        description="Export a cached streaming encoder (encoder_stream.onnx)")
    parser.add_argument("--model", required=True,
                        help="HF hub id or save_pretrained dir, for example "
                             "moonshine-ai/moonshine-streaming-small")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import onnx
    from transformers import MoonshineStreamingForConditionalGeneration

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32).eval()
    encoder = model.model.encoder
    windows = [tuple(w) for w in model.config.encoder_config.sliding_windows]
    hidden = model.config.encoder_config.hidden_size
    module = StreamingEncoder(encoder, windows).eval()
    depth = len(windows)
    cache = torch.zeros(depth, 1, module.cache_rows, module.heads * module.head_dim)
    pending = torch.zeros(depth, 1, module.pending_rows, hidden)
    example = (torch.randn(1, 7, hidden), cache, cache.clone(), pending,
               torch.tensor([40]), torch.tensor([0]))

    path = out / "encoder_stream.onnx"
    # The TorchScript tracer, as for the frontend in moonshine_voice/lora/export.py:
    # the cache slices depend on the `frames` value, which torch.export rejects
    # as data dependent.
    with torch.no_grad():
        torch.onnx.export(
            module, example, str(path),
            input_names=["features", "k_cache", "v_cache", "x_cache", "frames",
                         "flush"],
            output_names=["encoded", "k_cache_out", "v_cache_out", "x_cache_out"],
            dynamic_axes={"features": {1: "feat_len"}, "encoded": {1: "enc_len"}},
            opset_version=OPSET, do_constant_folding=True, dynamo=False)

    onnx_model = onnx.load(str(path))
    entry = onnx_model.metadata_props.add()
    entry.key = "encoder_stream_config"
    entry.value = json.dumps({
        "encoder_dim": hidden, "attention_dim": module.heads * module.head_dim,
        "windows": [list(w) for w in windows],
        "cache_rows": module.cache_rows, "pending_rows": module.pending_rows})
    onnx.save(onnx_model, str(path), save_as_external_data=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), {depth} layers, "
          f"cache {module.cache_rows} rows, lookahead {module.pending_rows} rows")


if __name__ == "__main__":
    main()
