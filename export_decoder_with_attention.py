#!/usr/bin/env python3
"""
Export the Moonshine decoder to ONNX format with cross-attention weights as
additional outputs.

Exports a "teacher-forced alignment" model (no KV cache) that takes:
  - input_ids: [batch, dec_len]          (full token sequence including BOS)
  - encoder_hidden_states: [batch, enc_len, 288]  (encoder output)

And returns:
  - logits: [batch, dec_len, vocab_size]
  - cross_attentions.0 .. cross_attentions.5: [batch, heads, dec_len, enc_len]

This is designed for alignment (DTW-based word timestamps), not generation.
For generation you would use the existing decoder_model_merged.ort.

Usage:
    python export_decoder_with_attention.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wrapper module
# ---------------------------------------------------------------------------

class MoonshineAlignmentDecoder(nn.Module):
    """
    Wraps the HuggingFace MoonshineForConditionalGeneration model to expose
    a simple forward(input_ids, encoder_hidden_states) -> (logits, *cross_attentions)
    interface suitable for ONNX export.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.num_layers = hf_model.config.decoder_num_hidden_layers  # 6

    def forward(self, input_ids, encoder_hidden_states):
        """
        Args:
            input_ids: [batch, dec_len] - decoder token IDs (teacher-forced)
            encoder_hidden_states: [batch, enc_len, hidden_size] - encoder output

        Returns:
            logits: [batch, dec_len, vocab_size]
            cross_attentions_0 .. cross_attentions_5: each [batch, heads, dec_len, enc_len]
        """
        # Call the decoder directly (not via the top-level forward which
        # has encoder_attention_mask routing issues), then project to vocab
        # with proj_out.
        decoder_outputs = self.model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            return_dict=True,
        )

        # Project to vocab
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.model.proj_out(hidden_states)

        # Cross-attention weights: tuple of 6 tensors,
        # each [batch, heads, dec_len, enc_len]
        cross_attentions = decoder_outputs.cross_attentions

        return (logits,) + tuple(cross_attentions)


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main():
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    from transformers import MoonshineForConditionalGeneration

    model_name = "UsefulSensors/moonshine-tiny"
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test-assets", "tiny-en",
    )
    output_path = os.path.join(output_dir, "alignment_model.onnx")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {model_name}")
    hf_model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, attn_implementation="eager"
    )
    hf_model.eval()
    print("Model loaded.")

    num_layers = hf_model.config.decoder_num_hidden_layers
    num_heads = hf_model.config.decoder_num_attention_heads
    hidden_size = hf_model.config.hidden_size
    vocab_size = hf_model.config.vocab_size
    print(f"  decoder layers: {num_layers}")
    print(f"  attention heads: {num_heads}")
    print(f"  hidden size:     {hidden_size}")
    print(f"  vocab size:      {vocab_size}")
    print()

    wrapper = MoonshineAlignmentDecoder(hf_model)
    wrapper.eval()

    # ------------------------------------------------------------------
    # 2. Create dummy inputs for tracing
    # ------------------------------------------------------------------
    batch_size = 1
    dec_len = 5
    enc_len = 40  # typical for ~1s audio

    dummy_input_ids = torch.tensor([[1, 100, 200, 300, 2]], dtype=torch.long)
    dummy_encoder_hidden = torch.randn(batch_size, enc_len, hidden_size)

    # Verify the wrapper works before export
    print("Testing wrapper forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, dummy_encoder_hidden)
    print(f"  logits shape:           {test_out[0].shape}")
    for i in range(num_layers):
        print(f"  cross_attentions.{i} shape: {test_out[1 + i].shape}")
    print()

    # ------------------------------------------------------------------
    # 3. Export to ONNX
    # ------------------------------------------------------------------
    # Build input/output names
    input_names = ["input_ids", "encoder_hidden_states"]

    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"cross_attentions.{i}")

    # Dynamic axes: batch and sequence lengths are variable
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
        "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
        "logits": {0: "batch_size", 1: "decoder_sequence_length"},
    }
    for i in range(num_layers):
        dynamic_axes[f"cross_attentions.{i}"] = {
            0: "batch_size",
            2: "decoder_sequence_length",
            3: "encoder_sequence_length",
        }

    print(f"Exporting ONNX to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_encoder_hidden),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print("Export complete.")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    print()

    # ------------------------------------------------------------------
    # 4. Validate with ONNX Runtime
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Validating with ONNX Runtime...")
    print("=" * 60)

    import onnxruntime as ort

    session = ort.InferenceSession(output_path)

    print()
    print("--- INPUTS ---")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

    print()
    print("--- OUTPUTS ---")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    # Run inference with test data
    test_dec_len = 7
    test_enc_len = 50
    test_input_ids = np.array([[1, 42, 567, 1234, 89, 10, 2]], dtype=np.int64)
    test_encoder_hidden = np.random.randn(1, test_enc_len, hidden_size).astype(np.float32)

    print()
    print(f"Running inference (dec_len={test_dec_len}, enc_len={test_enc_len})...")

    ort_outputs = session.run(
        None,
        {
            "input_ids": test_input_ids,
            "encoder_hidden_states": test_encoder_hidden,
        },
    )

    print(f"  logits shape:           {ort_outputs[0].shape}")
    for i in range(num_layers):
        ca = ort_outputs[1 + i]
        print(f"  cross_attentions.{i} shape: {ca.shape}")

    # Verify shapes
    assert ort_outputs[0].shape == (1, test_dec_len, vocab_size), \
        f"logits shape mismatch: {ort_outputs[0].shape}"
    for i in range(num_layers):
        expected = (1, num_heads, test_dec_len, test_enc_len)
        actual = ort_outputs[1 + i].shape
        assert actual == expected, \
            f"cross_attentions.{i} shape mismatch: {actual} != {expected}"

    # Verify attention weights are proper distributions (softmax output)
    for i in range(num_layers):
        ca = ort_outputs[1 + i]
        row_sums = ca.sum(axis=-1)  # sum over enc_len
        # Each row should sum to ~1.0 (softmax output)
        assert np.allclose(row_sums, 1.0, atol=1e-4), \
            f"cross_attentions.{i} rows don't sum to 1: min={row_sums.min():.4f}, max={row_sums.max():.4f}"

    print()
    print("All validations passed!")

    # ------------------------------------------------------------------
    # 5. Compare with PyTorch outputs
    # ------------------------------------------------------------------
    print()
    print("Comparing ONNX vs PyTorch outputs...")

    with torch.no_grad():
        pt_out = wrapper(
            torch.tensor(test_input_ids, dtype=torch.long),
            torch.tensor(test_encoder_hidden, dtype=torch.float32),
        )

    pt_logits = pt_out[0].numpy()
    logits_diff = np.abs(ort_outputs[0] - pt_logits).max()
    print(f"  logits max abs diff:           {logits_diff:.6f}")

    for i in range(num_layers):
        pt_ca = pt_out[1 + i].numpy()
        ort_ca = ort_outputs[1 + i]
        ca_diff = np.abs(ort_ca - pt_ca).max()
        print(f"  cross_attentions.{i} max abs diff: {ca_diff:.6f}")

    print()
    print(f"ONNX model saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
