#!/usr/bin/env python3
"""
Export a decoder_with_attention model for word-level timestamps.

Takes the original decoder_model_merged.ort, adds cross-attention weight
outputs by wiring the internal encoder_attn Softmax nodes to new graph
outputs, and saves the result as decoder_with_attention.ort.

The modified model is identical to the original — same weights, same
computation, same speed — just with 6 additional outputs (one per
decoder layer) that surface the attention weights already being computed
internally.

Usage:
    python scripts/export-decoder-with-attention.py <model_dir>

    model_dir should contain decoder_model_merged.ort (and optionally
    encoder_model.ort and tokenizer.bin). The script will create
    decoder_with_attention.ort in the same directory.

Example:
    python scripts/export-decoder-with-attention.py test-assets/tiny-en

Requirements:
    pip install onnx onnxruntime
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Add cross-attention outputs to a Moonshine decoder model"
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing decoder_model_merged.ort",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of decoder layers (auto-detected if not specified)",
    )
    args = parser.parse_args()

    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    model_dir = args.model_dir
    input_ort = os.path.join(model_dir, "decoder_model_merged.ort")
    output_onnx = os.path.join(model_dir, "decoder_with_attention.onnx")
    output_ort = os.path.join(model_dir, "decoder_with_attention.ort")

    if not os.path.exists(input_ort):
        print(f"Error: {input_ort} not found", file=sys.stderr)
        sys.exit(1)

    # Step 1: Load the original .ort and save as ONNX
    # (ORT can load .ort files; onnx library needs standard ONNX format)
    print(f"Loading {input_ort}...")
    tmp_onnx = os.path.join(model_dir, "_tmp_decoder.onnx")
    so = ort.SessionOptions()
    so.optimized_model_filepath = tmp_onnx
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort.InferenceSession(input_ort, so)
    print(f"  Saved as temporary ONNX: {os.path.getsize(tmp_onnx) / 1024 / 1024:.1f} MB")

    # Step 2: Modify the ONNX graph to add attention outputs
    model = onnx.load(tmp_onnx)

    # Find the If node (optimum's merged decoder uses an If for cache branching)
    top_node = None
    for node in model.graph.node:
        if node.op_type == "If":
            top_node = node
            break

    if top_node is None:
        print("Error: No If node found in decoder graph.", file=sys.stderr)
        print("This script expects a merged decoder exported by optimum.", file=sys.stderr)
        os.remove(tmp_onnx)
        sys.exit(1)

    # Auto-detect number of layers from encoder_attn Softmax nodes
    num_layers = args.num_layers
    if num_layers is None:
        for attr in top_node.attribute:
            if attr.type != onnx.AttributeProto.GRAPH:
                continue
            count = sum(
                1
                for node in attr.g.node
                if node.op_type == "Softmax"
                and any("encoder_attn" in inp for inp in node.input)
            )
            if count > 0:
                num_layers = count
                break

    if num_layers is None or num_layers == 0:
        print("Error: Could not detect encoder_attn Softmax nodes.", file=sys.stderr)
        os.remove(tmp_onnx)
        sys.exit(1)

    print(f"  Detected {num_layers} decoder layers")

    # Add attention outputs to both If branches (then_branch and else_branch)
    for attr in top_node.attribute:
        if attr.type != onnx.AttributeProto.GRAPH:
            continue
        g = attr.g

        # Ensure subgraph has a name (required by ORT validator)
        if not g.name:
            g.name = attr.name

        # Find Softmax nodes for encoder_attn in this subgraph
        for node in g.node:
            if node.op_type != "Softmax":
                continue
            if not any("encoder_attn" in inp for inp in node.input):
                continue

            # Extract layer index from the node name
            # e.g., "/model/decoder/layers.3/encoder_attn/Softmax"
            layer_idx = None
            for part in node.name.split("/"):
                if part.startswith("layers."):
                    try:
                        layer_idx = int(part.split(".")[1])
                    except (IndexError, ValueError):
                        pass

            if layer_idx is None:
                continue

            out_name = f"cross_attentions.{layer_idx}"

            # Add Identity node: softmax output → named output
            g.node.append(
                helper.make_node(
                    "Identity",
                    [node.output[0]],
                    [out_name],
                    name=f"attn_id_{attr.name}_{layer_idx}",
                )
            )
            g.output.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            )

    # Add cross_attentions to the If node's outputs and the top-level graph outputs
    for i in range(num_layers):
        name = f"cross_attentions.{i}"
        top_node.output.append(name)
        model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )

    # Save modified ONNX
    onnx.save(model, output_onnx)
    print(f"  Modified ONNX: {os.path.getsize(output_onnx) / 1024 / 1024:.1f} MB")

    # Step 3: Convert to ORT format
    so2 = ort.SessionOptions()
    so2.optimized_model_filepath = output_ort
    so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(output_onnx, so2)
    print(f"  ORT output: {os.path.getsize(output_ort) / 1024 / 1024:.1f} MB")

    # Clean up temp file
    os.remove(tmp_onnx)

    # Step 4: Verify
    attn_outputs = [o.name for o in sess.get_outputs() if "cross_attentions" in o.name]
    print(f"\nVerification:")
    print(f"  Total outputs: {len(sess.get_outputs())}")
    print(f"  Cross-attention outputs: {len(attn_outputs)}")
    for name in attn_outputs:
        print(f"    {name}")

    orig_size = os.path.getsize(input_ort) / 1024 / 1024
    new_size = os.path.getsize(output_ort) / 1024 / 1024
    print(f"\n  Original decoder: {orig_size:.1f} MB")
    print(f"  Modified decoder: {new_size:.1f} MB")
    print(f"  Size difference: {new_size - orig_size:+.1f} MB")

    print(f"\nDone. Output: {output_ort}")


if __name__ == "__main__":
    main()
