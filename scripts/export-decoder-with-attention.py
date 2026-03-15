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

    # Detect model type: non-streaming (decoder_model_merged) or streaming (decoder_kv)
    non_streaming_ort = os.path.join(model_dir, "decoder_model_merged.ort")
    streaming_ort = os.path.join(model_dir, "decoder_kv.ort")

    if os.path.exists(non_streaming_ort):
        input_ort = non_streaming_ort
        output_onnx = os.path.join(model_dir, "decoder_with_attention.onnx")
        output_ort = os.path.join(model_dir, "decoder_with_attention.ort")
        is_streaming = False
    elif os.path.exists(streaming_ort):
        input_ort = streaming_ort
        output_onnx = os.path.join(model_dir, "decoder_kv_with_attention.onnx")
        output_ort = os.path.join(model_dir, "decoder_kv_with_attention.ort")
        is_streaming = True
    else:
        print(f"Error: No decoder model found in {model_dir}", file=sys.stderr)
        print("Expected decoder_model_merged.ort or decoder_kv.ort", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_ort} ({'streaming' if is_streaming else 'non-streaming'})...")

    # Step 1: Load the original .ort and save as ONNX
    tmp_onnx = os.path.join(model_dir, "_tmp_decoder.onnx")
    so = ort.SessionOptions()
    so.optimized_model_filepath = tmp_onnx
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort.InferenceSession(input_ort, so)
    print(f"  Saved as temporary ONNX: {os.path.getsize(tmp_onnx) / 1024 / 1024:.1f} MB")

    # Step 2: Modify the ONNX graph to add attention outputs
    model = onnx.load(tmp_onnx)

    # Find the If node (non-streaming merged decoder uses If for cache branching)
    top_node = None
    for node in model.graph.node:
        if node.op_type == "If":
            top_node = node
            break

    num_layers = args.num_layers

    if top_node is not None:
        # Non-streaming: modify both If branches
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

        print(f"  Detected {num_layers} decoder layers (If-branched graph)")

        for attr in top_node.attribute:
            if attr.type != onnx.AttributeProto.GRAPH:
                continue
            g = attr.g
            if not g.name:
                g.name = attr.name

            for node in g.node:
                if node.op_type != "Softmax":
                    continue
                if not any("encoder_attn" in inp for inp in node.input):
                    continue

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
                g.node.append(
                    helper.make_node(
                        "Identity", [node.output[0]], [out_name],
                        name=f"attn_id_{attr.name}_{layer_idx}",
                    )
                )
                g.output.append(
                    helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
                )

        for i in range(num_layers):
            name = f"cross_attentions.{i}"
            top_node.output.append(name)
            model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
    else:
        # Streaming: flat graph, cross-attention Softmax nodes have mul_* inputs
        # (self-attention Softmax has masked_fill* inputs)
        cross_attn_nodes = [
            n for n in model.graph.node
            if n.op_type == "Softmax" and n.input[0].startswith("mul_")
        ]

        if num_layers is None:
            num_layers = len(cross_attn_nodes)

        if num_layers == 0:
            print("Error: Could not detect cross-attention Softmax nodes.", file=sys.stderr)
            os.remove(tmp_onnx)
            sys.exit(1)

        print(f"  Detected {num_layers} decoder layers (flat graph)")

        for i, node in enumerate(cross_attn_nodes[:num_layers]):
            out_name = f"cross_attentions.{i}"
            model.graph.node.append(
                helper.make_node(
                    "Identity", [node.output[0]], [out_name],
                    name=f"cross_attn_identity_{i}",
                )
            )
            model.graph.output.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
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
