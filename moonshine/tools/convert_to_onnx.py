import sys
import keras
import moonshine
from pathlib import Path


def convert_and_store(model, input_signature, output_file):
    from tf2onnx.convert import from_keras
    import onnx

    onnx_model, external_storage_dict = from_keras(
        model, input_signature=input_signature
    )
    assert external_storage_dict is None, f"External storage for onnx not supported"
    onnx.save_model(onnx_model, output_file)


def main():
    assert len(sys.argv) == 3, (
        "Usage: convert_to_onnx.py <moonshine model name> <output directory name>"
    )
    assert keras.config.backend() == "tensorflow", (
        "Should be run with the tensorflow backend"
    )

    import tensorflow as tf

    model_name = sys.argv[1]
    model = moonshine.load_model(model_name)
    output_dir = sys.argv[2]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    convert_and_store(
        model.preprocessor.preprocess,
        input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)],
        output_file=f"{output_dir}/preprocess.onnx",
    )

    seq_len_spec = tf.TensorSpec([1], dtype=tf.int32)

    convert_and_store(
        model.encoder.encoder,
        input_signature=[
            tf.TensorSpec([None, None, model.dim], dtype=tf.float32),
            seq_len_spec,
        ],
        output_file=f"{output_dir}/encode.onnx",
    )

    input_spec = tf.TensorSpec([None, None], dtype=tf.int32)
    context_spec = tf.TensorSpec([None, None, model.dim], dtype=tf.float32)
    cache_spec = [
        tf.TensorSpec(
            [None, None, model.n_head, model.inner_dim // model.n_head],
            dtype=tf.float32,
        )
        for _ in range(model.dec_n_layers * 4)
    ]

    convert_and_store(
        model.decoder.uncached_call,
        input_signature=[input_spec, context_spec, seq_len_spec],
        output_file=f"{output_dir}/uncached_decode.onnx",
    )

    convert_and_store(
        model.decoder.cached_call,
        input_signature=[input_spec, context_spec, seq_len_spec] + cache_spec,
        output_file=f"{output_dir}/cached_decode.onnx",
    )


if __name__ == "__main__":
    main()
