#!/bin/bash -ex

# --per-channel is load-bearing for accuracy, not a size/speed tradeoff. Without
# it every weight tensor gets a single scale, which is badly mismatched to the
# frontend: CausalConv1d uses weight_norm, that parametrization survives into the
# exported graph, and the raw direction tensor ends up sharing one scale across
# channels whose magnitudes span 17x. Measured on LibriSpeech test-clean, it is
# worth 7.58% -> 4.83% WER on tiny, 3.03% -> 2.61% on small, and 2.37% -> 2.17%
# on medium, for 0.5% more model size. See experiments.md in moonshine-internal.
PER_CHANNEL_ARGS=(--per-channel)

# Usage: quantize-streaming-model.sh <model_dir> [graph ...]
# With no graph names, quantizes every streaming graph. Name graphs (for example
# encoder_stream) to quantize only those, from a directory that holds just them.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$(cd "$1" && pwd)"
shift
cd "${MODEL_DIR}" || exit 1

if [ "$#" -gt 0 ]; then
    ONNX_NAMES=("$@")
else
    ONNX_NAMES=(frontend encoder adapter cross_kv decoder_kv)
    # Optional cached encoder from scripts/export-encoder-stream.py.
    if [ -f encoder_stream.onnx ]; then
        ONNX_NAMES+=(encoder_stream)
    fi
fi

for ONNX_NAME in "${ONNX_NAMES[@]}"; do
    if [ "${ONNX_NAME}" == "frontend" ]; then
	    METHOD="integer_weights"
		FILE_SUFFIX="quantized_weights"
	else
		METHOD="integer_activations"
		FILE_SUFFIX="quantized_activations"
	fi
    INPUT_ONNX="${ONNX_NAME}.onnx"
    if [ "${ONNX_NAME}" == "decoder_kv" ]; then
        # The decoder's output projection reuses the embedding table through
        # a Transpose, which the quantizer will not turn into an int8 MatMul.
        # Left alone it dequantizes the whole 84 MB table on every token; see
        # untie-output-projection.py. Rewrite into a sibling file so the float
        # export stays untouched.
        python3 "${SCRIPT_DIR}/untie-output-projection.py" \
          "${ONNX_NAME}.onnx" -o "${ONNX_NAME}_untied.onnx"
        INPUT_ONNX="${ONNX_NAME}_untied.onnx"
    fi
    python3 -m onnx_shrink_ray.shrink \
      --ir-version 10 \
      --method ${METHOD} \
      "${PER_CHANNEL_ARGS[@]}" \
      "${INPUT_ONNX}"
    if [ "${ONNX_NAME}" == "decoder_kv" ]; then
        mv "${ONNX_NAME}_untied_${FILE_SUFFIX}.onnx" "${ONNX_NAME}_${FILE_SUFFIX}.onnx"
        rm "${ONNX_NAME}_untied.onnx"
    fi
    if [ "${ONNX_NAME}" == "frontend" ]; then
        # integer_weights stores int8 + Cast/Mul/Add. Full ORT optimization
        # folds that chain back to float32 (~4x the file) and the frontend is
        # the wrong graph to pay dequant on every chunk, so split: fused
        # compute in frontend.model.ort, int8 weights dequantized once at load.
        python3 "${SCRIPT_DIR}/split-model-weights.py" \
          --per-channel --force \
          "${ONNX_NAME}_${FILE_SUFFIX}.onnx"
        mv "${ONNX_NAME}_${FILE_SUFFIX}.model.ort" "${ONNX_NAME}.model.ort"
        mv "${ONNX_NAME}_${FILE_SUFFIX}.weights.ort" "${ONNX_NAME}.weights.ort"
    else
        python3 -m onnxruntime.tools.convert_onnx_models_to_ort "${ONNX_NAME}_${FILE_SUFFIX}.onnx"
        mv "${ONNX_NAME}_${FILE_SUFFIX}.ort" "${ONNX_NAME}.ort"
    fi
done

python3 "${SCRIPT_DIR}/check-ort-weight-storage.py" "${MODEL_DIR}"
