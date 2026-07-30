#!/bin/bash -ex

# --per-channel is load-bearing for accuracy, not a size/speed tradeoff. Without
# it every weight tensor gets a single scale, which is badly mismatched to the
# frontend: CausalConv1d uses weight_norm, that parametrization survives into the
# exported graph, and the raw direction tensor ends up sharing one scale across
# channels whose magnitudes span 17x. Measured on LibriSpeech test-clean, it is
# worth 7.58% -> 4.83% WER on tiny, 3.03% -> 2.61% on small, and 2.37% -> 2.17%
# on medium, for 0.5% more model size. See experiments.md in moonshine-internal.
PER_CHANNEL_ARGS=(--per-channel)

cd "$1" || exit 1

for ONNX_NAME in frontend encoder adapter cross_kv decoder_kv; do
    if [ "${ONNX_NAME}" == "frontend" ]; then
	    METHOD="integer_weights"
		FILE_SUFFIX="quantized_weights"
	else
		METHOD="integer_activations"
		FILE_SUFFIX="quantized_activations"
	fi
    python3 -m onnx_shrink_ray.shrink \
      --ir-version 10 \
      --method ${METHOD} \
      "${PER_CHANNEL_ARGS[@]}" \
      ${ONNX_NAME}.onnx
    python3 -m onnxruntime.tools.convert_onnx_models_to_ort "${ONNX_NAME}_${FILE_SUFFIX}.onnx"
    mv "${ONNX_NAME}_${FILE_SUFFIX}.ort" "${ONNX_NAME}.ort"
    # python3 -m onnxruntime.tools.convert_onnx_models_to_ort ${ONNX_NAME}.onnx
done
