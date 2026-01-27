#!/bin/bash -ex

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
      ${ONNX_NAME}.onnx
    python3 -m onnxruntime.tools.convert_onnx_models_to_ort "${ONNX_NAME}_${FILE_SUFFIX}.onnx"
    mv "${ONNX_NAME}_${FILE_SUFFIX}.ort" "${ONNX_NAME}.ort"
    # python3 -m onnxruntime.tools.convert_onnx_models_to_ort ${ONNX_NAME}.onnx
done
