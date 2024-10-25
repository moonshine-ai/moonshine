# Moonshine Demos

This directory contains various scripts the demonstrate the capabilities of the Moonshine ASR models.

## onnx_standalone.py

This script demonstrates how to run a Moonshine model with the `onnxruntime` package alone, without depending on `torch` or `tensorflow`. This enables running on SBCs such as Raspberry Pi. Follow the instructions below to setup and run.

* Install `onnxruntime` (or `onnxruntime-gpu` if you want to run on GPUs) and `tokenizers` packages using your Python package manager of choice, such as `pip`.

* Download the `onnx` files from huggingface hub to a directory.

  ```shell
  mkdir moonshine_base_onnx
  cd moonshine_base_onnx
  wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/preprocess.onnx
  wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/encode.onnx
  wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/uncached_decode.onnx
  wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/cached_decode.onnx
  cd ..
  ```

* Run `onnx_standalone.py` to transcribe a wav file

  ```shell
  moonshine/moonshine/demo/onnx_standalone.py --models_dir moonshine_base_onnx --wav_file moonshine/moonshine/assets/beckett.wav
  ['Ever tried ever failed, no matter try again fail again fail better.']
  ```
