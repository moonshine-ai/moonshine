<p align="center">
  <img src="logo.png" width="192px" />
</p>

<h1 style="text-align:center;">Moonshine</h1>

[[Blog]](https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/) [[Paper]](https://arxiv.org/abs/2410.15608) [[Model Card]](https://github.com/usefulsensors/moonshine/blob/main/model-card.md) [[Podcast]](https://notebooklm.google.com/notebook/d787d6c2-7d7b-478c-b7d5-a0be4c74ae19/audio)

Moonshine is a family of speech-to-text models optimized for fast and accurate automatic speech recognition (ASR) on resource-constrained devices. It is well-suited to real-time, on-device applications like live transcription and voice command recognition. Moonshine obtains word-error rates (WER) better than similarly-sized tiny.en and base.en Whisper models from OpenAI on the datasets used in the [OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) maintained by HuggingFace:

<table>
<tr><th>Tiny</th><th>Base</th></tr>
<tr><td>

| WER        | Moonshine | Whisper |
| ---------- | --------- | ------- |
| Average    | **12.66** | 12.81   |
| AMI        | 22.77     | 24.24   |
| Earnings22 | 21.25     | 19.12   |
| Gigaspeech | 14.41     | 14.08   |
| LS Clean   | 4.52      | 5.66    |
| LS Other   | 11.71     | 15.45   |
| SPGISpeech | 7.70      | 5.93    |
| Tedlium    | 5.64      | 5.97    |
| Voxpopuli  | 13.27     | 12.00   |

</td><td>

| WER        | Moonshine | Whisper |
| ---------- | --------- | ------- |
| Average    | **10.07** | 10.32   |
| AMI        | 17.79     | 21.13   |
| Earnings22 | 17.65     | 15.09   |
| Gigaspeech | 12.19     | 12.83   |
| LS Clean   | 3.23      | 4.25    |
| LS Other   | 8.18      | 10.35   |
| SPGISpeech | 5.46      | 4.26    |
| Tedlium    | 5.22      | 4.87    |
| Voxpopuli  | 10.81     | 9.76    |

</td></tr> </table>

Moonshine's compute requirements scale with the length of input audio. This means that shorter input audio is processed faster, unlike existing Whisper models that process everything as 30-second chunks. To give you an idea of the benefits: Moonshine processes 10-second audio segments _5x faster_ than Whisper while maintaining the same (or better!) WER.

Moonshine Base is approximately 400MB, while Tiny is around 190MB. Both publicly-released models currently support English only.

This repo hosts inference code and demos for Moonshine.

- [Installation](#installation)
  - [1. Create a virtual environment](#1-create-a-virtual-environment)
  - [2a. Install the `useful-moonshine` package to use Moonshine with Torch, TensorFlow, or JAX](#2a-install-the-useful-moonshine-package-to-use-moonshine-with-torch-tensorflow-or-jax)
  - [2b. Install the `useful-moonshine-onnx` package to use Moonshine with ONNX](#2b-install-the-useful-moonshine-onnx-package-to-use-moonshine-with-onnx)
  - [3. Try it out](#3-try-it-out)
- [Examples](#examples)
  - [Live Captions](#live-captions)
  - [Running in the Browser](#running-in-the-browser)
  - [CTranslate2](#ctranslate2)
  - [HuggingFace Transformers](#huggingface-transformers)
- [TODO](#todo)
- [Citation](#citation)

## Installation

We currently offer two options for installing Moonshine:

1. `useful-moonshine`, which uses Keras (with support for Torch, TensorFlow, and JAX backends)
2. `useful-moonshine-onnx`, which uses the ONNX runtime

These instructions apply to both options; follow along to get started.

Note: We like `uv` for managing Python environments, so we use it here. If you don't want to use it, simply skip the `uv` installation and leave `uv` off of your shell commands.

### 1. Create a virtual environment

First, [install](https://github.com/astral-sh/uv) `uv` for Python environment management.

Then create and activate a virtual environment:

```shell
uv venv env_moonshine
source env_moonshine/bin/activate
```

### 2a. Install the `useful-moonshine` package to use Moonshine with Torch, TensorFlow, or JAX

The `useful-moonshine` inference code is written in Keras and can run with each of the backends that Keras supports: Torch, TensorFlow, and JAX. The backend you choose will determine which flavor of the `useful-moonshine` package to install. If you're just getting started, we suggest installing the (default) Torch backend:

```shell
uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
```

To run the provided inference code, you have to instruct Keras to use the PyTorch backend by setting an environment variable:

```shell
export KERAS_BACKEND=torch
```

To run with the TensorFlow backend, run the following to install Moonshine and set the environment variable:

```shell
uv pip install useful-moonshine[tensorflow]@git+https://github.com/usefulsensors/moonshine.git
export KERAS_BACKEND=tensorflow
```

  To run with the JAX backend, run the following:

```shell
uv pip install useful-moonshine[jax]@git+https://github.com/usefulsensors/moonshine.git
export KERAS_BACKEND=jax
# Use useful-moonshine[jax-cuda] for jax on GPU
```

### 2b. Install the `useful-moonshine-onnx` package to use Moonshine with ONNX

Using Moonshine with the ONNX runtime is preferable if you want to run the models on SBCs like the Raspberry Pi. We've prepared a separate version of
the package with minimal dependencies to support these use cases. To use it, run the following:

```shell
uv pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx
```

### 3. Try it out

You can test whichever type of Moonshine you installed by transcribing the provided example audio file with the `.transcribe` function:

```shell
python
>>> import moonshine # or import moonshine_onnx
>>> moonshine.transcribe(moonshine.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny') # or moonshine_onnx.transcribe(...)
['Ever tried ever failed, no matter try again, fail again, fail better.']
```

The first argument is a path to an audio file and the second is the name of a Moonshine model. `moonshine/tiny` and `moonshine/base` are the currently available models.

## Examples

Since the Moonshine models can be used with a variety of different runtimes and applications, we've included code samples showing how to use them in different situations. The [`demo`](/demo/) folder in this repository also has more information on many of them.

### Live Captions

You can try the Moonshine ONNX models with live input from a microphone with the [live captions demo](/demo/README.md#demo-live-captioning-from-microphone-input).

### Running in the Browser

You can try out the Moonshine ONNX models locally in a web browser with our [HuggingFace space](https://huggingface.co/spaces/UsefulSensors/moonshine-web). We've included the [source for this demo](/demo/moonshine-web/) in this repository; this is a great starting place for those wishing to build web-based applications with Moonshine.

### CTranslate2

The files for the CTranslate2 versions of Moonshine are available at [huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2](https://huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2), but they require [a pull request to be merged](https://github.com/OpenNMT/CTranslate2/pull/1808) before they can be used with the mainline version of the framework. Until then, you should be able to try them with [our branch](https://github.com/njeffrie/CTranslate2/tree/master), with [this example script](https://github.com/OpenNMT/CTranslate2/pull/1808#issuecomment-2439725339).

### HuggingFace Transformers

Both models are also available on the HuggingFace hub and can be used with the `transformers` library, as follows:

```python
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_array = ds[0]["audio"]["array"]

inputs = processor(audio_array, return_tensors="pt")

generated_ids = model.generate(**inputs)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

## TODO
* [x] Live transcription demo

* [x] ONNX model

* [x] HF transformers support

* [x] Demo Moonshine running in the browser

* [ ] CTranslate2 support (complete but [awaiting a merge](https://github.com/OpenNMT/CTranslate2/pull/1808))

* [ ] MLX support

* [ ] Fine-tuning code

* [ ] HF transformers.js support

* [ ] Long-form transcription demo 

## Known Issues

### UserWarning: You are using a softmax over axis 3 of a tensor of shape torch.Size([1, 8, 1, 1])
This is a benign warning arising from Keras. For the first token in the decoding loop, the attention score matrix's shape is 1x1, which triggers this warning. You can safely ignore it, or run with `python -W ignore` to suppress the warning.

## Citation
If you benefit from our work, please cite us:
```
@misc{jeffries2024moonshinespeechrecognitionlive,
      title={Moonshine: Speech Recognition for Live Transcription and Voice Commands}, 
      author={Nat Jeffries and Evan King and Manjunath Kudlur and Guy Nicholson and James Wang and Pete Warden},
      year={2024},
      eprint={2410.15608},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.15608}, 
}
```
