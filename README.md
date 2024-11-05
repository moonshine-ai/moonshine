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
  - [2. Install the Moonshine package](#2-install-the-moonshine-package)
  - [3. Try it out](#3-try-it-out)
- [Examples](#examples)
  - [Onnx standalone](#onnx-standalone)
  - [Live Captions](#live-captions)
  - [CTranslate2](#ctranslate2)
  - [HuggingFace Transformers](#huggingface-transformers)
- [TODO](#todo)
- [Citation](#citation)

## Installation

We like `uv` for managing Python environments, so we use it here. If you don't want to use it, simply skip the first step and leave `uv` off of your shell commands.

### 1. Create a virtual environment

First, [install](https://github.com/astral-sh/uv) `uv` for Python environment management.

Then create and activate a virtual environment:

```shell
uv venv env_moonshine
source env_moonshine/bin/activate
```

### 2. Install the Moonshine package

The `moonshine` inference code is written in Keras and can run with each of the backends that Keras supports: Torch, TensorFlow, and JAX. The backend you choose will determine which flavor of the `moonshine` package to install. If you're just getting started, we suggest installing the (default) Torch backend:

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

To run with ONNX runtime that is supported on platforms, run the following:

```shell
uv pip install useful-moonshine[onnx]@git+https://github.com/usefulsensors/moonshine.git
```

### 3. Try it out

You can test Moonshine by transcribing the provided example audio file with the `.transcribe` function:

```shell
python
>>> import moonshine
>>> moonshine.transcribe(moonshine.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny')
['Ever tried ever failed, no matter try again, fail again, fail better.']
```

The first argument is a path to an audio file and the second is the name of a Moonshine model. `moonshine/tiny` and `moonshine/base` are the currently available models.
Use the `moonshine.transcribe_with_onnx` function to use the ONNX runtime for inference. The parameters are the same as they are for `moonshine.transcribe`.

## Examples

The Moonshine models can be used with a variety of different runtimes and applications, so we've included code samples showing how to use them in different situations. The [`moonshine/demo`](/moonshine/demo/) folder in this repository also has more information on many of them.

### Onnx standalone

The latest versions of the Onnx Moonshine models are available on HuggingFace at [huggingface.co/UsefulSensors/moonshine/tree/main/onnx](https://huggingface.co/UsefulSensors/moonshine/tree/main/onnx). You can find [an example Python script](/moonshine/demo/onnx_standalone.py) and more information about running them [in the demo folder](/moonshine/demo/README.md#demo-standalone-file-transcription-with-onnx).

### Live Captions

You can try the Moonshine models with live input from a microphone on many platforms with the [live captions demo](/moonshine/demo/README.md#demo-live-captioning-from-microphone-input).

### CTranslate2

The files for the CTranslate2 versions of Moonshine are available at [huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2](https://huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2), but they require [a pull request to be merged](https://github.com/OpenNMT/CTranslate2/pull/1808) before they can be used with the mainline version of the framework. Until then, you should be able to try them with [our branch](https://github.com/njeffrie/CTranslate2/tree/master), with [this example script](https://github.com/OpenNMT/CTranslate2/pull/1808#issuecomment-2439725339).

### HuggingFace Transformers

Both models are also available on the HuggingFace hub and can be used with the `transformers` library, as follows:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoConfig, PreTrainedTokenizerFast

import torchaudio
import sys

audio, sr = torchaudio.load(sys.argv[1])
if sr != 16000:
  audio = torchaudio.functional.resample(audio, sr, 16000)

# 'usefulsensors/moonshine-base' for the base model
model = AutoModelForSpeechSeq2Seq.from_pretrained('usefulsensors/moonshine-tiny', trust_remote_code=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained('usefulsensors/moonshine-tiny')

tokens = model(audio)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

## TODO
* [x] Live transcription demo

* [x] ONNX model

* [x] HF transformers support

* [ ] CTranslate2 support (complete but [awaiting a merge](https://github.com/OpenNMT/CTranslate2/pull/1808))

* [ ] MLX support

* [ ] Fine-tuning code

* [ ] HF transformers.js support

* [ ] Long-form transcription demo 

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
