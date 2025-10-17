<p align="center">
  <img src="logo.png" width="192px" />
</p>

<h1 style="text-align:center;">Moonshine</h1>

[[Blog]](https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/) [[Paper 1]](https://arxiv.org/abs/2410.15608) [[Paper 2]](https://arxiv.org/abs/2509.02523) [[Model Card]](https://github.com/moonshine-ai/moonshine/blob/main/model-card.md) [[Podcast]](https://notebooklm.google.com/notebook/d787d6c2-7d7b-478c-b7d5-a0be4c74ae19/audio) | [[Join our Discord for questions and support](https://discord.gg/27qp9zSRXF)]

Moonshine is a family of speech-to-text models optimized for fast and accurate automatic speech recognition (ASR) on resource-constrained devices. It is well-suited to real-time, on-device applications like live transcription and voice command recognition. English Moonshine obtains word-error rates (WER) better than similarly-sized Tiny and Base Whisper on the [OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), and non-English Moonshine variants [outperform](#supported-languages) Whisper Small and Medium, which are 9x and 28x larger, respectively.

Moonshine processes audio segments between _5x-15x faster_ than Whisper while maintaining the same (or significantly better!) WER/CER. This is because its compute requirements scale with the length of input audio. Shorter input audio is processed faster, unlike Whisper models that process everything as 30-second chunks.

Unquantized Base is 62M parameters (or 400MB), while Tiny is 27M parameters (around 190MB).

## Supported Languages

Moonshine currently supports 8 languages. Below is a performance summary. Arabic, Chinese, Japanese, and Korean are character-error rates (CER); all others are WER.

| Language    | Tag  | Moonshine Tiny (27M) | Moonshine Base (62M) | Whisper Tiny (39M)  | Whisper Base (74M) | Whisper Small (244M) | Whisper Medium (769M) |
| ----------  | ---- | ---------      | -------        | -------      | -------      | -------       | -------        |
| Arabic      | `ar` | 24.76          |                | 52.40        | 48.25        | 32.44         | 25.44          |
| English     |      | 12.66          | 10.07          | 12.81        | 10.32        |               |                |
| Chinese     | `zh` | 32.77          |                | 68.51        | 59.13        | 46.76         | 40.41          |
| Japanese    | `ja` | 15.69          |                | 96.71        | 72.69        | 40.94         | 27.88          |
| Korean      | `ko` | 9.85           |                | 23.92        | 15.93        | 9.87          | 7.68           |
| Spanish     | `es` |                | TBA            |              |              |               |                |
| Ukrainian   | `uk` | 19.70          |                | 66.77        | 48.56        | 25.93         | 16.51          |
| Vietnamese  | `vi` | 15.92          |                | 96.4         | 52.79        | 26.46         | 18.49          |

<img width="2967" height="929" alt="error_delta_bar" src="https://github.com/user-attachments/assets/ec3c7f61-2d98-495f-9fa7-ed5bccfdc83d" />

Read [the paper](https://arxiv.org/abs/2509.02523) for more details on our non-English flavors of Moonshine.

## Supported Backends

With the release of new Moonshine languages, we have deprecated the Keras-based `moonshine` package. We recommend using Hugging Face `transformers` for vibe-checking the models, and using the ONNX runtime via `moonshine-onnx` for on-device applications. This table summarizes support:

| Model     | Language | `transformers` | ONNX      | Keras (deprecated) |
| -------   | ---      | ----           | ---       | ---                |
| `tiny-ar` | Arabic   | ✅              | ✅        | ❌                 |
| `tiny-zh` | Chinese  | ✅              | ✅        | ❌        |
| `tiny`    | English  | ✅              | ✅        | ✅        |
| `base`    | English  | ✅              | ✅        | ✅        |
| `tiny-ja` | Japanese | ✅              | ✅        | ❌        |
| `tiny-ko` | Korean   | ✅              | ✅        | ❌        |
| `base-es` | Spanish  | ✅              | ✅        | ❌        |
| `tiny-uk` | Ukrainian  | ✅              | ✅      | ❌        |
| `tiny-vi` | Vietnamese | ✅              | ✅      | ❌        |

## Table of Contents

- [Installation](#installation)
  - [1. Create a virtual environment](#1-create-a-virtual-environment)
  - [2. Install `useful-moonshine-onnx`](#2-install-useful-moonshine-onnx)
  - [3. Try it out](#3-try-it-out)
- [Examples](#examples)
  - [Hugging Face Transformers](#huggingface-transformers)
  - [Live Captions](#live-captions)
  - [CTranslate2](#ctranslate2)
  - [Web Applications](#web-applications)
  - [Discord](#discord)
- [License](#license)
- [Citation](#citation)

## Installation

We like `uv` for managing Python environments, so we use it here. If you don't want to use it, simply skip the `uv` installation and leave `uv` off of your shell commands.

### 1. Create a virtual environment

First, [install](https://github.com/astral-sh/uv) `uv` for Python environment management.

Then create and activate a virtual environment:

```shell
uv venv env_moonshine
source env_moonshine/bin/activate
```

### 2. Install `useful-moonshine-onnx`

Using Moonshine with the ONNX runtime is preferable if you want to run the models on SBCs like the Raspberry Pi. To use it, run the following:

```shell
uv pip install useful-moonshine-onnx@git+https://git@github.com/moonshine-ai/moonshine.git#subdirectory=moonshine-onnx
```

### 3. Try it out

You can test Moonshine by transcribing the provided example audio file with the `.transcribe` function:

```shell
python
>>> import moonshine_onnx
>>> moonshine_onnx.transcribe(moonshine_onnx.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny')
['Ever tried ever failed, no matter try again, fail again, fail better.']
```

The first argument is a path to an audio file and the second is the name of a Moonshine model. `moonshine/tiny` and `moonshine/base` are English-only models. If you wish to use one of the non-English Moonshine models, just append the language [IETF tag](https://en.wikipedia.org/wiki/IETF_language_tag) to the model name, e.g., `moonshine/tiny-ko`. See [the table](#supported-languages) for supported languages and their tags.

## Examples

Moonshine models can be used in many applications, so we've included code samples showing how to use them in different situations. The [`demo`](/demo/) folder in this repository also has more information on them.

### Hugging Face Transformers

Moonshine is supported by the `transformers` library, as follows:

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

If you wish to use one of the non-English Moonshine models, just append the [IETF code](https://en.wikipedia.org/wiki/IETF_language_tag) to the repo ID, e.g., `UsefulSensors/moonshine-tiny-ko`. See [the table](#supported-languages) for supported languages and their tags.

### Live Captions

You can try the Moonshine ONNX models with live input from a microphone with the [live captions demo](/demo/README.md#demo-live-captioning-from-microphone-input).

### CTranslate2

The files for the CTranslate2 versions of Moonshine are available at [huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2](https://huggingface.co/UsefulSensors/moonshine/tree/main/ctranslate2), but they require [a pull request to be merged](https://github.com/OpenNMT/CTranslate2/pull/1808) before they can be used with the mainline version of the framework. Until then, you should be able to try them with [our branch](https://github.com/njeffrie/CTranslate2/tree/master), with [this example script](https://github.com/OpenNMT/CTranslate2/pull/1808#issuecomment-2439725339).

### Web Applications

Use our [MoonshineJS](https://github.com/moonshine-ai/moonshine-js) library to run Moonshine models in the web browser with a few lines of Javascript.

### Discord

We have [an active Discord server](https://discord.gg/27qp9zSRXF) where we're happy to answer questions, offer support, and generally geek out about voice AI, so please come [join the conversations](https://discord.gg/27qp9zSRXF). 

## License
All inference code in this repo is released under the MIT license. The English Moonshine models are also released under the MIT license. 

All non-English Moonshine variants are released under the [Moonshine AI Community License](https://www.moonshine.ai/moonshine_community_license.txt) (TLDR: Models are free to use for researchers, developers, small businesses, and creators with less than $1M in annual revenue.). 

A copy of both licenses is included in this repository.

## Citation
If you benefit from our work, please cite our paper:
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

Please also cite our paper on non-English Moonshine variants if you find them useful:
```
@misc{king2025flavorsmoonshinetinyspecialized,
      title={Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices}, 
      author={Evan King and Adam Sabra and Manjunath Kudlur and James Wang and Pete Warden},
      year={2025},
      eprint={2509.02523},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.02523}, 
}
```
