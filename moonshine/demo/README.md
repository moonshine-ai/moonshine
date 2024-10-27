# Moonshine Demos

This directory contains various scripts to demonstrate the capabilities of the
Moonshine ASR models.

- [Moonshine Demos](#moonshine-demos)
- [Demo: Standalone file transcription with ONNX](#demo-standalone-file-transcription-with-onnx)
- [Demo: Live captioning from microphone input](#demo-live-captioning-from-microphone-input)
  - [Installation.](#installation)
    - [0. Setup environment](#0-setup-environment)
    - [1. Clone the repo and install extra dependencies](#1-clone-the-repo-and-install-extra-dependencies)
  - [Running the demo](#running-the-demo)
  - [Script notes](#script-notes)
    - [Speech truncation and hallucination](#speech-truncation-and-hallucination)
    - [Running on a slower processor](#running-on-a-slower-processor)
    - [Metrics](#metrics)
- [Citation](#citation)


# Demo: Standalone file transcription with ONNX

The script [`onnx_standalone.py`](/moonshine/demo/onnx_standalone.py)
demonstrates how to run a Moonshine model with the `onnxruntime`
package alone, without depending on `torch` or `tensorflow`. This enables
running on SBCs such as Raspberry Pi. Follow the instructions below to setup
and run.

1. Install `onnxruntime` (or `onnxruntime-gpu` if you want to run on GPUs) and `tokenizers` packages using your Python package manager of choice, such as `pip`.

2. Download the `onnx` files from huggingface hub to a directory.

```shell
mkdir moonshine_base_onnx
cd moonshine_base_onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/preprocess.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/encode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/uncached_decode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/cached_decode.onnx
cd ..
```

3. Run `onnx_standalone.py` to transcribe a wav file

```shell
moonshine/moonshine/demo/onnx_standalone.py --models_dir moonshine_base_onnx --wav_file moonshine/moonshine/assets/beckett.wav
['Ever tried ever failed, no matter try again fail again fail better.']
```


# Demo: Live captioning from microphone input

https://github.com/user-attachments/assets/aa65ef54-d4ac-4d31-864f-222b0e6ccbd3

This folder contains a demo of live captioning from microphone input, built on Moonshine. The script runs the Moonshine ONNX model on segments of speech detected in the microphone signal using a voice activity detector called [`silero-vad`](https://github.com/snakers4/silero-vad). The script prints scrolling text or "live captions" assembled from the model predictions to the console.

The following steps have been tested in a `uv` (v0.4.25) virtual environment on these platforms:

- macOS 14.1 on a MacBook Pro M3
- Ubuntu 22.04 VM on a MacBook Pro M2
- Ubuntu 24.04 VM on a MacBook Pro M2

## Installation

### 0. Setup environment

Steps to set up a virtual environment are available in the [top level README](/README.md) of this repo. Note that this demo is standalone and has no requirement to install the `useful-moonshine` package. Instead, you will clone the repo.

### 1. Clone the repo and install extra dependencies

You will need to clone the repo first:

```shell
git clone git@github.com:usefulsensors/moonshine.git
```

Then install the demo's requirements:

```shell
uv pip install -r moonshine/moonshine/demo/requirements.txt
```

There is a dependency on `torch` because of `silero-vad` package.  There is no
dependency on `tensorflow`.

#### Ubuntu: Install PortAudio

Ubuntu needs PortAudio for the `sounddevice` package to run. The latest version (19.6.0-1.2build3 as of writing) is suitable.

```shell
sudo apt update
sudo apt upgrade -y
sudo apt install -y portaudio19-dev
```

## Running the demo

First, check that your microphone is connected and that the volume setting is not muted in your host OS or system audio drivers. Then, run the script:

``` shell
python3 moonshine/moonshine/demo/live_captions.py
```

By default, this will run the demo with the Moonshine Base model using the ONNX runtime. The optional `--model_name` argument sets the model to use: supported arguments are `moonshine/base` and `moonshine/tiny`.

When running, speak in English language to the microphone and observe live captions in the terminal. Quit the demo with `Ctrl+C` to see a full printout of the captions.

An example run on Ubuntu 24.04 VM on MacBook Pro M2 with Moonshine base ONNX
model:

```console
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$ python3 moonshine/moonshine/demo/live_captions.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
Loading Moonshine model 'moonshine/base' (ONNX runtime) ...
Press Ctrl+C to quit live captions.

hine base model being used to generate live captions while someone is speaking. ^C

             model_name :  moonshine/base
       MIN_REFRESH_SECS :  0.2s

      number inferences :  25
    mean inference time :  0.14s
  model realtime factor :  27.82x

Cached captions.
This is an example of the Moonshine base model being used to generate live captions while someone is speaking.
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$
```

For comparison, this is the `faster-whisper` base model on the same instance.
The value of `MIN_REFRESH_SECS` was increased as the model inference is too slow
for a value of 0.2 seconds.  Our Moonshine base model runs ~ 7x faster for this
example.

```console
(env_moonshine_faster_whisper) parallels@ubuntu-linux-2404:~$ python3 moonshine/moonshine/demo/live_captions.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
Loading Faster-Whisper float32 base.en model  ...
Press Ctrl+C to quit live captions.

r float32 base model being used to generate captions while someone is speaking. ^C

             model_name :  base.en
       MIN_REFRESH_SECS :  1.2s

      number inferences :  6
    mean inference time :  1.02s
  model realtime factor :  4.82x

Cached captions.
This is an example of the Faster Whisper float32 base model being used to generate captions while someone is speaking.
(env_moonshine_faster_whisper) parallels@ubuntu-linux-2404:~$
```

## Script notes

You may customize this script to display Moonshine text transcriptions as you wish.

The script `live_captions.py` loads the English language version of Moonshine base ONNX model. It includes logic to detect speech activity and limit the context window of speech fed to the Moonshine model. The returned transcriptions are displayed as scrolling captions. Speech segments with pauses are cached and these cached captions are printed on exit.

### Speech truncation and hallucination

Some hallucinations will be seen when the script is running: one reason is speech gets truncated out of necessity to generate the frequent refresh and timeout transcriptions. Truncated speech contains partial or sliced words for which transcriber model transcriptions are unpredictable. See the printed captions on script exit for the best results.

### Running on a slower processor

If you run this script on a slower processor, consider using the `tiny` model.

```shell
python3 ./moonshine/moonshine/demo/live_captions.py --model_name moonshine/tiny
```

The value of `MIN_REFRESH_SECS` will be ineffective when the model inference time exceeds that value.  Conversely on a faster processor consider reducing the value of `MIN_REFRESH_SECS` for more frequent caption updates.  On a slower processor you might also consider reducing the value of `MAX_SPEECH_SECS` to avoid slower model inferencing encountered with longer speech segments.

### Metrics

The metrics shown on program exit will vary based on the talker's speaking style. If the talker speaks with more frequent pauses, the speech segments are shorter and the mean inference time will be lower. This is a feature of the Moonshine model described in [our paper](https://arxiv.org/abs/2410.15608). When benchmarking, use the same speech, e.g., a recording of someone talking.


# Citation

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
