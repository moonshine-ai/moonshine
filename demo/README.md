# Moonshine Demos

This directory contains scripts to demonstrate the capabilities of the
Moonshine ASR models.

-   [Moonshine Demos](#moonshine-demos)
-   [Demo: Running in the browser](#demo-running-in-the-browser)
    -   [Installation](#installation)
-   [Demo: Live captioning from microphone input](#demo-live-captioning-from-microphone-input)
    -   [Installation](#installation-1)
        -   [0. Setup environment](#0-setup-environment)
        -   [1. Clone the repo and install extra dependencies](#1-clone-the-repo-and-install-extra-dependencies)
            -   [Ubuntu: Install PortAudio](#ubuntu-install-portaudio)
    -   [Running the demo](#running-the-demo)
    -   [Script notes](#script-notes)
        -   [Speech truncation and hallucination](#speech-truncation-and-hallucination)
        -   [Running on a slower processor](#running-on-a-slower-processor)
        -   [Metrics](#metrics)
-   [Demo: Live captioning a WebRTC stream with FastRTC](#demo-live-captioning-a-webrtc-stream-with-fastrtc)
-   [Citation](#citation)

# Demo: Running in the browser

The Node.js project in [`moonshine-web`](/demo/moonshine-web/) demonstrates how to run the
Moonshine models in the web browser using `onnxruntime-web`. You can try this demo on your own device using our [HuggingFace space](https://huggingface.co/spaces/UsefulSensors/moonshine-web) without having to run the project from the source here. Of note, the [`moonshine.js`](/demo/moonshine-web/src/moonshine.js) script contains everything you need to perform inferences with the Moonshine ONNX models in the browser. If you would like to build on the web demo, follow these instructions to get started.

## Installation

You must have Node.js (or another JavaScript toolkit like [Bun](https://bun.sh/)) installed to get started. Install [Node.js](https://nodejs.org/en) if you don't have it already.

Once you have your JavaScript toolkit installed, clone the `moonshine` repo and navigate to this directory:

```shell
git clone git@github.com:moonshine-ai/moonshine.git
cd moonshine/demo/moonshine-web
```

Then install the project's dependencies:

```shell
npm install
```

The demo expects the Moonshine Tiny and Base ONNX models to be available in `public/moonshine/tiny` and `public/moonshine/base`, respectively. To preserve space, they are not included here. However, we've included a helper script that you can run to conveniently download them from HuggingFace:

```shell
npm run get-models
```

This project uses Vite for bundling and development. Run the following to start a development server and open the demo in your web browser:

```shell
npm run dev
```

# Demo: Live captioning from microphone input

https://github.com/user-attachments/assets/aa65ef54-d4ac-4d31-864f-222b0e6ccbd3

The [`moonshine-onnx/live_captions.py`](/demo/moonshine-onnx/live_captions.py) script contains a demo of live captioning from microphone input, built on Moonshine. The script runs the Moonshine ONNX model on segments of speech detected in the microphone signal using a voice activity detector called [`silero-vad`](https://github.com/snakers4/silero-vad). The script prints scrolling text or "live captions" assembled from the model predictions to the console.

The following steps have been tested in `uv` virtual environments on these platforms:

-   macOS 14.1 on a MacBook Pro M3
-   Ubuntu 22.04 VM on a MacBook Pro M2
-   Ubuntu 24.04 VM on a MacBook Pro M2
-   Debian 12.8 (64-bit) on a Raspberry Pi 5 (Model B Rev 1.0)

## Installation

### 0. Setup environment

Steps to set up a virtual environment are available in the [top level README](/README.md) of this repo. After creating a virtual environment, do the following:

### 1. Clone the repo and install extra dependencies

You will need to clone the repo first:

```shell
git clone git@github.com:moonshine-ai/moonshine.git
```

Then install the demo's requirements including mitigation for a failure to build
and install `llvmlite` without `numba` package:

```shell
uv pip install numba

uv pip install -r moonshine/demo/moonshine-onnx/requirements.txt
```

Note that while `useful-moonshine-onnx` has no requirement for `torch`, this demo introduces a dependency for it because of the `silero-vad` package.

#### Ubuntu: Install PortAudio

Ubuntu needs PortAudio for the `sounddevice` package to run. The latest version (19.6.0-1.2build3 as of writing) is suitable.

```shell
sudo apt update
sudo apt upgrade -y
sudo apt install -y portaudio19-dev
```

## Running the demo

First, check that your microphone is connected and that the volume setting is not muted in your host OS or system audio drivers. Then, run the script:

```shell
python3 moonshine/demo/moonshine-onnx/live_captions.py
```

By default, this will run the demo with the Moonshine Base model using the ONNX runtime. The optional `--model_name` argument sets the model to use: supported arguments are `moonshine/base` and `moonshine/tiny`.

When running, speak in English language to the microphone and observe live captions in the terminal. Quit the demo with `Ctrl+C` to see a full printout of the captions.

An example run on Ubuntu 24.04 VM on MacBook Pro M2 with Moonshine base ONNX
model:

```console
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$ python3 moonshine/demo/moonshine-onnx/live_captions.py
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
for a value of 0.2 seconds. Our Moonshine base model runs ~ 7x faster for this
example.

```console
(env_moonshine_faster_whisper) parallels@ubuntu-linux-2404:~$ python3 moonshine/demo/moonshine-onnx/live_captions.py
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

The script `moonshine-onnx/live_captions.py` loads the English language version of Moonshine base ONNX model. It includes logic to detect speech activity and limit the context window of speech fed to the Moonshine model. The returned transcriptions are displayed as scrolling captions. Speech segments with pauses are cached and these cached captions are printed on exit.

### Speech truncation and hallucination

Some hallucinations will be seen when the script is running: one reason is speech gets truncated out of necessity to generate the frequent refresh and timeout transcriptions. Truncated speech contains partial or sliced words for which transcriber model transcriptions are unpredictable. See the printed captions on script exit for the best results.

### Running on a slower processor

If you run this script on a slower processor, consider using the `tiny` model.

```shell
python3 ./moonshine/demo/moonshine-onnx/live_captions.py --model_name moonshine/tiny
```

The value of `MIN_REFRESH_SECS` will be ineffective when the model inference time exceeds that value. Conversely on a faster processor consider reducing the value of `MIN_REFRESH_SECS` for more frequent caption updates. On a slower processor you might also consider reducing the value of `MAX_SPEECH_SECS` to avoid slower model inferencing encountered with longer speech segments.

### Metrics

The metrics shown on program exit will vary based on the talker's speaking style. If the talker speaks with more frequent pauses, the speech segments are shorter and the mean inference time will be lower. This is a feature of the Moonshine model described in [our paper](https://arxiv.org/abs/2410.15608). When benchmarking, use the same speech, e.g., a recording of someone talking.

# Demo: Live captioning a WebRTC stream with FastRTC

https://github.com/user-attachments/assets/dbf21903-7d95-46c0-b07e-d5cba8a0d45b

The [`moonshine-onnx/live_captions_web.py`](/demo/moonshine-onnx/live_captions_web.py) script demonstrates the use of Moonshine for live captioning of a WebRTC stream using [FastRTC](https://github.com/freddyaboulton/fastrtc). This demo is intended to show ML engineers who are mainly comfortable with or prefer Python how to build realtime web applications with Moonshine entirely in Python. To run the demo, make sure you've installed the requirements:

```shell
uv pip install -r moonshine/demo/moonshine-onnx/requirements.txt
```

Then run the script and navigate to the Gradio UI in your browser using the URL given:

```shell
python moonshine/demo/moonshine-onnx/live_captions_web.py
...
* Running on local URL:  http://127.0.0.1:7860
```

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
