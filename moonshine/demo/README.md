# Moonshine Demos

This directory contains various scripts to demonstrate the capabilities of the
Moonshine ASR models.

- [Moonshine Demos](#moonshine-demos)
- [Standalone file transcription.](#standalone-file-transcription)
- [Live caption microphone demo.](#live-caption-microphone-demo)
  - [Installation.](#installation)
    - [Environment.](#environment)
    - [Download the ONNX models.](#download-the-onnx-models)
  - [Run the demo.](#run-the-demo)
  - [Script notes.](#script-notes)
    - [Speech truncation.](#speech-truncation)
    - [Running on a slower processor.](#running-on-a-slower-processor)
    - [Metrics.](#metrics)
- [Future work.](#future-work)
- [Citation.](#citation)


# Standalone file transcription.

The script
[onnx_standalone.py](/moonshine/demo/onnx_standalone.py)
demonstrates how to run a Moonshine model with the `onnxruntime`
package alone, without depending on `torch` or `tensorflow`. This enables
running on SBCs such as Raspberry Pi. Follow the instructions below to setup
and run.

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


# Live caption microphone demo.

The script
[live_captions.py](/moonshine/demo/live_captions.py) runs Moonshine model on segments of speech detected in the microphone
signal using a voice activity detector called
[silero-vad](https://github.com/snakers4/silero-vad).  The script prints
scrolling text or "live captions" assembled from the model predictions.

The following steps were tested in `uv` virtual environment v0.4.25 created in
Ubuntu 22.04 and Ubuntu 24.04 home folders running on a MacBook Pro M2 (ARM)
virtual machine.

- [Moonshine Demos](#moonshine-demos)
- [Standalone file transcription.](#standalone-file-transcription)
- [Live caption microphone demo.](#live-caption-microphone-demo)
  - [Installation.](#installation)
    - [Environment.](#environment)
    - [Download the ONNX models.](#download-the-onnx-models)
  - [Run the demo.](#run-the-demo)
  - [Script notes.](#script-notes)
    - [Speech truncation.](#speech-truncation)
    - [Running on a slower processor.](#running-on-a-slower-processor)
    - [Metrics.](#metrics)
- [Future work.](#future-work)
- [Citation.](#citation)

## Installation.

This install does not depend on `tensorflow`.  We're using
[silero-vad](https://github.com/snakers4/silero-vad)
which has `torch` dependency.

### Environment.

Moonshine installation steps are available in the
[top level README](/README.md) of this repo.  Note that this demo is standalone
and has no requirement to install `useful-moonshine`.

First install the `uv` standalone installer as
[described here](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).
Close the shell and re-open after the install.  If you don't want to use `uv`
simply skip the virtual environment installation and subsequent activation, and
leave `uv` off the shell commands for `pip install`.

Create the virtual environment and install dependences for Moonshine.
```console
cd
uv venv env_moonshine_demo
source env_moonshine_demo/bin/activate
```

You will need to clone the repo first:
```console
git clone git@github.com:usefulsensors/moonshine.git
```

Then install the demo's extra requirements:
```console
uv pip install -r moonshine/moonshine/demo/requirements.txt
```

Ubuntu needs PortAudio installing for the package `sounddevice` to run.  The
latest version 19.6.0-1.2build3 is suitable.
```console
cd
sudo apt update
sudo apt upgrade -y
sudo apt install -y portaudio19-dev
```

### Download the ONNX models.

The script finds ONNX base or tiny models in the
`demo/models//moonshine_base_onnx` and `demo/models//moonshine_tiny_onnx`
sub-folders.

Download Moonshine `onnx` model files from huggingface hub.
```console
cd
mkdir moonshine/moonshine/demo/models
mkdir moonshine/moonshine/demo/models/moonshine_base_onnx
mkdir moonshine/moonshine/demo/models/moonshine_tiny_onnx

cd
cd moonshine/moonshine/demo/models/moonshine_base_onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/preprocess.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/encode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/uncached_decode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/cached_decode.onnx

cd
cd moonshine/moonshine/demo/models/moonshine_tiny_onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/tiny/preprocess.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/tiny/encode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/tiny/uncached_decode.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/tiny/cached_decode.onnx
```

## Run the demo.

Check your microphone is connected and the microphone volume setting is not
muted in your host OS or system audio drivers.
```console
cd
source env_moonshine_demo/bin/activate

python3 moonshine/moonshine/demo/live_captions.py
```
Speak in English language to the microphone and observe live captions in the
terminal.  Quit the demo with ctrl + C to see console print of the captions.

An example run on Ubuntu 24.04 VM on MacBook Pro M2 with Moonshine base ONNX
model.
```console
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$ python3 moonshine/moonshine/demo/live_captions.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
Loading Moonshine model '/home/parallels/moonshine/moonshine/demo/models/moonshine_base_onnx' ...
Press Ctrl+C to quit live captions.

hine base model being used to generate live captions while someone is speaking. ^C

             model_size :  moonshine_base_onnx
       MIN_REFRESH_SECS :  0.2s

      number inferences :  25
    mean inference time :  0.14s
  model realtime factor :  27.82x

Cached captions.
This is an example of the Moonshine base model being used to generate live captions while someone is speaking.
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$
```

For comparison this is the Faster-Whisper int8 base model on the same instance.
The value of `MIN_REFRESH_SECS` was increased as the model inference is too slow
for a value of 0.2 seconds.
```console
(env_moonshine_faster_whisper) parallels@ubuntu-linux-2404:~$ python3 moonshine/moonshine/demo/live_captions.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
Loading Faster-Whisper int8 base.en model  ...
Press Ctrl+C to quit live captions.

sper int8 base model being used to generate captions while someone is speaking. ^C

             model_size :  base.en
       MIN_REFRESH_SECS :  1.0s

      number inferences :  7
    mean inference time :  0.86s
  model realtime factor :  5.77x

Cached captions.
This is an example of the faster whisper int8 base model being used to generate captions while someone is speaking.
(env_moonshine_faster_whisper) parallels@ubuntu-linux-2404:~$
```

## Script notes.

You may customize this script to display Moonshine text transcriptions as you wish.

The script `live_captions.py` loads the English language version of Moonshine
base ONNX model.  The script includes logic to detect speech activity and limit
the context window of speech fed to the Moonshine model.  The returned
transcriptions are displayed as scrolling captions.  Speech segments with pauses
are cached and these cached captions are printed on exit.  The printed captions
on exit will not contain the latest displayed caption when there was no pause
in the talker's speech prior to pressing ctrl + C.  Stop speaking and wait
before pressing ctrl + C.  If you are running on a slow or throttled processor
such that the model inferences are not realtime, after speaking stops you should
wait longer for the speech queue to be processed before pressing ctrl + C.

### Speech truncation.

Some hallucinations will be seen when the script is running: one reason is
speech gets truncated out of necessity to generate the frequent refresh and
timeout transcriptions.  Truncated speech contains partial or sliced words for
which transcriber model transcriptions are unpredictable.  See the printed
captions on script exit for the best results.

### Running on a slower processor.
If you run this script on a slower processor consider using the `tiny` model.
```console
cd
source env_moonshine_demo/bin/activate

python3 ./moonshine/moonshine/demo/live_captions.py moonshine_tiny_onnx
```
The value of `MIN_REFRESH_SECS` will be ineffective when the model inference
time exceeds that value.  Conversely on a faster processor consider reducing
the value of `MIN_REFRESH_SECS` for more frequent caption updates.  On a slower
processor you might also consider reducing the value of `MAX_SPEECH_SECS` to
avoid slower model inferencing encountered with longer speech segments.

### Metrics.
The metrics shown on program exit will vary based on the talker's speaking
style.  If the talker speaks with more frequent pauses the speech segments are
shorter and the mean inference time will be lower.  This is a feature of the
Moonshine model described in
[our paper](https://arxiv.org/abs/2410.15608).
When benchmarking use the same speech such as a recording of someone talking.

# Future work.

* [x] ONNX runtime model version.

# Citation.

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
