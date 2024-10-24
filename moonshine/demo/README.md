# Live caption microphone demo.

This folder contains the demo Python script
[live_captions.py](/moonshine/demo/live_captions.py).
The script runs Moonshine model on segments of speech detected in the microphone
signal using a voice activity detector called
[SileroVAD](https://github.com/snakers4/silero-vad).  The script prints
scrolling text or "live captions" assembled from the model predictions.

The following steps were tested in `uv` virtual environment v0.4.25 created in
Ubuntu 22.04 and Ubuntu 24.04 home folders running on a MacBook Pro M2 (ARM)
virtual machine.

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

# Installation.

## Environment.

Moonshine installation steps are available in the
[top level README](/README.md) of this repo.  The same set of steps are included
here.

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

uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
```

The demo requires these additional dependencies.
```console
uv pip install -r moonshine/moonshine/demo/requirements.txt
```

If you can not find the `moonshine` repo folder in your home folder then
clone with git command and then run pip installs (this issue seen on an single
board computer running Ubuntu).
```console
cd
uv venv env_moonshine_demo
source env_moonshine_demo/bin/activate

git clone git@github.com:usefulsensors/moonshine.git

uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
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

## Download the ONNX models.

The script finds ONNX base or tiny models in the `demo/models/base` and
`demo/models/tiny` sub-folders.

TODO: add download instructions for the ONNX models.

# Run the demo.

Check your microphone is connected and the microphone volume setting is not
muted in your host OS or system audio drivers.
```console
cd
source env_moonshine_demo/bin/activate

python3 ./moonshine/moonshine/demo/live_captions.py base
```
Speak in English language to the microphone and observe live captions in the
terminal.  Quit the demo with ctrl + C to see console print of the captions.

An example run on Ubuntu 24.04 VM on MacBook Pro M2 with Moonshine base ONNX
model (Credit: BBC World Service).
```console
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$ python3 moonshine/moonshine/demo/live_captions.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
Loading Moonshine model '/home/parallels/moonshine/moonshine/demo/models/base' ...
Press Ctrl+C to quit live captions.

ca at bbcworldservice.com/documentaries or wherever you get your BBC podcasts.  ^C
      number inferences :  179
    mean inference time :  0.20s
  model realtime factor :  21.20x

Cached captions.
Kamala Harris and Donald Trump have seen President Zelensky's victory plan setting out his country's vision for the future. Ukrainians are just days away from discovering how much and for how long the next president of its biggest aid provider is willing to help them to stay in the fight. This edition of the inquiry was presented by me, Charmaine Cozier. The producer was Jill Collins, researcher Matt Dawson, editor Tara McDermott and Technica producer Ben Howton. This is the BBC World Service, and Alvin Hole is going home. This is Wakala County, Florida. When you cross that Kana line, oh yeah, you're going to feel free. As a child, I thought of it as a place of incredible beauty. But this is also a place where the past is very, very, Part from being the past. You go down the road and the like people left on the left And white people have on the right. I believe that to understand the United States, you need to know about places like Wakala County. It's a power struggle. That's what i feel you know fear Do you make you do a lot of things? In an election year, in a divided country, I've come home to see family and friends and to share their America. Alvin holes are the ramarica at bbcworldservice.com/documentaries or wherever you get your BBC podcasts.
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$
```

# Script notes.

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

## Speech truncation.

Some hallucinations will be seen when the script is running: one reason is
speech gets truncated out of necessity to generate the frequent refresh and
timeout transcriptions.  Truncated speech contains partial or sliced words for
which transcriber model transcriptions are unpredictable.  See the printed
captions on script exit for the best results.

## Running on a slower processor.
If you run this script on a slower processor consider using the `tiny` model.
```console
cd
source env_moonshine_demo/bin/activate

python3 ./moonshine/moonshine/demo/live_captions.py tiny
```
The value of `MIN_REFRESH_SECS` will be ineffective when the model inference
time exceeds that value.  Conversely on a faster processor consider reducing
the value of `MIN_REFRESH_SECS` for more frequent caption updates.  On a slower
processor you might also consider reducing the value of `MAX_SPEECH_SECS` to
avoid slower model inferencing encountered with longer speech segments.

## Metrics.
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
