# Live caption microphone demo.

This folder contains the demo Python script
[live_captions.py](/moonshine/demo/live_captions.py).
The script runs Moonshine model on segments of speech detected in the microphone
signal using a voice activity detector called
[SileroVAD](https://github.com/snakers4/silero-vad).  The script prints
scrolling text or "live captions" assembled from the model predictions.

The following steps were tested in `uv` virtual environment v0.4.25 created in
Ubuntu 22.04 home folder running on a MacBook Pro M2 virtual machine.

- [Live caption microphone demo.](#live-caption-microphone-demo)
- [Installation.](#installation)
- [Run the demo.](#run-the-demo)
- [Script notes.](#script-notes)
- [Future work.](#future-work)
- [Citation.](#citation)

# Installation.

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
```
cd
uv venv env_moonshine_demo
source env_moonshine_demo/bin/activate

git clone git@github.com:usefulsensors/moonshine.git

uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
uv pip install -r moonshine/moonshine/demo/requirements.txt
```

# Run the demo.

Check your microphone is connected and the microphone volume setting is not
muted in your host OS or system audio drivers.
```console
cd
source env_moonshine_demo/bin/activate

cd moonshine/moonshine

export KERAS_BACKEND=torch

python3 ./demo/live_captions.py
```
Speak in English language to the microphone and observe live captions in the
terminal.  Quit the demo with ctrl + C to see console print of the captions.

An example run on Ubuntu VM on MacBook Pro M2.
```console
(env_moonshine_demo) parallels@ubuntu-linux-22-04-02-desktop:~/moonshine/moonshine$ export KERAS_BACKEND=torch
(env_moonshine_demo) parallels@ubuntu-linux-22-04-02-desktop:~/moonshine/moonshine$ python3 ./demo/live_captions.py
Loading Moonshine model ...
/home/parallels/env_moonshine_demo/lib/python3.10/site-packages/keras/src/ops/nn.py:545: UserWarning: You are using a softmax over axis 3 of a tensor of shape torch.Size([1, 8, 1, 1]). This axis has size 1. The softmax operation will always return the value 1, which is likely not what you intended. Did you mean to use a sigmoid instead?
  warnings.warn(
Press Ctrl+C to quit live captions.

^C
      number inferences :  40
    mean inference time :  0.77s
  model realtime factor :  7.10x

Cached captions.
Being in Germany after nearly dying from being poisoned by Russian agents, and you and he walked through the terminal after he landed, and then he was immediately arrested in customs, and imprisoned never to be free again. Did you know at that moment that that may be the last time you were together? I didn't think about that at this moment. I knew that we are going at our homeland. We wanted to go there on you that it was very important for. My husband to cop back to russia to show that he is not afraid to show and to encourage all his supporters not to be afraid i knew that it's very important for him and I knew that it could be dangerous but I knew that he would never do it in another way.
(env_moonshine_demo) parallels@ubuntu-linux-22-04-02-desktop:~/moonshine/moonshine$
```

# Script notes.

You may customize this script to display Moonshine text transcriptions as you wish.

The script `live_captions.py` loads the English language version of Moonshine
model.  The script includes logic to detect speech activity and limit the
context window of speech fed to the Moonshine model.  The returned
transcriptions are displayed as scrolling captions.  Speech segments with pauses
are cached and these cached captions are printed on exit.  The printed captions
on exit will not contain the latest displayed caption when there was no pause
in the talker's speech prior to pressing ctrl + C.  Stop speaking and wait
before pressing ctrl + C.  If you are running on a slow or throttled processor
such that the model inferences are not realtime, after speaking stops you should
wait longer for the speech queue to be processed before pressing ctrl + C.

Some hallucinations will be seen when the script is running: one reason is
speech gets truncated out of necessity to generate the frequent refresh and
timeout transcriptions.  Truncated speech contains partial or sliced words for
which transcriber model transcriptions are unpredictable.  See the printed
captions on script exit for the best results.

If you run this script on a slower processor, the value of `MIN_REFRESH_SECS`
will be ineffective when the model inference time exceeds this value.
Conversely on a faster processor consider reducing this value for more frequent
caption updates.  For example on Microsoft Surface Pro 11 Snapdragon with ONNX
runtime and Moonshine base model we're running a value of
`MIN_REFRESH_SECS = 0.2` which looks faster to the eye.  Also on a slower
processor you might consider reducing the value of `MAX_SPEECH_SECS` to avoid
slower model inferencing encountered with longer speech segments.

# Future work.

* [ ] ONNX runtime model version.

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
