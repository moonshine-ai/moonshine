
https://github.com/user-attachments/assets/aa65ef54-d4ac-4d31-864f-222b0e6ccbd3

# Demo: Live captioning from microphone input

This folder contains a demo of live captioning from microphone input, built on Moonshine. The script runs the Moonshine model on segments of speech detected in the microphone signal using a voice activity detector called [`silero-vad`](https://github.com/snakers4/silero-vad). The script prints scrolling text or "live captions" assembled from the model predictions to the console.

The following steps have been tested in a `uv` (v0.4.25) virtual environment on these platforms:

- macOS 14.1 on a MacBook Pro M3
- Ubuntu 22.04 VM on a MacBook Pro M2
- Ubuntu 24.04 VM on a MacBook Pro M2

## Installation

### 0. Setup environment and install the Moonshine package

Moonshine installation steps are available in the [top level README](/README.md) of this repo. Ensure you have the package installed before moving onto the next steps.

### 1. Clone the repo and install extra dependencies

You will need to clone the repo first:

```shell
git clone git@github.com:usefulsensors/moonshine.git
```

Then install the demo's extra requirements:

```shell
uv pip install -r moonshine/moonshine/demo/requirements.txt
```

### Ubuntu: Install PortAudio

Ubuntu needs PortAudio for the `sounddevice` package to run. The latest version (19.6.0-1.2build3 as of writing) is suitable.
```shell
sudo apt update
sudo apt upgrade -y
sudo apt install -y portaudio19-dev
```

### 2. Prepare ONNX models

The script finds ONNX base or tiny models in the `demo/models/base` and `demo/models/tiny` sub-folders.

TODO: add instructions for the ONNX models.

### 3. Run the demo

First, check that your microphone is connected and that the volume setting is not muted in your host OS or system audio drivers.

Then, run the demo:

```shell
python3 ./moonshine/moonshine/demo/live_captions.py base
```

Speak in English language to the microphone and observe live captions in the terminal. Quit the demo with `Ctrl+C` to see a full printout of the captions.

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
Kamala Harris and Donald Trump have seen President Zelensky's victory plan setting out his country's
vision for the future. Ukrainians are just days away from discovering how much and for how long the
next president of its biggest aid provider is willing to help them to stay in the fight. This edition
of the inquiry was presented by me, Charmaine Cozier. The producer was Jill Collins, researcher Matt
Dawson, editor Tara McDermott and Technica producer Ben Howton. This is the BBC World Service, and Alvin
Hole is going home. This is Wakala County, Florida. When you cross that Kana line, oh yeah, you're going
to feel free. As a child, I thought of it as a place of incredible beauty. But this is also a place where
the past is very, very, Part from being the past. You go down the road and the like people left on the
left And white people have on the right. I believe that to understand the United States, you need to know
about places like Wakala County. It's a power struggle. That's what i feel you know fear Do you make you
do a lot of things? In an election year, in a divided country, I've come home to see family and friends
and to share their America. Alvin holes are the ramarica at bbcworldservice.com/documentaries or wherever
you get your BBC podcasts.
(env_moonshine_demo) parallels@ubuntu-linux-2404:~$
```

## Notes

You may customize this script to display Moonshine text transcriptions as you wish.

The script `live_captions.py` loads the English language version of Moonshine base ONNX model. It includes logic to detect speech activity and limit the context window of speech fed to the Moonshine model. The returned transcriptions are displayed as scrolling captions. Speech segments with pauses are cached and these cached captions are printed on exit. The printed captions on exit will not contain the latest displayed caption when there was no pause in the talker's speech prior to pressing `Ctrl+C`. Stop speaking and wait before pressing `Ctrl+C`. If you are running on a slow or throttled processor such that the model inferences are not realtime, after speaking stops you should wait longer for the speech queue to be processed before pressing `Ctrl+C`.

### Speech truncation and hallucination

Some hallucinations will be seen when the script is running: one reason is speech gets truncated out of necessity to generate the frequent refresh and timeout transcriptions. Truncated speech contains partial or sliced words for which transcriber model transcriptions are unpredictable. See the printed captions on script exit for the best results.

### Running on slower processors

If you run this script on a slower processor, consider using the `tiny` model:

```shell
python3 ./moonshine/moonshine/demo/live_captions.py tiny
```
The value of `MIN_REFRESH_SECS` will be ineffective when the model inference time exceeds that value.  Conversely on a faster processor consider reducing the value of `MIN_REFRESH_SECS` for more frequent caption updates.  On a slower processor you might also consider reducing the value of `MAX_SPEECH_SECS` to avoid slower model inferencing encountered with longer speech segments.

### Understanding metrics

The metrics shown on program exit will vary based on the talker's speaking style. If the talker speaks with more frequent pauses, the speech segments are shorter and the mean inference time will be lower. This is a feature of the Moonshine model described in [our paper](https://arxiv.org/abs/2410.15608). When benchmarking, use the same speech, e.g., a recording of someone talking.

## TODO

* [x] ONNX runtime model version

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
