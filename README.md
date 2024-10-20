# Moonshine

---

[[Blog]](https://petewarden.com/2024/10/20/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/) [[Paper]](https://github.com/usefulsensors/moonshine/blob/main/moonshine_paper.pdf) [[Model card]](https://github.com/usefulsensors/moonshine/blob/main/model-card.md) [[Podcast]](https://notebooklm.google.com/notebook/d787d6c2-7d7b-478c-b7d5-a0be4c74ae19/audio)

Moonshine is an automatic speech recognition (ASR) for English. It's intended
use is in real time scenarios such as live transcription and voice command
recognition. Moonshine obtains word-error rates (WER) better than similarly
sized Whisper models from OpenAI on the datasets used in the [OpenASR
leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
maintained by HuggingFace.

| WER        | Moonshine Tiny | Whisper Tiny.en |
| ---------- | -------------- | --------------- |
| Average    | **12.66**      | 12.81           |
| AMI        | 22.77          | 24.24           |
| Earnings22 | 21.25          | 19.12           |
| Gigaspeech | 14.41          | 14.08           |
| LS Clean   | 4.52           | 5.66            |
| LS Other   | 11.71          | 15.45           |
| SPGISpeech | 7.70           | 5.93            |
| Tedlium    | 5.64           | 5.97            |
| Voxpopuli  | 13.27          | 12.00           |

| WER        | Moonshine Base | Whisper Base.en |
| ---------- | -------------- | --------------- |
| Average    | **10.07**      | 10.32           |
| AMI        | 17.79          | 21.13           |
| Earnings22 | 17.65          | 15.09           |
| Gigaspeech | 12.19          | 12.83           |
| LS Clean   | 3.23           | 4.25            |
| LS Other   | 8.18           | 10.35           |
| SPGISpeech | 5.46           | 4.26            |
| Tedlium    | 5.22           | 4.87            |
| Voxpopuli  | 10.81          | 9.76            |

## Setup

* Install `uv` for Python environment management
  
  - Follow instructions [here](https://github.com/astral-sh/uv)

* Create and activate virtual environment
  
  ```shell
    uv venv env_moonshine
    source env_moonshine/bin/activate
  ```

* Install the `useful-moonshine` package from this github repo

  ```shell
  uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
  ```

  `moonshine` inference code is written in Keras and can run with the backends
  that Keras supports. The above command will install with the PyTorch
  backend. To run the provided inference code, you have to instruct Keras to use
  the PyTorch backend by setting and environment variable .

  ```shell
  export KERAS_BACKEND=torch
  ```

  To run with TensorFlow backend, run the following to install Moonshine.

  ```shell
  uv pip install useful-moonshine[tensorflow]@git+https://github.com/usefulsensors/moonshine.git
  export KERAS_BACKEND=tensorflow
  ```

  To run with jax backend, run the following:

  ```shell
  uv pip install useful-moonshine[jax]@git+https://github.com/usefulsensors/moonshine.git
  export KERAS_BACKEND=jax
  # Use useful-moonshine[jax-cuda] for jax on GPU
  ```

* Test transcribing an audio file

  ```shell
  python
  >>> import moonshine
  >>> moonshine.transcribe(moonshine.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny')
  ['Ever tried ever failed, no matter try again, fail again, fail better.']
  ```

  * The first argument is the filename for an audio file, the second is the name of a moonshine model. `moonshine/tiny` and `moonshine/base` are the currently available models.
