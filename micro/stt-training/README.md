# Train your own on-device command recognizer

This folder is a self-contained recipe for training a small speech-command
classifier and deploying it to the RP2350 (Raspberry Pi Pico 2) firmware in
[`moonshine-micro/stt`](../stt). You give it a list of words; it gives you back
a quantized `.tflite` model (about 1.3 MB, ~1 M parameters) that runs in real
time on a microcontroller.

The model is the same `WordCNN` (a MobileNetV2-style log-mel classifier) that
the shipped spelling example uses, with the same mel geometry and op set, so the
export is a drop-in for the existing firmware — only the vocabulary changes.

Training data comes from two sources, both free:

- **Moonshine Voice ZipVoice** synthesizes every command word in 15 different
  voices (covers all your words, even rare ones).
- **People's Speech** supplies real-speaker recordings, mined and force-aligned
  from the [MLCommons corpus](https://mlcommons.org/datasets/peoples-speech/),
  plus generic non-command speech for a `_unknown_` reject class.

**The only file you normally edit is [`words.txt`](words.txt).** Its contents
drive the number and names of the model's classes all the way through to the
firmware.

---

## What you need

- Linux with an NVIDIA GPU (training runs on CPU/Mac too, just slower).
- Python 3.10–3.12, or Docker if CUDA on your box lives in a container (see
  [Running on an NVIDIA GPU box where CUDA lives in Docker](#running-on-an-nvidia-gpu-box-where-cuda-lives-in-docker)).
- ~15 GB of free disk for generated/mined audio and checkpoints (more if you
  download the optional MUSAN/RIR augmentation assets).
- Internet access (ZipVoice model download + anonymous People's Speech streaming;
  no Hugging Face account or token needed).

## Step 0 — Setup

```bash
cd moonshine-micro/stt-training
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The first synthesis run downloads the ZipVoice models from the Moonshine CDN and
caches them; subsequent runs are offline for TTS.

On PyPI/Linux the default `torch`/`torchaudio` wheels are CUDA builds, so this
plain install gives you GPU support as long as you have a recent NVIDIA driver.
`torch` and `torchaudio` are pinned to the **same** version in
`requirements.txt` on purpose — `litert-torch` (used only for export) pins an
exact `torch`, and an unpinned `torchaudio` will silently install a build for a
different `torch` and fail to load (`undefined symbol` / `Could not load
libtorchaudio`). If you bump one, bump both together.

### Running on an NVIDIA GPU box where CUDA lives in Docker

Some GPU machines only expose CUDA through NVIDIA's container runtime (the host
Python may be too new for CUDA wheels). In that case, use the NGC PyTorch image
as a CUDA-capable Python 3.12 environment and build a normal venv **inside** it:

```bash
# On the host: start a container with the GPU and this repo mounted.
docker run -d --name stt --gpus all --ipc=host \
    -v "$PWD":/workspace/repo -w /workspace/repo/moonshine-micro/stt-training \
    nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity

docker exec -it stt bash
```

```bash
# Inside the container:
unset PIP_CONSTRAINT          # NGC pins torch to its custom build; this frees pip
python -m venv /workspace/venv
. /workspace/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Two things matter here:

- **`unset PIP_CONSTRAINT`** — NGC images set `PIP_CONSTRAINT=/etc/pip/constraint.txt`,
  which pins `torch` to NVIDIA's prebuilt version and makes `pip install -r
  requirements.txt` fail to resolve. Clearing it lets pip install the standard
  CUDA wheels (which bundle their own CUDA runtime and support recent GPUs).
- We deliberately install stock wheels into a fresh venv rather than reuse the
  image's built-in torch: NGC's custom torch has an ABI that the PyPI
  `torchaudio` wheel can't link against.

Then run everything (`./run_all.sh`, the per-step commands, etc.) inside that
activated venv. Use `docker exec -d stt bash -lc '. /workspace/venv/bin/activate
&& ./run_all.sh > run.log 2>&1'` for a long unattended run.

## The one-command path

If you just want the default 20-word robot vocabulary, edit nothing and run:

```bash
./run_all.sh
```

This runs every step below in order and prints the two deploy artifacts at the
end. Override any knob inline, e.g. `PS_ROWS=50000 EPOCHS=40 ./run_all.sh`
(see [`config.sh`](config.sh) for all of them). The rest of this README explains
each step so you can run them individually and tune them.

## Step 1 — Choose your words

Edit [`words.txt`](words.txt): one command per line, `#` for comments. The
default is a 20-word robot-command set chosen to avoid rhymes and minimal pairs
(the hardest thing for a small classifier):

```
go stop left right forward reverse turn spin faster slower
wait follow come home dance circle dock climb patrol explore
```

A reserved `_unknown_` class is **added automatically** (you don't list it), so
the robot can ignore speech that isn't a command. Tips: avoid near-homophones
(`go`/`no`, `left`/`lift`), and prefer words that also occur in everyday speech
so there's more real data to mine.

## Step 2 — Synthesize command words (ZipVoice)

```bash
python tools/synthesize.py --words-file words.txt
```

Writes `data/tts/<voice>/<word>/*.wav` at 16 kHz — every word × 15 built-in
voices × a few speeds. Re-running skips clips that already exist. Use
`--dry-run` to preview, `--voices zipvoice_american_male ...` to limit voices.

## Step 3 — Mine People's Speech

Scan transcripts for your command words and fetch matching audio:

```bash
python tools/mine_peoples_speech.py --words-file words.txt \
    --split train --limit 200000
```

And mine generic (non-command) utterances for the `_unknown_` class:

```bash
python tools/mine_peoples_speech.py --words-file words.txt \
    --split train --limit 20000 --unknown
```

Both write JSONL manifests + a 16 kHz audio cache under `data/mined/`. They
resume: re-running continues where you left off, so you can grow the dataset by
raising `--limit` or paging with `--offset`.

No Hugging Face account or token is required. Scanning reads the Hub's
auto-generated **Parquet export** anonymously over HTTP range requests, pulling
only the tiny `text` column while searching and downloading a clip's audio only
when its transcript actually contains one of your words — so you never fetch the
full ~400 GB dataset. (This replaces the old `datasets-server` REST API, which
is queue-backed and frequently returns HTTP 503.)

> People's Speech is real, messy audio: some words will get few matches. That's
> fine — ZipVoice covers every word, and the extractor's quality filters drop
> bad clips. Common words yield thousands of clips; rare ones may yield none.

## Step 4 — Cut aligned clips

```bash
python tools/extract_clips.py --words-file words.txt
```

Force-aligns each mined utterance with torchaudio's `MMS_FA` model, cuts one
clip around each command word (neighbour-clamped so adjacent words can't bleed
in), and cuts random 1 s windows for `_unknown_`. Output is the
Speech-Commands layout the trainer reads: `data/peoples_speech/<label>/*.wav`.
Resumes via `data/peoples_speech/extracted_keys.jsonl`.

## Step 5 — (optional) Download noise + reverb

```bash
python tools/download_musan_rirs.py
```

Fetches a subset of [MUSAN](https://www.openslr.org/17/) noise and
[OpenSLR-26](https://www.openslr.org/26/) room impulse responses into
`data/musan/noise/` and `data/rirs/`. Training works without this (it falls back
to synthetic colored noise), but real noise/reverb makes the model noticeably
more robust in a real room.

## Step 6 — Train

```bash
python -m stt_training.train --words-file words.txt --epochs 60
```

Auto-discovers the roots from step 2/4, builds a speaker-independent train/val
split (no voice appears in both), and trains with on-the-fly GPU augmentation
(gain, shift, noise, band-limiting, MUSAN/RIR if present), SpecAugment, mixup and
class-balanced sampling. Checkpoints land in `checkpoints/run_<timestamp>/`
(`word_cnn.pt` = best val accuracy). Mel geometry and stem stride default to the
firmware-required values — don't change them unless you also re-tune the arena.

## Step 7 — Evaluate

```bash
python -m stt_training.evaluate --checkpoint checkpoints/run_XXXX
```

Reports overall + macro accuracy, per-class recall, and the most common
confusions on the held-out speaker split. If two commands confuse each other a
lot, they probably sound too similar — swap one in `words.txt`. Add
`--tflite checkpoints/run_XXXX/spelling_cnn_mel_int8.tflite` after step 8 to
confirm the quantized model matches.

## Step 8 — Export for the RP2350

```bash
python -m stt_training.export --checkpoint checkpoints/run_XXXX
```

Converts the classifier to int8 LiteRT, calibrated on your real clips, and
re-serializes the flatbuffer with inlined weight buffers so TensorFlow Lite
Micro can load it. Produces two files (names already match what the firmware
looks for):

```
spelling_cnn_mel_int8.tflite   # the int8 model (~1.3 MB)
spelling_cnn_meta.json         # class order + audio config
```

An `int8 parity: argmax agreement N/N` line confirms the quantized model agrees
with PyTorch on the calibration clips.

---

## Integrating into the RP2350 demo

The firmware compiles the model and class list in as C arrays (the Pico has no
filesystem). From the repo root:

1. **Copy the two artifacts** into the firmware's model folder:

   ```bash
   cp checkpoints/run_XXXX/spelling_cnn_mel_int8.tflite \
      ../models/spelling_cnn_mel_int8.tflite
   cp checkpoints/run_XXXX/spelling_cnn_meta.json \
      ../models/spelling_cnn_meta.json
   ```

   (Paths are relative to `moonshine-micro/stt-training/`; the firmware looks for
   these exact names under `moonshine-micro/models/`.)

2. **Regenerate the embedded blobs.** This reads the `.tflite` + `meta.json` and
   writes `model_data.{h,cc}`, `classes.{h,cc}`, `mel_tables.{h,cc}`,
   `audio_config.h`, and `test_clips.{h,cc}`:

   ```bash
   python ../stt/scripts/generate_embedded_data.py \
       --wavs-dirs moonshine-micro/stt-training/data/peoples_speech,moonshine-micro/stt-training/data/tts
   ```

   `classes.h` and `kNumClasses` are regenerated from `spelling_cnn_meta.json`,
   so your vocabulary (and its size) propagates automatically — the firmware
   picks up however many classes you trained. `audio_config.h` / `mel_tables.*`
   likewise regenerate from the meta sidecar, so the on-device feature extractor
   always matches the exported model.

3. **Build and flash** (see [`examples/rp2350/README.md`](../examples/rp2350/README.md)):

   ```bash
   ../examples/rp2350/scripts/build.sh
   ../examples/rp2350/scripts/flash.sh echo    # live echo demo, or: test
   ../examples/rp2350/scripts/monitor.sh
   ```

### Things to re-check when the vocabulary changes

- **Flash / arena budget.** Model size scales mainly with the number of classes
  (only the final layer grows), so a ≤~30-word vocabulary stays close to the
  ~1.3 MB reference and fits the default arena. If you add many classes or widen
  the model (`--width-mult`), re-validate that the firmware still links and that
  `AllocateTensors()` succeeds — an over-budget arena fails at startup.
- **Label readback.** Unlike the letters example (which maps `a`→"ay" for TTS),
  command labels are ordinary words and pass through any TTS readback unchanged.
  Treat a `_unknown_` prediction as "no command" and ignore it in your app.
- **The `test` firmware** embeds a clip per class and reports on-device accuracy;
  it's the quickest way to confirm the new model behaves the same on-device as it
  did in `evaluate.py`.

---

## Layout

```
stt-training/
  words.txt                 # your vocabulary (edit this)
  requirements.txt
  config.sh                 # pipeline defaults
  run_all.sh                # one-command end-to-end
  stt_training/             # the python package
    words.py                # words.txt -> classes (+ _unknown_)
    model.py                # WordCNN architecture
    features.py             # log-mel front-end + SpecAugment
    augment.py              # GPU waveform augmentation
    dataset.py              # local-wav dataset, speaker split, sampler, losses
    train.py                # training loop + CLI
    export.py               # int8 LiteRT export + meta.json (+ TFLM buffer inlining)
    evaluate.py             # .pt and .tflite evaluation
    checkpoint.py           # shared checkpoint loading
  tools/
    synthesize.py           # ZipVoice synthesis
    mine_peoples_speech.py  # People's Speech mining (commands + _unknown_)
    extract_clips.py        # MMS_FA alignment + clip cutting
    download_musan_rirs.py  # optional noise/RIR download
  data/                     # generated audio + checkpoints (gitignored)
```
