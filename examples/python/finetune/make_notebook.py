"""Assemble the LoRA Colab notebook. Kept as a generator so the notebook can be
regenerated after the numbers in it are re-measured."""

import json
import sys

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- intro
md(r"""
# Adapt Moonshine Streaming Medium to a new domain with LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moonshine-ai/moonshine/blob/main/examples/python/finetune/moonshine_lora_domain_adaptation.ipynb)

This notebook takes the public **Moonshine Streaming Medium** speech-to-text model and
teaches it a domain it is bad at — air traffic control **phraseology**, recorded as
close-talk headset in a quiet simulator, not VHF radio — by training a
**286,720-parameter LoRA adapter**, about 0.11% of the model, on two hours of audio.
Everything it uses is public: the model, both datasets, and the same helpers as
`python -m moonshine_voice.lora`. The cells below call into that package rather than
reimplementing the trainer.

The same recipe ships in the Python package as an opt-in extra, so inference installs
do not pull in PyTorch or Transformers. `[finetune]` and `[lora]` install the same
packages:

```bash
pip install 'moonshine-voice[finetune]'
python -m moonshine_voice.lora --dataset atcosim --output-dir ./lora_atc
```

`moonshine-voice finetune --help` is the full flag list. Use `--train-manifest` for your
own JSONL of `{audio, text}` rows. This notebook is the measured phraseology example;
`--dataset uwb_atcc` is the real-VHF research example (CC BY-NC-SA 4.0). Neither is a
band-limit of the other.

Everything below is measured by the notebook itself, on the full 1,901-utterance
in-domain test set and all 2,620 utterances of LibriSpeech test-clean.

| | air traffic control (in-domain) | LibriSpeech (general) |
|---|---|---|
| Moonshine Streaming Medium, as published | 50.7% WER | 2.3% WER |
| **+ LoRA adapter, 2 h in-domain + replay** | **26.3% WER** | 2.8% WER |
| + LoRA adapter, no replay | 26.0% WER | 3.8% WER |

The first thing to take from that table is that 286,720 parameters, trained on two
hours of audio, cut in-domain error in half.

Sections 8 and 9 then take the adapted model the rest of the way: they export it to
the five ONNX graphs Moonshine's runtime actually loads, quantize it with the public
script, score it through that runtime, and then hand the runtime the domain's word list
so its key-term biasing has something to work with. Through the runtime the same model
reads 51.6% published against 32.3% adapted, and the word list takes it to 29.8%,
which is worth as much as another hour of training audio, for the cost of a text file.

The second is the third row, and it is the reason this notebook is written the way it
is. Dropping replay buys **nothing in-domain** and makes the model measurably worse at
ordinary speech. How much worse depends sharply on model size: at 266 M parameters it
costs a point or so of LibriSpeech WER, but run this same notebook on Streaming Tiny
and the no-replay arm takes the canary from 4.6% to 17.2% (a broken model) while its
in-domain number *improves*. Neither outcome is visible from the in-domain column, so
we score a general-domain canary throughout and stop on both numbers together.

Read those rows to about a point. Over five runs of each arm the replay arm landed
25.6-28.0% in-domain with a 2.8-2.9% canary and the no-replay arm 26.0-29.5% with a
3.6-4.2% canary: the in-domain ranges overlap, the canary ranges do not.

**What you need:** a GPU runtime (Runtime → Change runtime type → T4 GPU), about 3 GB
of downloads, and no Hugging Face account or token. Training peaks under 3 GB of GPU
memory. On the GPU we developed it on (an RTX 5090) sections 1-7 run in about fifteen
minutes and the export and runtime sections add half an hour, most of it the quantizer
and the CPU decoding; on a T4, budget two to three hours, or cut
`EVAL_UTTS`/`CANARY_UTTS`/`RUNTIME_UTTS` down to a few hundred and switch `MODEL_ID` to
Streaming Tiny for a quick look first.

**Licensing.** The model is [Moonshine Streaming
Medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium). The replay audio
is [`moonshine-ai/yodas-en-replay`](https://huggingface.co/datasets/moonshine-ai/yodas-en-replay),
CC-BY-3.0. The in-domain corpus is
[ATCOSIM](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html),
free for research and commercial development but **not redistributable**; this
notebook reads a third-party Hub mirror of it for convenience, and if you build on
this you should get the corpus from its copyright holders. Our contribution to it is
the [speaker-disjoint split
definition](https://huggingface.co/datasets/moonshine-ai/atcosim-speaker-disjoint-splits),
which contains no audio.
""")

code(r'''
# From a clone of this repo, install the local extra so the notebook matches the trainer:
#   pip install -e "language-bindings/python[finetune]"
!pip install -q "moonshine-voice[finetune]"

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "NONE - go to Runtime > Change runtime type")
''')

code(r'''
from pathlib import Path
import time

import torch
from moonshine_voice.lora import SAMPLE_RATE

MODEL_ID       = "moonshine-ai/moonshine-streaming-medium"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
WORK           = Path("work"); WORK.mkdir(exist_ok=True)

TRAIN_HOURS    = 2.0            # in-domain audio the adapter trains on
DEV_HOURS      = 0.25           # in-domain held out to choose the stopping point
REPLAY_HOURS   = 6.0            # general-domain audio mixed in
REPLAY_RATIO   = 0.5            # share of training batches drawn from replay
REPLAY_DEV_HRS = 0.2            # general-domain held out to detect forgetting
RANK           = 8
LR             = 1e-3
BATCH          = 8
MAX_STEPS      = 3000
EVAL_EVERY     = 100
PATIENCE       = 4
WARMUP         = 100
EVAL_UTTS      = None           # in-domain utterances scored; None = all 1,901
CANARY_UTTS    = None           # LibriSpeech test-clean utterances; None = all 2,620
SEED           = 0

torch.manual_seed(SEED)

# Shared kwargs for both training arms. Change a knob above and both runs pick it up.
FIT = dict(train_hours=TRAIN_HOURS, dev_hours=DEV_HOURS,
           replay_dev_hours=REPLAY_DEV_HRS, rank=RANK, lr=LR, batch_size=BATCH,
           max_steps=MAX_STEPS, eval_every=EVAL_EVERY, patience=PATIENCE,
           warmup=WARMUP, seed=SEED, device=DEVICE)

print("device", DEVICE, "| sample rate", SAMPLE_RATE)
''')

# --------------------------------------------------------------------------- data
md(r"""
## 1. The data, and one trap worth knowing about

ATCOSIM is ten hours of air traffic controller speech recorded on a close-talk
headset in a quiet simulator — **not VHF radio**. Clipped, fast, full of callsigns
and numbers. Its transcripts follow air traffic convention: lowercase, unpunctuated,
and **digits spelled out as words** (`lufthansa four six five two`). That last detail
turns out to matter enormously, and we will come back to it. Real tower radio is a
different corpus (`--dataset uwb_atcc`); do not treat a filtered copy of ATCOSIM as a
substitute.

The trap is in the splits. The widely-used Hub mirror ships a `train`/`test` split
that is **utterance-random, not speaker-disjoint**: there are only ten speakers, and
all four in the scored `test` half also appear in `train`. Train an adapter on that
`train` split and you have trained on the voices you are about to be scored on, which
in our measurements overstates the adaptation win by about 6 WER points. So we use a
published split definition that holds out whole speakers, and train only on the six
speakers the evaluation never sees.

We read the metadata columns straight off the Hub and download audio shards only when
we actually decode from them, which keeps this to a few hundred megabytes instead of
the full 2.4 GB.
""")

code(r'''
from moonshine_voice.lora import hours_of, index_atcosim

indexed = index_atcosim()
train_pool, scored, other = indexed.train, indexed.scored, indexed.other
all_rows = train_pool + scored + other
print(f"{len(all_rows)} utterances, {hours_of(all_rows):.2f} h, "
      f"{len({r.speaker for r in all_rows})} speakers")
print(f"train pool (speaker-disjoint): {len(train_pool):5d} utts  {hours_of(train_pool):.2f} h  "
      f"speakers {sorted({r.speaker for r in train_pool})}")
print(f"scored:                        {len(scored):5d} utts  {hours_of(scored):.2f} h  "
      f"speakers {sorted({r.speaker for r in scored})}")
''')

# --------------------------------------------------------------------------- model
md(r"""
## 2. The model, and what it does with this audio

`MoonshineStreamingForConditionalGeneration` is the PyTorch implementation of
Moonshine's streaming architecture in Hugging Face Transformers: a convolutional
frontend, a sliding-window encoder, and a decoder that attends to the encoder while
generating tokens. Medium is 266 M parameters, with a 14-layer decoder 640 units wide.
This is the float PyTorch form of the model; the on-device runtime ships a quantized
ONNX export of the same architecture.

Everything here works unchanged on the smaller sizes: set `MODEL_ID` to
`moonshine-ai/moonshine-streaming-tiny` (44 M parameters) for a run that is roughly
six times cheaper, at the cost of a much weaker starting point.

Listen to what it does with air traffic audio. One clip first; the rest of this
section is the same idea in text, then the corpus numbers.
""")

code(r'''
from IPython.display import Audio, display
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
from moonshine_voice.lora import decode_atcosim as decode, transcribe as lora_transcribe

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MoonshineStreamingForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print(f"{sum(p.numel() for p in model.parameters()):,} parameters, "
      f"vocab {model.config.vocab_size}, {model.config.num_hidden_layers} decoder layers")

def transcribe(model, waves, **kwargs):
    """Greedy decode, batched by length. Identical settings for every arm we score."""
    return lora_transcribe(model, processor, waves, DEVICE, **kwargs)

clip = scored[7]
clip_wave = decode([clip])[0]
display(Audio(clip_wave, rate=SAMPLE_RATE))
print("REF  ", clip.text)
print("HYP  ", transcribe(model, [clip_wave])[0])
''')

md(r"""
That is a competent general model meeting a domain whose conventions it has never
seen. It gets most of the words and then writes them as ordinary English: `4393`
where this domain writes `four three nine three`, cased and punctuated where the
domain transcript is neither. It also loses the callsigns that carry the meaning:
`Svisa` for `swiss air`, `Airlight` for `aero lloyd`.

Case and punctuation are free, because the normalizer below throws them away. The
number formatting is not, and it is most of the 50% word error rate you are about to
see. Notice how little of that is **acoustics**: it is convention. That is the usual
shape of a domain gap, and it is exactly what a small adapter can fix.

### How we score it

Two numbers, both computed with the standard Whisper English normalizer (lowercase,
strip punctuation, expand numbers) so we are comparing words and not typography:

- **In-domain WER** on ATCOSIM's scored speakers: did adaptation work?
- **LibriSpeech test-clean WER**: did we break everything else? This is the canary.
  A model that wins in-domain by forgetting general English is not adapted, it is
  damaged, and only the second number tells you.
""")

code(r'''
from moonshine_voice.lora import corpus_wer as wer, english_normalizer

normalize = english_normalizer()

# Why the digits matter so much: the normalizer turns spelled-out numbers into
# joined digits, but leaves the model's hyphenated style as separate tokens.
print(repr(normalize("lufthansa four six five two turn right heading two one zero")))
print(repr(normalize("The fans are 4-6-5-2, turn right heading 2-1-0.")))
''')

code(r'''
import time
from moonshine_voice.lora import librispeech_eval, sample_indices

eval_rows = [scored[i] for i in sample_indices(len(scored), EVAL_UTTS, SEED)]
eval_waves = decode(eval_rows)
canary_refs, canary_waves = librispeech_eval(CANARY_UTTS, SEED)

print(f"scoring {len(eval_rows)} in-domain and {len(canary_refs)} LibriSpeech "
      f"utterances per arm")
started = time.time()
baseline = {"atcosim": wer([r.text for r in eval_rows],
                           transcribe(model, eval_waves)),
            "librispeech": wer(canary_refs, transcribe(model, canary_waves))}
print(f"baseline  in-domain {baseline['atcosim']:.2f}%   "
      f"LibriSpeech {baseline['librispeech']:.2f}%   ({time.time() - started:.0f}s)")
''')

# --------------------------------------------------------------------------- replay
md(r"""
## 3. Replay: the part everyone skips

Fine-tune on a narrow domain and the model drifts away from everything else, gently
at this size, catastrophically for the smaller ones, and never visibly if the only
thing you measure is the domain. The fix that works is boring: keep showing the model
ordinary speech while it learns the new domain. Half the training batches come from a
general-domain corpus. That is all "replay" means.

The catch is that the replay transcripts have to be written the way the model already
writes: **cased and punctuated**. WER normalizers throw away case and punctuation, so
a replay set with bare transcripts scores fine on every benchmark while quietly
teaching the model to stop punctuating. Most permissively-licensed English corpora
fail this test: LibriSpeech is uppercase and unpunctuated, and YouTube caption dumps
are only about a quarter terminally punctuated.

So we built one. [`moonshine-ai/yodas-en-replay`](https://huggingface.co/datasets/moonshine-ai/yodas-en-replay)
is 61 hours of Creative-Commons YouTube audio relabeled with Whisper large-v3-turbo
and filtered to 100% cased, 90% terminally punctuated text, CC-BY-3.0. It is also in
none of the standard benchmark sets, so mixing it in does not flatter the evaluation.

Both corpora get decoded once into a flat `int16` file. Random access into parquet row
groups per example is slow enough to dominate a run this small, so it is cheaper to pay
the decode cost once up front.
""")

code(r'''
from moonshine_voice.lora import (
    atcosim_source, build_cache, encode_text, open_blob, replay_source)

domain_index = build_cache(
    "atcosim", WORK, TRAIN_HOURS + DEV_HOURS,
    lambda hours: atcosim_source(train_pool, hours, "none"),
    lambda text: encode_text(processor, text))
replay_index = build_cache(
    "replay", WORK, REPLAY_HOURS + REPLAY_DEV_HRS, replay_source,
    lambda text: encode_text(processor, text))

domain_audio = open_blob(WORK, "atcosim", domain_index)
replay_audio = open_blob(WORK, "replay", replay_index)
print("replay text, as the model likes to see it:")
for entry in replay_index["entries"][:2]:
    print("  ", processor.tokenizer.decode(entry["tokens"], skip_special_tokens=True))
''')

# --------------------------------------------------------------------------- lora
md(r"""
## 4. The adapter: 286,720 parameters, and where to put them

LoRA replaces a frozen weight $W$ with $W + BA$, where $A$ projects down to rank $r$
and $B$ projects back up. $B$ starts at zero, so an untrained adapter is *exactly* the
original model, and at the end $BA$ can be folded into $W$ so the adapted model has
no extra layers in it at all.

Where you put them matters more than how many you use. We benchmarked every candidate
site in the exported decoder graph and found two very different costs:

- **Decoder self-attention q/k/v** is effectively **free**. The decoder runs one token
  at a time and is dispatch-bound, so a rank-8 side path adds no measurable latency.
- **Cross-attention K/V costs about 13%** on our exported graph, because those
  projections run over the whole audio memory span. Three adapters in the wrong place
  cost more than eighteen in the right one.

So: decoder self-attention only, 3 projections × 14 layers = 42 adapted linears. The
q/k/v of a layer all read the same layer-normed input, so they **share one
down-projection** and get their own up-projections: one matmul in the fused graph,
20,480 parameters per layer, **286,720** for the model. That is 0.11% of Medium, and
just over a megabyte on disk. `moonshine_voice.lora.adapter` is that layout
(`LoRALinear`, shared-A `add_lora`); the cell below just counts the parameters.
""")

code(r'''
from moonshine_voice.lora import LoRALinear

# Medium decoder: 14 layers, 640-wide, rank-8 shared A across q/k/v.
width, layers = 640, 14
params = layers * RANK * (width + 3 * width)
print(LoRALinear.__doc__.strip().splitlines()[0])
print(f"shared-A q/k/v LoRA: {params:,} parameters")
''')

# --------------------------------------------------------------------------- training
md(r"""
## 5. Training

Batching, loss and stopping, in the order they matter:

**Targets are `[BOS] … [EOS]`, padded with 0**, and the loss is cross-entropy between
`logits[:, :-1]` and `tokens[:, 1:]`, ignoring padding. We shift explicitly rather
than passing `labels=` so the alignment is visible and version-independent. If you
ever want to check an alignment like this, compute the loss both ways: the correct
shift on a pretrained model scores about 2.2 here and a wrong one about 10.

**Batches are length-sorted**, so padding does not dominate a corpus of three-second
utterances.

**Replay batches are interleaved to hit `REPLAY_RATIO`** of each epoch, drawn fresh
each epoch. Where the replay pool is smaller than the in-domain set, batches cycle;
otherwise asking for 50% quietly delivers 15%.

**The stopping point is chosen on `in-domain dev loss + held-out replay dev loss`.**
This is the subtle one. Selecting on in-domain loss alone picks whichever step adapted
hardest, which is precisely the step that forgot the most. Summing the two means a step
only counts as an improvement if what it gains in-domain is not paid for out of general
capability.

`fit_adapter` is that loop. Both arms below reuse the caches from section 3; the
no-replay run only changes `replay_ratio`.

This cell usually takes around 10 minutes to run.
""")

code(r'''
from moonshine_voice.lora import fit_adapter

adapted = fit_adapter(
    MODEL_ID, processor, domain_index, domain_audio,
    replay_index=replay_index, replay_audio=replay_audio,
    replay_ratio=REPLAY_RATIO, tag="replay",
    adapter_path=WORK / "adapter_replay.safetensors", **FIT)
''')

md(r"""
The same clip as section 2, before the corpus numbers. If the adapter worked, this is
where you see it: spelled-out digits, callsign templates, no extra punctuation.
""")

code(r'''
display(Audio(clip_wave, rate=SAMPLE_RATE))
print("REF    ", clip.text)
print("before ", transcribe(model, [clip_wave])[0])
print("after  ", transcribe(adapted, [clip_wave])[0])
''')

code(r'''
results = {"published": baseline}
results["lora + replay"] = {
    "atcosim": wer([r.text for r in eval_rows], transcribe(adapted, eval_waves)),
    "librispeech": wer(canary_refs, transcribe(adapted, canary_waves))}

def table(results):
    print(f"{'':22s}  {'in-domain':>10s}  {'LibriSpeech':>12s}")
    for name, r in results.items():
        print(f"{name:22s}  {r['atcosim']:9.2f}%  {r['librispeech']:11.2f}%")
table(results)
''')

md(r"""
## 6. What happens without replay

The same arm again with `replay_ratio=0`: nothing changes except that every batch is
in-domain. This is the version most people write, and if you don't mind another 30 minutes to run, it is worth seeing what it costs
before you trust an in-domain number on its own.
""")

code(r'''
no_replay = fit_adapter(
    MODEL_ID, processor, domain_index, domain_audio,
    replay_ratio=0.0, tag="noreplay",
    adapter_path=WORK / "adapter_noreplay.safetensors", **FIT)
results["lora, no replay"] = {
    "atcosim": wer([r.text for r in eval_rows], transcribe(no_replay, eval_waves)),
    "librispeech": wer(canary_refs, transcribe(no_replay, canary_waves))}
table(results)
del no_replay
''')

md(r"""
Both arms learned the domain equally well. One of them also got worse at everything
else, for free.

At this size that is the whole story, and it is a mild one: over five runs of each arm
the no-replay arm scores 26.0-29.5% in-domain against 25.6-28.0% with replay (the same
range, give or take) while its canary sits at 3.6-4.2% against a tight 2.8-2.9%. You
pay about a point of general-domain WER and get nothing back.

Shrink the model and the same experiment turns nasty. On Streaming Tiny, which starts
at 91.2% in-domain and 4.6% on the canary, the arms measure 45.5% / 6.1% with replay
and 42.1% / 17.2% without: the
no-replay arm now looks like the *winner* on the only column most people plot, while
its general-domain error has tripled. Since the models people actually deploy on
device are the small ones, we write the recipe for that case.

Two dials control where you land:

- **Replay hours.** At 2 hours of replay instead of 6, three seeds average 26.2%
  in-domain and 2.90% on the canary, against 27.4% and 2.86% for 6 hours, a difference
  smaller than the run-to-run spread. At this size the extra replay is neither buying
  nor costing anything measurable; on the smaller models it does matter, which is why
  our production recipe uses 16 hours. Measure it on your own domain rather than
  inheriting the number.
- **What you select on.** Stopping on `in-domain + general` dev loss, as above, refuses
  any step that buys in-domain accuracy out of general capability. Stopping on
  in-domain loss alone (all the no-replay arm can do, having no general-domain data)
  keeps going long after the damage starts.

## 7. Shipping it

Two artifacts come out of this, and they are for different jobs:

- **`adapter_replay.safetensors`, 1.1 MB**: the thing you would actually distribute,
  version, or let a customer download. It is meaningless without the base model.
- **The merged model**: because $W + BA$ is just a weight matrix, the adapted model
  has the same architecture and the same key names as the original. Anything that
  accepts the base model accepts this, with no adapter code at inference and no
  latency cost. That is also why the placement above matters less than it might seem
  for a single-tenant deployment: merged, the adapter is free wherever you put it. The
  placement earns its keep when the adapter has to stay separate, which is the
  multi-tenant and download-an-adapter case.
""")

code(r'''
adapted.save_pretrained(WORK / "adapted")
processor.save_pretrained(WORK / "adapted")

reloaded = MoonshineStreamingForConditionalGeneration.from_pretrained(
    WORK / "adapted").to(DEVICE)
print("REF ", clip.text)
print("HYP ", transcribe(reloaded, [clip_wave])[0])
print("\nfiles:", sorted(p.name for p in (WORK / 'adapted').iterdir()))
''')

# ------------------------------------------------------------------ onnx export
md(r"""
## 8. Exporting to the format that ships

A `save_pretrained` directory is a PyTorch checkpoint, and nothing on a phone, a
laptop or a Raspberry Pi runs one. The Moonshine runtime loads **five ONNX graphs**,
because streaming cuts the model up differently than batch inference does:

| graph | runs | carries |
|---|---|---|
| `frontend` | once per audio chunk | the sample and convolution state that makes chunked filtering match offline filtering |
| `encoder` | once per chunk of frames | nothing: its sliding-window masks are baked into the graph |
| `adapter` | once per chunk | encoder width to decoder width, plus positions |
| `cross_kv` | once per encoder update | the cross-attention keys and values the decoder reads |
| `decoder_kv` | once per emitted token | the self-attention KV cache |

`export_checkpoint` produces all five from a Transformers checkpoint, along
with the `streaming_config.json` the runtime reads its shapes from and a copy of
`tokenizer.bin` (fine-tuning does not touch the tokenizer, so the published one is
correct). The public `quantize-streaming-model.sh` from the deployment repo then turns
each `.onnx` into the quantized `.ort` file the runtime memory-maps.

Worth knowing before you export a fine-tune: **a decoder-only adapter changes exactly
one of the five graphs.** Merging LoRA into the decoder's self-attention leaves 42 of
this checkpoint's 362 tensors different from the published model, and all 42 live in
`decoder_kv`. Feed both exports the same audio and the features, encoder output,
adapter output and cross-attention K/V agree to the bit; the logits do not. So
`--graphs decoder_kv` is enough to ship an adapter (twenty seconds of work) with the
other four graphs copied from the published download. We pass `--graphs all` here
because it is the same command either way and it demonstrates that the whole path runs
from public weights. Budget a couple of minutes for the export and ten to twenty for
the quantizer, which is doing most of the work.
""")

code(r'''
# onnxscript is what torch.onnx.export's dynamo path is written against, and it is
# not a dependency of onnx, so an export dies at the first graph without it.
!pip install -q onnxruntime onnx_shrink_ray

import os, shutil
os.environ["MOONSHINE_VOICE_CACHE"] = str((WORK / "moonshine_cache").resolve())

QUANTIZER = ("https://raw.githubusercontent.com/moonshine-ai/moonshine/"
             "main/scripts/quantize-streaming-model.sh")
!curl -sSL -o quantize-streaming-model.sh {QUANTIZER}

# The published model, for its tokenizer.bin and as the runtime baseline to beat.
!moonshine-voice download --stt --language en 2>&1 | tail -3
PUBLISHED = next((WORK / "moonshine_cache").glob(
    "**/medium-streaming-en/*/streaming_config.json")).parent
print("published runtime model:", PUBLISHED)
''')

code(r'''
import subprocess
from moonshine_voice.lora import export_checkpoint

def run(command, keep=8):
    """Run a command, showing its tail on success and everything on failure."""
    print("$", " ".join(str(part) for part in command))
    done = subprocess.run([str(part) for part in command],
                          capture_output=True, text=True)
    lines = (done.stdout + done.stderr).strip().splitlines()
    print("\n".join(lines if done.returncode else lines[-keep:]))
    if done.returncode:
        raise SystemExit(f"{command[0]} failed ({done.returncode})")

def to_runtime_dir(checkpoint, name, graphs="all"):
    """Export, quantize, and lay out a directory the runtime can load."""
    float_dir, ort_dir = WORK / f"{name}_float", WORK / name
    started = time.time()
    export_checkpoint(checkpoint, float_dir, graphs=graphs,
                      tokenizer_bin=PUBLISHED / "tokenizer.bin")
    run(["bash", "quantize-streaming-model.sh", float_dir])
    ort_dir.mkdir(exist_ok=True)
    for graph in ("frontend", "encoder", "adapter", "cross_kv", "decoder_kv"):
        source = float_dir / f"{graph}.ort"
        shutil.copy(source if source.exists() else PUBLISHED / f"{graph}.ort", ort_dir)
    for extra in ("streaming_config.json", "tokenizer.bin"):
        shutil.copy(float_dir / extra, ort_dir)
    size = sum(p.stat().st_size for p in ort_dir.iterdir()) / 1e6
    print(f"\n{ort_dir}: {size:.0f} MB in {time.time() - started:.0f}s")
    return ort_dir

ADAPTED_ORT = to_runtime_dir(WORK / "adapted", "adapted_ort")
''')

md(r"""
The quantized adapted model is the same size as the published one, which is the point:
the adapter is merged, so there is no adapter to ship, no extra graph to load and no
inference-time cost. Before scoring it, one clip through both paths (PyTorch on the
GPU and the quantized graphs on the CPU) as a check that the export is sound rather
than merely well-formed.
""")

code(r'''
from moonshine_voice import ModelArch, Transcriber
import numpy as np

def runtime_transcribe(transcriber, waves):
    out = []
    for wave in waves:
        result = transcriber.transcribe_without_streaming(
            np.asarray(wave, dtype=np.float32).tolist())
        out.append(" ".join(line.text for line in result.lines).strip())
    return out

clips = [scored[i] for i in (7, 11, 23)]
clip_waves = decode(clips)
runtime = Transcriber(str(ADAPTED_ORT), ModelArch.MEDIUM_STREAMING)
for clip, torch_hyp, ort_hyp in zip(clips, transcribe(adapted, clip_waves),
                                   runtime_transcribe(runtime, clip_waves)):
    print(f"REF    {clip.text}\ntorch  {torch_hyp}\nonnx   {ort_hyp}\n")
''')

# --------------------------------------------------------------- key-term biasing
md(r"""
## 9. Scoring what ships, and adding key-term biasing on top

Two things change when we score through the runtime instead of PyTorch, and they move
WER in opposite directions: the weights are quantized to 8 bits, and decoding is
streaming, so the model commits to tokens with less audio in hand than
`generate` had. Expect the runtime number to be a point or two off the PyTorch one in
either direction. It is the number your users get, so it is the one to quote.

The runtime also has a lever the PyTorch model does not: **key-term biasing**. Hand
`set_context` a passage of text and the runtime tokenizes the rare, multi-subword
words in it into a prefix trie over token IDs, then adds a bonus to the logits of any
token that continues a term already in progress. Nothing is retrained and nothing is
prepended to the prompt: it is a nudge applied just before each argmax, so a callsign
like `lufthansa` gets helped subword by subword instead of having to win outright at
the first one.

The two levers turn out to fix different halves of the problem, and we can show that
rather than assert it. Splitting the full test set into the 118 rare multi-subword
words and everything else, on the same audio:

| | rare terms | everything else |
|---|---|---|
| published Medium | 61.2% WER | 73.2% WER |
| + LoRA adapter | 63.3% WER | **20.5%** WER |

The adapter took ordinary words from 73% error to 20% and left the names exactly where
it found them. That is the "conventions, not vocabulary" result in one line: two hours
of audio taught the model how this domain talks, and nothing about how to spell a
waypoint it saw twice. Biasing is the instrument for that second column.

**Where the terms come from matters more than anything else about them**, and the
obvious source is the wrong one. Feeding the runtime the training transcripts (the
text we already have) is worth 0.6 points here. Feeding it the *domain's word list*
is worth 2.6. The reason is blunt: the words that fail are not in the training audio.
`france` is wrong 98% of the time on the test set and appears **zero times** in the two
hours we trained on, as do `milan`, `geneva`, `reims` and `saronno`; 178 of the test
set's 442 distinct words are absent from the training transcripts. A term list drawn
from them can reach only a quarter of the model's remaining errors, against
three-fifths for a glossary. So below we hand the runtime the sector's vocabulary,
taken from ATCOSIM utterances that this notebook neither trains on nor scores, which
is exactly what a deployment has, since an airline directory or a sector's waypoint
list is not something you need audio to know.

Two rules follow, both measured on the full test set:

| | what to use | what happens if you don't |
|---|---|---|
| put **all** of the vocabulary on the list | 520 words → 29.8% | the 82 hardest words alone → 34.8% |
| scale `keyterm_boost` to the **width** of the list | 3.0 for hundreds of terms | boost 1 → 31.1%, boost 6 → 31.2% |

A narrow list at a high boost is worse than no biasing at all: those few words then win
against everything and the rest of the transcript pays for it. A wide list at the same
boost behaves like a vocabulary prior instead, lifting every plausible domain word
together, and the competition between them survives. The default of 2.0 is sensible in
the middle and wrong at both ends. And the list has to be the *right* vocabulary: 520
words of ordinary English at the same boost cost five points.

The cells below score 200 utterances, because runtime decoding here happens on the CPU.
That is enough to see a 2.6-point effect but not to split hairs: at this size overall
WER carries roughly ±1.5 points of sampling noise, so treat the ordering as real and
the decimals as decoration. The numbers quoted above come from all 1,901.
""")

code(r'''
RUNTIME_UTTS = 200      # runtime decoding is CPU-bound; raise it if you have the time

rt_rows = [scored[i] for i in sample_indices(len(scored), RUNTIME_UTTS, SEED)]
rt_waves = decode(rt_rows)
rt_refs = [r.text for r in rt_rows]
print(f"{len(rt_rows)} utterances, {sum(len(w) for w in rt_waves) / SAMPLE_RATE / 60:.1f} min")

from collections import Counter

# The two candidate sources of terms. The transcripts are what we trained on; the
# glossary is every word the sector uses, taken from ATCOSIM utterances that are
# neither trained on nor scored here; a stand-in for the customer's own word list.
train_text = " ".join(r.text for r in train_pool)
glossary_text = " ".join(r.text for r in other)

def vocabulary(text, min_chars=3):
    """Every word worth biasing, most frequent first."""
    counts = Counter(text.lower().split())
    return [word for word, _ in counts.most_common() if len(word) >= min_chars]

GLOSSARY = vocabulary(glossary_text)
UNSEEN = [w for w in GLOSSARY if w not in set(train_text.lower().split())]
print(f"training transcripts: {len(train_text.split()):,} words, "
      f"{len(set(train_text.lower().split())):,} distinct")
print(f"glossary: {len(glossary_text.split()):,} words, {len(GLOSSARY):,} terms, "
      f"{len(UNSEEN):,} of them never spoken in the training audio")
''')

code(r'''
BOOST = 3.0    # a wide list wants a strong boost; see the table above

def runtime_wer(model_dir, keyterms=None, context=None, boost=BOOST):
    transcriber = Transcriber(str(model_dir), ModelArch.MEDIUM_STREAMING,
                              options={"keyterm_boost": boost})
    if keyterms:
        transcriber.set_keyterms(keyterms)
    elif context:
        # What set_context does for you: pick terms out of a passage. It ranks by
        # frequency and keeps only words of two or more subwords.
        transcriber.set_context(context, 200)
    started = time.time()
    hyps = runtime_transcribe(transcriber, rt_waves)
    audio = sum(len(w) for w in rt_waves) / SAMPLE_RATE
    print(f"  {audio / (time.time() - started):.1f}x realtime")
    return wer(rt_refs, hyps), hyps

def term_recall(hyps, terms):
    """Share of term occurrences in the references that survive into the output."""
    wanted, found, total = set(terms), 0, 0
    for ref, hyp in zip(rt_refs, hyps):
        spoken = Counter(word for word in normalize(ref).split() if word in wanted)
        heard = Counter(normalize(hyp).split())
        total += sum(spoken.values())
        found += sum(min(count, heard[word]) for word, count in spoken.items())
    return 100 * found / max(total, 1)

arms = (("published", PUBLISHED, {}),
        ("+ LoRA adapter", ADAPTED_ORT, {}),
        ("+ LoRA + training text", ADAPTED_ORT, {"context": train_text, "boost": 1.0}),
        ("+ LoRA + glossary", ADAPTED_ORT, {"keyterms": GLOSSARY}))

runtime_results = {}
for label, model_dir, options in arms:
    print(label)
    runtime_results[label] = runtime_wer(model_dir, **options)

print(f"\n{'':26s}  {'WER':>7s}  {'recall, unseen words':>21s}")
for label, (value, hyps) in runtime_results.items():
    print(f"{label:26s}  {value:6.2f}%  {term_recall(hyps, UNSEEN):20.1f}%")
''')

md(r"""
The recall column is the one to read: it counts only the words the adapter never saw
in training, which is what the glossary is there to fix. Here is the same thing one
utterance at a time, where switching the glossary on changed which of those words came
out, in both directions. The left column is what the bias recovered and is the reason to
use it; the right column is what it invented and is the reason not to push the boost
past the width of your list.
""")

code(r'''
def term_changes(before, after, limit=3):
    """Utterances where biasing changed which key terms the output contains."""
    wanted, recovered, invented = set(UNSEEN), [], []
    for ref, plain, biased in zip(rt_refs, before, after):
        spoken = set(normalize(ref).split()) & wanted
        was, now = set(normalize(plain).split()), set(normalize(biased).split())
        gained = spoken & (now - was)
        false = (now & wanted) - spoken - was
        if gained and len(recovered) < limit:
            recovered.append((sorted(gained), ref, plain, biased))
        if false and len(invented) < limit:
            invented.append((sorted(false), ref, plain, biased))
    return recovered, invented

recovered, invented = term_changes(runtime_results["+ LoRA adapter"][1],
                                   runtime_results["+ LoRA + glossary"][1])
for title, cases in (("RECOVERED", recovered), ("INVENTED", invented)):
    for terms, ref, plain, biased in cases:
        print(f"{title} {terms}\n  ref      {ref}\n  plain    {plain}"
              f"\n  biased   {biased}\n")
''')

md(r"""
On the full test set the adapter alone scores 32.4%, the adapter with the training
transcripts as context 31.8%, and the adapter with the glossary 29.8%. On the 82 words
the adapter never saw, the glossary takes recall from 23.6% to 41.7% and term WER from
76.6% to 58.6%, for 0.68 false alarms per thousand words. Those words are 2.5% of the
corpus, which is why a lever that nearly halves their error rate moves the overall
number by two and a half points rather than ten.

That is the honest shape of this lever, and it is worth stating plainly because the
temptation is to expect a headline number. Biasing is free (no training data, no GPU,
one function call) and it is the only one of the two that can take a term list that
changes after you ship. What it cannot do is teach the model that this domain writes
`four six five two` rather than `4652`: that is a convention, and conventions are what
the adapter bought. Neither lever substitutes for the other, which is why the last row
of the table is the one to deploy.
""")

md(r"""
## 10. Doing this for your own domain

Replace `atcosim_source` with your own audio and transcripts and the rest of the
notebook is unchanged. The things that will actually decide whether it works:

**Match your transcripts to the model's conventions, or convert them deliberately.**
This is the most common way these runs fail, and it fails quietly. If your corpus is
ALL CAPS and unpunctuated while the model emits cased punctuated text, most of what
the adapter learns is a typography that the WER normalizer then throws away: the loss
drops convincingly, the WER does not move, and the model starts emitting confident
misspellings. We have watched an adapter spend its whole capacity on that and produce
`yesturate` for a word it used to get right. Lowercase the corpus, or restore its
punctuation, before you train. Here, ATCOSIM's spelled-out digits *are* the convention
we want to learn, so we keep them; the decision is deliberate either way.

**Hold out whole speakers, not random utterances.** Otherwise you measure speaker
adaptation and call it domain adaptation.

**Always score a general-domain canary.** In-domain WER alone cannot distinguish
adaptation from damage.

**Expect conventions, not vocabulary.** An adapter buys register, phrasing, formatting
and disfluencies. It does not reliably teach the spelling of words it saw once or
twice. For a long tail of names or product terms, keyword biasing at decode time is
the better lever, and the two compose.

**One hour is enough to see it work**; the curve flattens quickly after a few hours.
Start small, keep the canary honest, and only buy more data once the pipeline is
proven.

Outside Colab, skip rewriting the cells and pass a JSONL manifest to the package:

```bash
pip install 'moonshine-voice[finetune]'
python -m moonshine_voice.lora --train-manifest domain.jsonl --output-dir ./lora_out
```

Each line of the manifest is `{"audio": "clips/001.wav", "text": "the transcript"}`.
`--dataset atcosim` is this notebook's ATC phraseology run with the speaker-disjoint
split, replay, and the defaults above. Real VHF is a different command. On published
Medium, two hours of UWB-ATCC decoder LoRA moved held-out ATCO2 from 82.2% to 59.6%
WER (the ATCOSIM phraseology adapter only reached 74.5%) and in-domain UWB from
107.8% to 73.1%, with LibriSpeech at 2.7%. UWB-ATCC is CC BY-NC-SA 4.0 (research
only) — for a product radio SKU use your own hours:

```bash
python -m moonshine_voice.lora --dataset uwb_atcc --sites both \
    --eval --eval-dataset atco2 --canary --output-dir ./lora_vhf
python -m moonshine_voice.lora --dataset uwb_atcc --adapt full \
    --eval --eval-dataset atco2 --canary --output-dir ./ft_vhf
```

`--adapt full` and `--sites encoder|both` need `--graphs all` at export time.
Quantize the exported graphs with `scripts/quantize-streaming-model.sh` and load the
resulting directory with `Transcriber` as usual; the inference wheel does not need
the extra.

### More

- Model: [`moonshine-ai/moonshine-streaming-medium`](https://huggingface.co/moonshine-ai/moonshine-streaming-medium)
- Replay corpus: [`moonshine-ai/yodas-en-replay`](https://huggingface.co/datasets/moonshine-ai/yodas-en-replay)
- ATCOSIM split: [`moonshine-ai/atcosim-speaker-disjoint-splits`](https://huggingface.co/datasets/moonshine-ai/atcosim-speaker-disjoint-splits)
- UWB-ATCC (real VHF, NC): [`Jzuluaga/uwb_atcc`](https://huggingface.co/datasets/Jzuluaga/uwb_atcc)
- Package extra: `pip install 'moonshine-voice[finetune]'`, then `python -m moonshine_voice.lora --help`
- Deployment runtime: [github.com/moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)
""")


def build():
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4",
                      "name": "moonshine_lora_domain_adaptation.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        __import__("pathlib").Path(__file__).with_name(
            "moonshine_lora_domain_adaptation.ipynb"))
    json.dump(build(), open(out, "w"), indent=1)
    print(f"wrote {out}: {len(CELLS)} cells "
          f"({sum(1 for k, _ in CELLS if k == 'code')} code)")
