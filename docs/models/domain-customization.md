# Domain Customization

Transcription gets easier when you know something about what is about to be said, and usually you do: an application knows its own jargon, its product names, and the names in the user's contacts. There are two ways to tell the model what to expect. Runtime context is a list of key terms, or a passage of text to find them in, applied with no training step, which is what the rest of this section is about, and it works on words the model can already spell. Teaching it a new accent, dialect or acoustic environment needs [retraining](domain-customization.md#retraining) instead.

- [Runtime Context](#runtime-context)
    - [Supply a Context](#supply-a-context)
    - [Supply a Key Terms List](#supply-a-key-terms-list)
    - [How it works](#how-it-works)
    - [Tuning the strength](#tuning-the-strength)
    - [Measuring it on your own data](#measuring-it-on-your-own-data)
    - [What a long list costs](#what-a-long-list-costs)
    - [What it costs in time](#what-it-costs-in-time)
    - [What it can't do](#what-it-cant-do)
- [Retraining](#retraining)
    - [Worked example: air traffic control](#worked-example-air-traffic-control)
    - [Worked example: real VHF](#worked-example-real-vhf)
    - [Your own data](#your-own-data)
    - [Shipping the result](#shipping-the-result)
    - [Pitfalls](#pitfalls)

## Runtime Context

The most straightforward way to improve a model's accuracy for particular names or phrases for the application to supply hints. You can achieve up to a 40% reduction in errors with no latency cost and only a very small impact on general accuracy.

### Supply a Context

Often you have context without having a list. The user is dictating into a document, or looking at a ticket, or halfway through a thread, and the words worth listening for are already on screen — you just have not enumerated them. Hand over the text and they will be found for you:

```python
from moonshine_voice import Transcriber, ModelArch

transcriber = Transcriber(
    model_path,
    ModelArch.TINY_STREAMING,
    options={"context": open("migration-plan.md").read()},
)

# ...or follow the document as the user moves through it:
transcriber.set_context(current_page_text)
```

What gets picked is decided by the model's own tokenizer. That vocabulary is ordered by frequency, so an everyday word has a token to itself while jargon and proper nouns have to be spelled out of several subwords, and needing more than one is the signal used here. Given the passage

> Migration notes for the platform team. We will move the remaining services onto Kubernetes this quarter, with Ceph behind the storage classes and etcd holding the cluster state. Ask about the ingress before the meeting.

Tiny Streaming chooses `Migration`, `Kubernetes`, `Ceph`, `etcd` and `ingress`, and leaves every function word and every ordinary noun alone. Because the judgment comes from the tokenizer rather than from a word list we ship, it follows whichever language the loaded model was built for at no extra cost.

Terms are ranked by how often the passage says them, with the strangest-looking word winning a tie, and the list is then capped — 200 terms by default, or whatever you pass as `context_max_terms` at load time and as the second argument to `set_context()`. Keep the cap modest. As [What a long list costs](#what-a-long-list-costs) below shows, length is charged against every word you did *not* ask for, so the terms a passage leans on hardest are worth more than its whole long tail. Passing a book is fine; the cap is what keeps that from being a bad idea.

Everything else behaves like a key terms list, because that is what it becomes: it can be called while audio is streaming, takes effect on the next transcription, and needs a streaming architecture. Capitalization is taken from the passage, so a passage that writes "Kubernetes" is what makes the transcript write it that way too. The one thing to know is that only single words are proposed — a passage cannot tell us that "Anushka Sharma" is one name rather than two — and words containing digits are skipped, since a passage has far more dates and quantities in it than it has names like "IPv6". Name those outright with `keyterms` alongside the passage, and both sets are used.

The `context` and `context_max_terms` load options work anywhere transcriber options do. Replacing the passage on a running transcriber is wrapped in every binding — `set_context()` in Python, `setContext()` in Swift, Java and JavaScript, `moonshine_transcriber_set_context()` in C — and the max-terms argument takes 0 to mean the default of 200.

### Supply a Key Terms List

When you do know the words — a product catalog, a contact list, the phrases your own interface uses — name them. A list is more precise than a passage: it can carry multi-word terms, it spends no slots on words that happened to be nearby, and nothing is inferred. There's no training step, so the list can be different for every transcriber and can change while audio is streaming:

```python
from moonshine_voice import Transcriber, ModelArch

transcriber = Transcriber(
    model_path,
    ModelArch.TINY_STREAMING,
    options={"keyterms": "Kubernetes,Ceph,etcd"},
)

# ...or follow whatever the user is looking at, mid-stream:
transcriber.set_keyterms(["Anushka Sharma", "Jurgen Klopp"])
```

Match the capitalization and spelling you want in the output, since that's what gets produced. Terms can be phrases as well as words, and a list is a plain comma-separated string, so anywhere you can pass transcriber options you can pass key terms: `keyterms` and `keyterm_boost` in the Python options dictionary above, as `TranscriberOption` name/value pairs from Swift and Java, in the `options` record from JavaScript, or in the options array of `moonshine_load_transcriber_from_files()` / `moonshine_load_transcriber_from_memory_files()` in C. Replacing the terms on a transcriber that is already running is wrapped in every binding — `set_keyterms()` in Python, `setKeyterms()` in Swift, Java and JavaScript, `moonshine_transcriber_set_keyterms()` in C — and takes a list of terms rather than a joined string in the languages that have one. Only the streaming architectures support any of this; the older Tiny and Base models raise an error.

### How it works

Each term is tokenized and stored in a prefix tree over the model's subwords, and during decoding a bonus is added to the logits of the tokens that would continue one of those paths. A term spanning several subwords, like "Kubernetes", is therefore favored piece by piece rather than having to win in a single step. The bonus grows with how far into a term you already are, as `boost * (1 + ln(depth))`: cheap to start down a path, strongly rewarded to finish one. That ramp matters because greedy decoding cannot take back a token it has emitted, so a flat bonus would make a wrong first subword as attractive as a real completion, and terms would fire a syllable at a time on unrelated audio. The work per decoded token is one lookup per live path, bounded by the terms you passed rather than by the vocabulary, which is why the cost stays small.

### Tuning the strength

`keyterm_boost` defaults to 2.0, which removes about a quarter of the errors on the words you listed for at most a quarter of a point on everything else. Where to go from there is a genuine trade rather than a single best setting, and the table below is the shape of it: a test set built from LibriSpeech test-clean where each utterance's rare words are its key terms, padded to a hundred terms with rare words from elsewhere in the corpus so that most of what the decoder is told to listen for is not actually there. That leaves about one word in six of the corpus being a key term. Each cell is the word error rate on the listed words, then on every other word.

| Boost       | Tiny: terms / other | Small: terms / other | Medium: terms / other |
| ----------- | ------------------- | -------------------- | --------------------- |
| 0 (off)     | 13.02% / 6.84%      | 9.69% / 4.70%        | 7.89% / 3.90%         |
| 1           | 10.99% / 6.76%      | 8.27% / 4.74%        | 6.29% / 3.89%         |
| 2 (default) | 10.09% / 7.11%      | 7.06% / 4.85%        | 5.58% / 3.93%         |
| 3           | 8.92% / 7.47%       | 5.97% / 4.98%        | 5.07% / 4.28%         |
| 4           | 8.68% / 8.41%       | 5.71% / 5.47%        | 4.94% / 4.57%         |
| 6           | 15.80% / 17.05%     | 11.74% / 12.40%      | 14.05% / 11.07%       |

All three models agree, and the shape is what to take from it. The two halves move in opposite directions the whole way up, so the boost is a dial between them and not a setting with one right value. On the terms, every step up to 4 helps, though the gain flattens after 3. On everything else the first step is free — a boost of 1 is within noise of unbiased on all three models and slightly better than it on two — and then the cost climbs, more than doubling between 3 and 4 on Tiny and Small. Past 4 both halves fall apart together: at 6 the decoder is busy finishing terms nobody said, and false alarms run some thirty to forty times what they are at the default.

So 2.0 is the conservative default, buying about a quarter of the errors on your terms for a quarter of a point at worst elsewhere. Reach for 3.0 when the list matters more than the words around it: that is a third of the term errors, for between a quarter and six tenths of a point on everything else. Reach for 1.0 for the opposite trade, a sixth to a fifth of the term errors for no measurable cost at all. Do not go above 4.0, where this stops being a stronger version of the feature and becomes a broken one.

Those rows are samples of the corpus — 500 utterances, or 700 for Tiny — so read the shape rather than the third digit. Repeating Tiny's rows over all 2,620 utterances moved the numbers and not the trade: 12.23% / 6.92% unbiased, 8.75% / 7.11% at the default, and 7.85% / 8.48% at a boost of 4.0, the terms still improving and everything else still paying for it.

One caveat on all of it: these rows were measured with hundred-term lists, and length and strength multiply. If you are sending thousands of terms, stay at or below the default and read [What a long list costs](#what-a-long-list-costs) before raising it.

### Measuring it on your own data

The two halves above are the point of the harness, because they move in opposite directions and a single word error rate hides the trade. `scripts/make-keyterm-testset.py` builds a test set out of any ASR corpus without needing one recorded specially, taking each utterance's rare words as its key terms and padding every list with rare words from other utterances. `scripts/eval-keyterm-biasing.py` then sweeps the boost over it, reporting both halves along with term recall and a false-alarm rate:

<!-- doc-test: skip -->
```bash
python scripts/make-keyterm-testset.py --output-dir /tmp/keyterm-testset
python scripts/eval-keyterm-biasing.py \
    --manifest /tmp/keyterm-testset/manifest-100.jsonl \
    --boosts 0,1,2,4
```

It takes a manifest of your own recordings just as happily — `{"audio": ..., "text": ..., "keyterms": [...]}` per line, as in [`scripts/data/keyterm-eval-example.jsonl`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/data/keyterm-eval-example.jsonl) — and `--distractors-file` pads a short list out to the size you will really send. Do pad it: false alarms scale with how many terms are live, so a list of one term looks far cleaner than the hundred a real deployment passes.

### What a long list costs

List length costs accuracy separately from the boost, and it is worth knowing how much before shipping a big one. Here is Tiny Streaming on the whole of test-clean (2,620 utterances, 53,027 words) biased towards dictionary words the corpus never says, so nothing is won back on the terms and only the damage shows:

| Key terms | Boost 1.0 | Boost 2.0 (default) | Boost 4.0 |
| --------- | --------- | ------------------- | --------- |
| none      |           | 4.83%               |           |
| 1         |           | 4.84%               |           |
| 10        | 4.83%     | 4.86%               | 5.21%     |
| 100       | 4.91%     | 5.37%               | 10.22%    |
| 1,000     | 5.06%     | 5.88%               | 13.35%    |
| 10,000    | 5.17%     | 6.09%               | 13.66%    |

A single term is all but free, a hundredth of a point. At the default the growth stays gentle: a hundred terms nobody says cost half a point, a thousand cost a point, and ten thousand a point and a quarter, so lists in the thousands are usable if that is what your domain needs. A boost of 1.0 is gentler still, costing a third of a point even at ten thousand terms.

The 4.0 column is the one to take seriously, because it is where length and strength multiply. Ten terms are nearly as safe there as anywhere, but a hundred already cost five and a half points and a thousand cost eight and a half, arriving both in place of real words and on top of them: against the unbiased run, substitutions grew the most in absolute terms and insertions the fastest, nearly six-fold by ten thousand terms. So read [Tuning the strength](#tuning-the-strength) as being about a curated list. The case for a stronger boost is a case about a *short* list of terms you expect to hear, and it does not survive being pointed at thousands of terms that might not turn up. `scripts/eval-librispeech.py` takes `--keyterms-file` and `--keyterm-boost` if you want to repeat this against a general corpus rather than a domain one.

### What it costs in time

Almost nothing per utterance, so budget for this the same as an unbiased transcriber. These are average end-of-phrase latencies on a physical iPad (A16), each row against its own unbiased run of the same build:

| Key terms | Boost | Tiny        | Small        | Medium        |
| --------- | ----- | ----------- | ------------ | ------------- |
| 100       | 4.0   | 39 → 40ms   | 98 → 98ms    | 180 → 181ms   |
| 10,000    | 0     | +3-4ms      | +3-4ms       | +3-4ms        |
| 10,000    | 4.0   | 38 → 38ms   | 97 → 120ms   | 180 → 212ms   |

A hundred terms cost about a millisecond on Tiny and less than the run-to-run spread on the other two, and it makes no difference whether the terms are ever said: a list that never fires costs the same as one firing constantly. The middle row isolates the machinery from its effects by setting the boost to 0, which keeps every code path live while leaving the transcript byte-identical to an unbiased run — even ten thousand terms cost only a few milliseconds a phrase. The last row is what the biasing *taking effect* costs, and it was measured at the old default of 4.0: the false alarms a list that size produces at that boost keep revising the running hypothesis, and each revision is re-decoded, which also raised total decoding work by 15-25%. At the current default there are far fewer such revisions, so treat that row as an upper bound. A Pixel 10a agreed throughout: ten thousand terms cost under 10ms with the bonuses disabled and 30-60ms with them live at 4.0. Pass `--keyterms` to [`scripts/test-mobile-latency.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-mobile-latency.sh) to repeat any of it on your own hardware.

Installing a list is the one cost that scales with its length rather than being paid per token, because every term is tokenized as it goes in: 10,000 terms take about 30 milliseconds on a MacBook Pro, paid when you set them and not on every utterance. Short lists are unmeasurable. Swap terms mid-stream to follow the user's context as freely as you like; even a ten-thousand-term swap is well under a frame.

### What it can't do

Key terms only nudge the decoder towards words it can already spell, so they will not conjure a spelling the tokenizer cannot produce, and they cannot help with a new accent, dialect or recording environment — see [Retraining](domain-customization.md#retraining) below for that. The measurements here also come from single-word terms in read English speech, so if your terms are multi-word product names or unusual spellings, treat the numbers as a starting point and rerun the sweep on your own audio.

## Retraining

Runtime context helps with words the model can already spell. Teaching it a new accent, dialect, acoustic environment, or domain convention — air-traffic phraseology, spelled-out digits, and so on — needs a small amount of in-domain audio and a trained adapter. The default is a rank-8 LoRA on the decoder's self-attention q/k/v, about 0.11% of Streaming Medium, folded back into the base weights so inference is unchanged: no extra layers, no extra latency, and the same `Transcriber` load path.

What decoder LoRA buys is conventions, not vocabulary, and not a new microphone. Two hours of ATCOSIM taught Medium to write `four six five two` instead of `4652` and to keep callsign templates, and left waypoint names it had barely seen exactly where it found them. Those names are still [runtime key-term biasing](#supply-a-key-terms-list), and the two compose. ATCOSIM is close-talk headset in a quiet simulator, **not VHF radio**. Real radio is a second axis: see [Worked example: real VHF](#worked-example-real-vhf).

The trainer is an opt-in extra. A default `pip install moonshine-voice` does not pull in PyTorch or Transformers, and `import moonshine_voice` does not import this path. `[finetune]` and `[lora]` install the same packages.

<!-- doc-test: skip -->
```bash
pip install 'moonshine-voice[finetune]'
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moonshine-ai/moonshine/blob/main/examples/python/finetune/moonshine_lora_domain_adaptation.ipynb)

The [notebook](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/finetune/moonshine_lora_domain_adaptation.ipynb) is the measured walkthrough on ATCOSIM phraseology, with the same defaults as the command below. It calls the same helpers (`fit_adapter`, ATCOSIM loaders, export) as the CLI rather than inlining them. `moonshine-voice finetune` is an alias of `python -m moonshine_voice.lora`.

### Worked example: air traffic control

Needs a GPU. On the machine this was developed on (an RTX 5090) the two-hour Medium run finishes in about fifteen minutes and peaks under 3 GB of GPU memory, so it fits a Colab T4.

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora --dataset atcosim --output-dir ./lora_atc
```

That downloads the speaker-disjoint split from [`moonshine-ai/atcosim-speaker-disjoint-splits`](https://huggingface.co/datasets/moonshine-ai/atcosim-speaker-disjoint-splits), trains on two hours of ATCOSIM from speakers the test set never contains, mixes in 50% general-domain replay from [`moonshine-ai/yodas-en-replay`](https://huggingface.co/datasets/moonshine-ai/yodas-en-replay), and writes `adapter.safetensors` plus a merged `adapted/` checkpoint.

Measured by the notebook on the full 1,901-utterance test set and all of LibriSpeech test-clean:

| | air traffic control | LibriSpeech |
| --- | --- | --- |
| Moonshine Streaming Medium, as published | 50.7% WER | 2.3% WER |
| **+ LoRA, 2 h in-domain + replay** | **26.3% WER** | 2.8% WER |
| + LoRA, no replay | 26.0% WER | 3.8% WER |

Replay is on by default because the in-domain column cannot tell adaptation from forgetting. `--no-replay` is there so you can see that for yourself: it buys nothing in-domain at this size and makes ordinary speech worse.

ATCOSIM is free for research and commercial development but **not redistributable**. The Hub mirror this command reads is a convenience; for anything you ship, get the corpus from [TU Graz](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html). Our contribution is the split definition, which contains no audio.

Index the corpus without training:

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora --dataset atcosim --prepare-only
```

### Worked example: real VHF

ATCOSIM is not radio. A Butterworth band-limit of it does not transfer to held-out VHF; training on real VHF does. The public built-in corpus for that is [`Jzuluaga/uwb_atcc`](https://huggingface.co/datasets/Jzuluaga/uwb_atcc) (UWB-ATCC, Czech tower radio). `--dataset uwb_atcc` drops the one session the published train split shares with its test set (`TWR-34720N`). ATCO2-test-set-1h is a different airport's radio and is eval-only transfer — never a training source.

UWB-ATCC is **CC BY-NC-SA 4.0**. It is a research example. Do not train a commercial radio SKU on it. For a product adapter, pass your own VHF hours with `--train-manifest`.

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora --dataset uwb_atcc --sites both \
    --eval --eval-dataset atco2 --canary --output-dir ./lora_vhf
```

`--sites decoder` is still the default and is the right first bet for phraseology. `--sites both` is the cheap radio adapter: it also wraps encoder self-attention (default LR `1e-4`; decoder-only stays at `1e-3`). `--adapt full` unfreezes the backbone (default LR `1e-5`), writes no `adapter.safetensors`, and is the ceiling on real radio. Full fine-tune of Medium is not a Colab T4 job; re-export with `--graphs all`.

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora --dataset uwb_atcc --adapt full \
    --eval --eval-dataset atco2 --canary --output-dir ./ft_vhf
```

Measured on published Streaming Medium, two hours, 50% replay, session-disjoint UWB train, ATCO2-test-set-1h never in train:

| | UWB-ATCC (in-domain VHF) | ATCO2 (held-out VHF) | LibriSpeech |
| --- | --- | --- | --- |
| Moonshine Streaming Medium, as published | 107.8% WER | 82.2% WER | 2.3% WER |
| + ATCOSIM decoder LoRA (phraseology, not radio) | — | 74.5% WER | 3.0% WER |
| **+ UWB decoder LoRA** | **73.1% WER** | **59.6% WER** | **2.7% WER** |

Phraseology still transfers without radio (ATCOSIM adapter: ATCO2 82→75). Real VHF adds a second cut (ATCO2 75→60, in-domain 108→73) and the canary stays healthy. `--sites encoder|both` and `--adapt full` are in the CLI; on Medium they need a lower `--lr` than decoder-only and can NaN if you leave the decoder default of `1e-3`. The defaults above (`1e-4` / `1e-5`) are the starting point. A shippable radio SKU uses `--train-manifest` on hours you can use commercially, not this NC corpus.

### Your own data

A JSONL file, one clip per line:

```json
{"audio": "clips/001.wav", "text": "lufthansa four six five two"}
{"audio": "clips/002.wav", "text": "turn right heading two one zero"}
```

JSON `{"utterances": [...]}` and `path<TAB>text` TSV work too. Audio is any format `soundfile` reads; it is resampled to 16 kHz mono. Match transcript style to the model (cased, punctuated English) unless the domain's convention is what you want to teach — ATCOSIM's spelled-out digits are kept on purpose.

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora \
    --train-manifest domain.jsonl \
    --output-dir ./lora_out
```

Sensible defaults: rank 8, `--sites decoder`, learning rate `1e-3` (or `1e-4` when `--sites` includes the encoder, `1e-5` for `--adapt full`), batch 8, 50% replay from yodas-en-replay, early stopping on in-domain plus held-out general loss. `python -m moonshine_voice.lora --help` lists every flag. For real radio of your own, start with `--sites both`; reach for `--adapt full` when LoRA leaves accuracy on the table.

Score a held-out set and a LibriSpeech canary after training:

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora \
    --train-manifest domain.jsonl \
    --eval-manifest domain_test.jsonl \
    --eval --canary \
    --output-dir ./lora_out
```

### Shipping the result

<!-- doc-test: skip -->
```bash
python -m moonshine_voice.lora --export \
    --model ./lora_out/adapted --output-dir ./float \
    --tokenizer-bin /path/to/published/tokenizer.bin
bash scripts/quantize-streaming-model.sh ./float
```

A decoder-only adapter changes only `decoder_kv`, so `--graphs decoder_kv` plus the published `.ort` files for the other four graphs is enough. `--sites encoder|both` or `--adapt full` changes the encoder, so use `--graphs all`. Load the directory with `Transcriber` from the inference wheel — no `[finetune]` extra, no adapter code at runtime.

Then add the domain's word list with `set_keyterms` / `set_context`. The adapter has already learned how this domain talks; biasing is what recovers the names the training audio barely said.

### Pitfalls

- **Orthography mismatch silently eats the adapter.** An ALL-CAPS corpus teaches a typography the WER normalizer discards. `--text-mode auto` lowercases a corpus that is more than 90% uppercase.
- **ATCOSIM is not radio.** It is close-talk headset in a quiet simulator. Decoder LoRA on it teaches phraseology. A filtered copy of it does not become VHF. Real radio is `--dataset uwb_atcc` or your own `--train-manifest`.
- **UWB-ATCC is CC BY-NC-SA 4.0.** Research example only. A shippable radio SKU trains on hours you have a right to use commercially.
- **Do not train on ATCO2 mixes.** `jlvdoorn/atco2-asr` and similar overlap ATCO2-test-set-1h. `--eval-dataset atco2` scores that hour; it never trains on it.
- **Hold out whole speakers or sessions**, not random utterances, or you measure speaker adaptation and call it domain adaptation. `--dataset atcosim` and `--dataset uwb_atcc` already do.
- **Always score a general-domain canary.** In-domain WER alone cannot distinguish adaptation from damage.
- **Do not use the decoder LoRA learning rate on the encoder or a full fine-tune.** `--sites encoder|both` defaults to `1e-4` and `--adapt full` to `1e-5` because `1e-3` NaNs on Medium. If loss goes non-finite the trainer skips the step and keeps the last finite checkpoint.
- **`labels=` was double-shifted before Transformers 5.15.** The trainer computes cross-entropy itself against explicit `decoder_input_ids`, which is right on either version. The extra still pins `transformers>=5.15`.

