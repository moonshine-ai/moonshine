# Domain Customization

Transcription gets easier when you know something about what is about to be said, and usually you do: an application knows its own jargon, its product names, and the names in the user's contacts. There are two ways to tell the model what to expect. Key terms are a runtime list with no training step, which is what the rest of this section is about, and they work on words the model can already spell. Teaching it a new accent, dialect or acoustic environment needs [retraining](domain-customization.md#retraining) instead.

- [Key terms](#key-terms)
    - [Getting Started](#getting-started)
    - [How it works](#how-it-works)
    - [Tuning the strength](#tuning-the-strength)
    - [Measuring it on your own data](#measuring-it-on-your-own-data)
    - [What a long list costs](#what-a-long-list-costs)
    - [What it costs in time](#what-it-costs-in-time)
    - [What it can't do](#what-it-cant-do)
- [Retraining](#retraining)

## Key terms

### Getting Started

For individual words and phrases the model is unlikely to produce on its own — product names, jargon, the names in a user's contact list — you can pass a list of key terms at runtime. There's no training step, so the list can be different for every transcriber and can change while audio is streaming:

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

`keyterm_boost` defaults to 2.0, and that default is not a compromise between recall and accuracy — it is where the key terms themselves come out most accurately. Going higher recognizes fewer of them *and* damages the words around them. The table below is how it was chosen: a test set built from LibriSpeech test-clean where each utterance's rare words are its key terms, padded to a hundred terms with rare words from elsewhere in the corpus so that most of what the decoder is told to listen for is not actually there. That leaves about one word in six of the corpus being a key term. Each cell is the word error rate on the listed words, then on every other word.

| Boost       | Tiny: terms / other | Small: terms / other | Medium: terms / other |
| ----------- | ------------------- | -------------------- | --------------------- |
| 0 (off)     | 13.02% / 6.84%      | 9.69% / 4.70%        | 7.89% / 3.90%         |
| 1           | 11.89% / 6.81%      | 8.85% / 4.79%        | 7.06% / 3.90%         |
| 2 (default) | 11.18% / 6.95%      | 8.15% / 4.82%        | 6.48% / 3.89%         |
| 4           | 11.42% / 7.47%      | 8.21% / 5.11%        | 6.99% / 4.18%         |

All three models agree, and the shape is what to take from it. Biasing works: at the default it removes an eighth to a sixth of the errors on the words you listed, for about a tenth of a point on everything else. But the curve turns over, and above the default a stronger bonus is worse on both halves — the decoder commits to a term's first subword on the strength of the bonus alone and then has to finish a word nobody said. A boost above the default is not a more aggressive version of this feature, it is a broken one. Reach for 1.0 instead if general accuracy matters more to you than the list does.

Those rows are samples of the corpus — 500 utterances, or 700 for Tiny — so read the shape rather than the third digit. Repeating Tiny's rows over all 2,620 utterances moved the numbers and nothing else: 12.23% / 6.92% unbiased, 10.49% / 7.06% at the default, and 11.00% / 7.65% at a boost of 4.0, again worse on both.

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
| 1         |           | 4.83%               |           |
| 10        | 4.88%     | 5.00%               | 5.73%     |
| 100       | 4.95%     | 5.17%               | 7.11%     |
| 1,000     | 4.96%     | 5.53%               | 11.54%    |
| 10,000    | 5.05%     | 6.39%               | 13.24%    |

A single term is free — the transcript came back identical, error for error. At the default the growth is gentle: a hundred terms nobody says cost a third of a point, and ten thousand cost a point and a half, so lists in the thousands are usable if that is what your domain needs. The 4.0 column shows what the same lists did at the boost this library used to default to, where ten thousand terms cost eight and a half points, arriving both in place of real words and on top of them; substitutions grew the most in absolute terms and insertions the fastest, more than quadrupling. `scripts/eval-librispeech.py` takes `--keyterms-file` and `--keyterm-boost` if you want to repeat this against a general corpus rather than a domain one.

### What it costs in time

Almost nothing per utterance, so budget for this the same as an unbiased transcriber. These are average end-of-phrase latencies on a physical iPad (A16), each row against its own unbiased run of the same build:

| Key terms | Boost | Tiny        | Small        | Medium        |
| --------- | ----- | ----------- | ------------ | ------------- |
| 100       | 4.0   | 39 → 40ms   | 98 → 98ms    | 180 → 181ms   |
| 10,000    | 0     | +3-4ms      | +3-4ms       | +3-4ms        |
| 10,000    | 4.0   | 38 → 38ms   | 97 → 120ms   | 180 → 212ms   |

A hundred terms cost about a millisecond on Tiny and less than the run-to-run spread on the other two, and it makes no difference whether the terms are ever said: a list that never fires costs the same as one firing constantly. The middle row isolates the machinery from its effects by setting the boost to 0, which keeps every code path live while leaving the transcript byte-identical to an unbiased run — even ten thousand terms cost only a few milliseconds a phrase. The last row is what the biasing *taking effect* costs, and it was measured at the old default of 4.0: the false alarms a list that size produces at that boost keep revising the running hypothesis, and each revision is re-decoded, which also raised total decoding work by 15-25%. At the current default there are far fewer such revisions, so treat that row as an upper bound. A Pixel 10a agreed throughout: ten thousand terms cost under 10ms with the bonuses disabled and 30-60ms with them live at 4.0. Pass `--keyterms` to [`scripts/test-mobile-latency.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-mobile-latency.sh) to repeat any of it on your own hardware.

Installing a list is the one cost that scales with its length rather than being paid per token, because every term is tokenized as it goes in: 10,000 terms take about 0.85 seconds on a MacBook Pro and around a second on an iPad, paid when you set them and not on every utterance. Short lists are unmeasurable. If you swap terms mid-stream to follow the user's context, swap collections of tens or hundreds freely; a ten-thousand-term swap is a visible pause.

### What it can't do

Key terms only nudge the decoder towards words it can already spell, so they will not conjure a spelling the tokenizer cannot produce, and they cannot help with a new accent, dialect or recording environment — see [Retraining](domain-customization.md#retraining) below for that. The measurements here also come from single-word terms in read English speech, so if your terms are multi-word product names or unusual spellings, treat the numbers as a starting point and rerun the sweep on your own audio.

## Retraining

To teach a model a new accent, dialect or acoustic environment, rather than a new vocabulary, you'll need more comprehensive offline training. This is something we hope to add official support for in the future, but you can find a community project working on fine-tuning at [github.com/pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr).
