# Available Models

Here are the models currently available. See [Downloading Models](../using/downloading-models.md) for how to obtain them. This library uses the Onnx model format, converted to the memory-mappable OnnxRuntime (`.ort`) flatbuffer encoding. For `safetensor` versions, see the [HuggingFace](huggingface.md) section.

**Moonshine models are MIT by default, in every language and at every size.** Streaming speech-to-text is what each language with a streaming model now selects by default. The older non-streaming models for languages other than English stay under the non-commercial [Moonshine Community License](../license.md); all other STT models are under the MIT License. Where a streaming replacement exists they are deprecated and remain reachable only by naming the architecture.

- [Current models](#current-models)
- [Deprecated models](#deprecated-models)

English WER in the first table is the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) *average* across eight datasets, measured on the floating-point reference. Quantized English scores are a little higher, especially at Tiny; see [Accuracy](accuracy.md) for LibriSpeech-clean float-vs-quantized numbers. Non-English streaming scores are 400-clip macros on the panels named in [Accuracy](accuracy.md) (WER, or no-space CER for Japanese and Mandarin). Korean and Ukrainian have no streaming model yet, so their figures are the older full-FLEURS scores from `scripts/eval-model-accuracy.py` and are not comparable to the streaming rows.

## Current models

| Language   | Architecture     | # Parameters | WER/CER | License |
| ---------- | ---------------- | ------------ | ------- | ------- |
| English    | Medium Streaming | 245 million  | 6.65%   | MIT     |
| English    | Small Streaming  | 123 million  | 7.84%   | MIT     |
| English    | Tiny Streaming   | 34 million   | 12.00%  | MIT     |
| Arabic     | Tiny Streaming   | 34 million   | 15.5%   | MIT     |
| German     | Small Streaming  | 123 million  | 7.5%    | MIT     |
| German     | Tiny Streaming   | 34 million   | 12.0%   | MIT     |
| Japanese   | Small Streaming  | 123 million  | 17.2%†  | MIT     |
| Japanese   | Tiny Streaming   | 34 million   | 19.7%†  | MIT     |
| Mandarin   | Tiny Streaming   | 34 million   | 16.1%†  | MIT     |
| Spanish    | Small Streaming  | 123 million  | 4.9%    | MIT     |
| Spanish    | Tiny Streaming   | 34 million   | 6.2%    | MIT     |
| Tagalog    | Tiny Streaming   | 34 million   | 14.9%   | MIT     |
| Vietnamese | Tiny Streaming   | 34 million   | 9.4%    | MIT     |
| English    | Base             | 58 million   | 10.07%  | MIT     |
| English    | Tiny             | 26 million   | 12.66%  | MIT     |
| Korean     | Tiny             | 26 million   | 6.46%   | Community |
| Ukrainian  | Base             | 58 million   | 14.55%  | Community |

† No-space character error rate, not word error rate.

The first architecture listed for each language is the default `"ar"`, `"de"`,
`"en"` and so on will download. Korean and Ukrainian still default to their
non-streaming Community models because no streaming checkpoint is published for
them yet.

## Deprecated models

These Community-licensed non-streaming models have a streaming replacement in
the table above. They still load if you name the architecture, but new
integrations should use the streaming model. Their FLEURS scores come from
`scripts/eval-model-accuracy.py` (a character-weighted average on the full test
set) and are **not comparable** to the streaming 400-clip figures.

| Language   | Architecture | # Parameters | WER/CER | License   |
| ---------- | ------------ | ------------ | ------- | --------- |
| Arabic     | Base         | 58 million   | 5.63%   | Community |
| Japanese   | Base         | 58 million   | 13.62%  | Community |
| Japanese   | Tiny         | 26 million   | —       | Community |
| Mandarin   | Base         | 58 million   | 25.76%  | Community |
| Spanish    | Base         | 58 million   | 4.33%   | Community |
| Vietnamese | Base         | 58 million   | 8.82%   | Community |

Japanese Tiny has never been scored with `scripts/eval-model-accuracy`, so its
cell is left empty rather than filled from another harness.

One common issue to watch out for if you're using models that don't use the
Latin alphabet (so not English, Spanish, German, Vietnamese or Tagalog) is that
you'll need to set the [`max_tokens_per_second` option](../api/classes.md#transcriber-options)
to 13.0 when you create the transcriber. This is because the most common pattern
for hallucinations is endlessly repeating the last few words, and our heuristic
to detect this is to check if there's an unusually high number of tokens for the
duration of a segment. Unfortunately the base number of tokens per second for
non-Latin languages is much higher than for English, thanks to how we're
tokenizing, so you have to manually set the threshold higher to avoid cutting
off valid outputs.
