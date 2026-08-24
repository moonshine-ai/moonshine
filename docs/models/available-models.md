# Available Models

Here are the models currently available. See [Downloading Models](../using/downloading-models.md) for how to obtain them. This library uses the Onnx model format, converted to the memory-mappable OnnxRuntime (`.ort`) flatbuffer encoding. For `safetensor` versions, see the [HuggingFace](huggingface.md) section.

| Language   | Architecture     | # Parameters | WER/CER | License   |
| ---------- | ---------------- | ------------ | ------- | --------- |
| English    | Tiny             | 26 million   | 12.66%  | MIT       |
| English    | Tiny Streaming   | 34 million   | 12.00%  | MIT       |
| English    | Base             | 58 million   | 10.07%  | MIT       |
| English    | Small Streaming  | 123 million  | 7.84%   | MIT       |
| English    | Medium Streaming | 245 million  | 6.65%   | MIT       |
| Arabic     | Base             | 58 million   | 5.63%   | Community |
| Japanese   | Base             | 58 million   | 13.62%  | Community |
| Korean     | Tiny             | 26 million   | 6.46%   | Community |
| Mandarin   | Base             | 58 million   | 25.76%  | Community |
| Spanish    | Base             | 58 million   | 4.33%   | Community |
| Ukrainian  | Base             | 58 million   | 14.55%  | Community |
| Vietnamese | Base             | 58 million   | 8.82%   | Community |

Streaming speech-to-text is also published for Arabic, German, Japanese, Mandarin, Spanish, Tagalog and Vietnamese — Tiny in all seven, and Small in German, Japanese and Spanish. **All streaming models are released under the MIT License, in every language**, unlike the non-streaming models above for languages other than English, which are under the non-commercial [Moonshine Community License](../license.md). Streaming is what each of these languages now selects by default; the non-streaming models stay reachable by naming the architecture.

Their accuracy is not listed in the table above because it has not yet been measured with `scripts/eval-model-accuracy`, and quoting figures from a different harness in the same column would make the rows look comparable when they are not. Per-language numbers, including the cost of quantization, are on each model's [HuggingFace](huggingface.md) card.

The English evaluations were done using the [HuggingFace OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) datasets and methodology. The other languages were evaluated using the FLEURS dataset and the [`scripts/eval-model-accuracy`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/eval-model-accuracy.py) script, with the character or word error rate chosen per language.

Note that the English WER figures above are the Open ASR Leaderboard *average* across eight datasets, measured on the floating-point reference models. The quantized models this library actually ships score a little higher, especially at the Tiny size. See [Accuracy (Word Error Rate)](accuracy.md) below for a float-vs-quantized comparison and instructions on reproducing the numbers.

One common issue to watch out for if you're using models that don't use the Latin alphabet (so any languages except English and Spanish) is that you'll need to set the [`max_tokens_per_second` option](../api/classes.md#transcriber-options) to 13.0 when you create the transcriber. This is because the most common pattern for hallucinations is endlessly repeating the last few words, and our heuristic to detect this is to check if there's an unusually high number of tokens for the duration of a segment. Unfortunately the base number of tokens per second for non-Latin languages is much higher than for English, thanks to how we're tokenizing, so you have to manually set the threshold higher to avoid cutting off valid outputs.
