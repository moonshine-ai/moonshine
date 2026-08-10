# Available Models

Here are the models currently available. See [Downloading Models](../using/downloading-models.md) for how to obtain them. This library uses the Onnx model format, converted to the memory-mappable OnnxRuntime (`.ort`) flatbuffer encoding. For `safetensor` versions, see the [HuggingFace](huggingface.md) section.

| Language   | Architecture     | # Parameters | WER/CER |
| ---------- | ---------------- | ------------ | ------- |
| English    | Tiny             | 26 million   | 12.66%  |
| English    | Tiny Streaming   | 34 million   | 12.00%  |
| English    | Base             | 58 million   | 10.07%  |
| English    | Small Streaming  | 123 million  | 7.84%   |
| English    | Medium Streaming | 245 million  | 6.65%   |
| Arabic     | Base             | 58 million   | 5.63%   |
| Japanese   | Base             | 58 million   | 13.62%  |
| Korean     | Tiny             | 26 million   | 6.46%   |
| Mandarin   | Base             | 58 million   | 25.76%  |
| Spanish    | Base             | 58 million   | 4.33%   |
| Ukrainian  | Base             | 58 million   | 14.55%  |
| Vietnamese | Base             | 58 million   | 8.82%   |

The English evaluations were done using the [HuggingFace OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) datasets and methodology. The other languages were evaluated using the FLEURS dataset and the [`scripts/eval-model-accuracy`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/eval-model-accuracy.py) script, with the character or word error rate chosen per language.

Note that the English WER figures above are the Open ASR Leaderboard *average* across eight datasets, measured on the floating-point reference models. The quantized models this library actually ships score a little higher, especially at the Tiny size. See [Accuracy (Word Error Rate)](accuracy.md) below for a float-vs-quantized comparison and instructions on reproducing the numbers.

One common issue to watch out for if you're using models that don't use the Latin alphabet (so any languages except English and Spanish) is that you'll need to set the [`max_tokens_per_second` option](../api/classes.md#transcriber-options) to 13.0 when you create the transcriber. This is because the most common pattern for hallucinations is endlessly repeating the last few words, and our heuristic to detect this is to check if there's an unusually high number of tokens for the duration of a segment. Unfortunately the base number of tokens per second for non-Latin languages is much higher than for English, thanks to how we're tokenizing, so you have to manually set the threshold higher to avoid cutting off valid outputs.
