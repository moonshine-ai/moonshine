# Accuracy (Word Error Rate)

Beyond knowing which models are available, you'll often want to understand how
accurate they are and how to reproduce the numbers yourself. The
[`scripts/eval-librispeech.py`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/eval-librispeech.py) script measures Word
Error Rate (WER) on the LibriSpeech `test-clean` set using the same dataset and
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
methodology (corpus-level WER with the Whisper English text normalizer) reported
in our [Moonshine v2 paper](https://arxiv.org/abs/2602.12241).

- [Reproducing these numbers](#reproducing-these-numbers)
- [Takeaways](#takeaways)

There's an important subtlety here that can be confusing: **the WER numbers
in the paper were measured with the floating-point models running in the Hugging
Face Transformers library, not the quantized models this framework ships.** As the
paper notes in section 4.1.2, we use the Transformers implementation to measure
accuracy and our own C++/ONNX library to measure latency. The models you download
here are 8-bit quantized `.ort` files chosen for on-device speed and size.

The table below shows LibriSpeech `test-clean` WER for all three streaming models,
comparing the paper's floating-point reference against the quantized models this
library ships. All numbers use whole-utterance (non-streaming) transcription with
the VAD disabled, so they're a like-for-like comparison of raw model accuracy.

| Model            | Paper (float) | Reproduced float (HF Transformers) | Shipped quantized model (this library) |
| ---------------- | ------------- | ---------------------------------- | -------------------------------------- |
| Tiny Streaming   | 4.49%         | 4.52%                              | 4.83%                                  |
| Small Streaming  | 2.49%         | 2.55%                              | 2.61%                                  |
| Medium Streaming | 2.08%         | 2.16%                              | 2.17%                                  |

Every shipped model is now within 0.31% WER of its floating-point reference. That
was not always true: models published before 2026-07-30 quantized each weight
tensor with a single scale factor, which cost Tiny Streaming 7.57% instead of
4.83%. Switching to per-channel weight scales fixed it, for 0.5% more model size.
See [Quantization](quantization.md) for the details.

## Reproducing these numbers

The evaluation script downloads the dataset and models for you. Install the
dependencies and run it:

<!-- doc-test: skip -->
```bash
# Core dependencies for evaluating the shipped (quantized) models.
pip install moonshine-voice datasets soundfile scipy jiwer openai-whisper

# Evaluate a shipped quantized model on LibriSpeech test-clean (VAD disabled).
python scripts/eval-librispeech.py --backend moonshine_c --model-arch tiny_streaming
python scripts/eval-librispeech.py --backend moonshine_c --model-arch small_streaming
python scripts/eval-librispeech.py --backend moonshine_c --model-arch medium_streaming
```

To reproduce the paper's floating-point reference numbers you also need a recent
version of Transformers (the streaming models were added in Transformers 5.x) and
PyTorch, then pass `--backend hf`:

<!-- doc-test: skip -->
```bash
pip install "transformers>=5.13" torch
python scripts/eval-librispeech.py --backend hf --model-arch tiny_streaming
```

The script disables the VAD by default (`vad_threshold=0` plus a very large
`vad_max_segment_duration`) because the LibriSpeech clips are already single
utterances, so any VAD segmentation only adds errors. Pass `--enable-vad` to see
the effect of the segmenter, or `--backend moonshine_c_streaming` to measure the
chunked, real-time streaming path instead of whole-utterance transcription. Use
`--limit N` for a quick smoke test on the first `N` clips.

## Takeaways

- **8-bit quantization now costs very little accuracy at any size.** The penalty
  against the floating-point reference is +0.31% WER on Tiny, +0.06% on Small and
  +0.01% on Medium. There is no longer much reason to run the floating-point
  checkpoint in Transformers just for accuracy, even on Tiny.
- **Disable the VAD when evaluating pre-segmented data.** On already-segmented
  clips like LibriSpeech, leaving the default VAD enabled adds roughly +1.5–2%
  WER on Tiny (mostly extra insertions at segment boundaries). The VAD is there to
  chop up continuous live audio, not clean single utterances.
- **Watch which number you're comparing against.** The per-model WER in the
  [Available Models](available-models.md) table is the Open ASR Leaderboard *average*
  across eight datasets (12.00% for Tiny Streaming, 7.84% for Small Streaming, and
  6.65% for Medium Streaming), which is much higher than the LibriSpeech-clean-only
  numbers above. Make sure you're comparing the same benchmark.
