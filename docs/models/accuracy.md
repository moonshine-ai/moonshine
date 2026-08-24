# Accuracy (Word Error Rate)

Beyond knowing which models are available, you'll often want to understand how
accurate they are and how to reproduce the numbers yourself. The
[`scripts/eval-librispeech.py`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/eval-librispeech.py)
script measures corpus-level error rate (total edits over total reference units)
with the VAD disabled, so the figure is the model's transcription accuracy rather
than the live segmenter. English uses LibriSpeech `test-clean` and the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
methodology from the [Moonshine v2 paper](https://arxiv.org/abs/2602.12241).
Other languages use a seeded 400-clip sample of the public sets named in the
table — a quick check, not a leaderboard submission. This script is not part of
`scripts/test-*.sh` or the release build.

- [English (LibriSpeech)](#english-librispeech)
- [Other languages](#other-languages)
- [Reproducing these numbers](#reproducing-these-numbers)
- [Takeaways](#takeaways)

There's an important subtlety here that can be confusing: **the English WER
numbers in the paper were measured with the floating-point models running in the
Hugging Face Transformers library, not the quantized models this framework
ships.** As the paper notes in section 4.1.2, we use the Transformers
implementation to measure accuracy and our own C++/ONNX library to measure
latency. The models you download here are 8-bit quantized `.ort` files chosen
for on-device speed and size. Non-English streaming figures below are already
the quantized models.

## English (LibriSpeech)

LibriSpeech `test-clean` WER for the three English streaming models, comparing
the paper's floating-point reference against the quantized models this library
ships. All numbers use whole-utterance (non-streaming) transcription with the
VAD disabled.

| Model            | Paper (float) | Reproduced float (HF Transformers) | Shipped quantized model (this library) |
| ---------------- | ------------- | ---------------------------------- | -------------------------------------- |
| Tiny Streaming   | 4.49%         | 4.52%                              | 4.83%                                  |
| Small Streaming  | 2.49%         | 2.55%                              | 2.61%                                  |
| Medium Streaming | 2.08%         | 2.16%                              | 2.17%                                  |

Every shipped English streaming model is now within 0.31% WER of its
floating-point reference. That was not always true: models published before
2026-07-30 quantized each weight tensor with a single scale factor, which cost
Tiny Streaming 7.57% instead of 4.83%. Switching to per-channel weight scales
fixed it, for 0.5% more model size. See [Quantization](quantization.md) for the
details.

The per-model WER in the [Available Models](available-models.md) table is the
Open ASR Leaderboard *average* across eight datasets (12.00% Tiny Streaming,
7.84% Small Streaming, 6.65% Medium Streaming), which is much higher than the
LibriSpeech-clean-only numbers above.

## Other languages

These are the quantized streaming models this library ships, scored as deployed
(batch 1, VAD off) on a seeded 400-clip sample. Japanese and Mandarin use
no-space CER because those writing systems do not mark word boundaries; the
other rows are WER. Korean Tiny and Ukrainian Base have no streaming successor
yet, so their figures stay the older full-FLEURS scores from
`scripts/eval-model-accuracy.py` and are not comparable to the 400-clip rows.

| Language   | Model            | Panel                                      | Metric | Quantized |
| ---------- | ---------------- | ------------------------------------------ | ------ | --------- |
| Arabic     | Tiny Streaming   | Common Voice + FLEURS                      | WER    | 15.5%     |
| German     | Small Streaming  | FLEURS + MLS                               | WER    | 7.5%      |
| German     | Tiny Streaming   | FLEURS + MLS                               | WER    | 12.0%     |
| Spanish    | Small Streaming  | FLEURS + MLS                               | WER    | 4.9%      |
| Spanish    | Tiny Streaming   | FLEURS + MLS                               | WER    | 6.2%      |
| Japanese   | Small Streaming  | FLEURS + ReazonSpeech                      | CER    | 17.2%     |
| Japanese   | Tiny Streaming   | FLEURS + ReazonSpeech                      | CER    | 19.7%     |
| Mandarin   | Tiny Streaming   | FLEURS + WenetSpeech                       | CER    | 16.1%     |
| Tagalog    | Tiny Streaming   | FLEURS                                     | WER    | 14.9%     |
| Vietnamese | Tiny Streaming   | FLEURS + LSVSC                             | WER    | 9.4%      |
| Korean     | Tiny             | FLEURS (full, `eval-model-accuracy`)       | WER    | 6.46%     |
| Ukrainian  | Base             | FLEURS (full, `eval-model-accuracy`)       | WER    | 14.55%    |

Tagalog is a snapshot of a run that had not finished training when it was
taken, and it is the only language whose panel is a single read-speech set.

Do not compare these rows to the English LibriSpeech table, or to the deprecated
non-streaming Community models on [Available Models](available-models.md): those
used a different dataset mix and a character-weighted average.

## Reproducing these numbers

The evaluation script downloads the dataset and models for you. It is a manual
accuracy check, not a CI or release step. Install the dependencies and run it:

<!-- doc-test: skip -->
```bash
# Core dependencies for evaluating the shipped (quantized) models.
pip install moonshine-voice datasets soundfile scipy jiwer openai-whisper

# English streaming models on LibriSpeech test-clean (VAD disabled).
python scripts/eval-librispeech.py --backend moonshine_c --model-arch tiny_streaming
python scripts/eval-librispeech.py --backend moonshine_c --model-arch small_streaming
python scripts/eval-librispeech.py --backend moonshine_c --model-arch medium_streaming

# Quick FLEURS check for another language (400 seeded clips).
python scripts/eval-librispeech.py --language ar --model-arch tiny_streaming --quick
python scripts/eval-librispeech.py --language de --model-arch small_streaming --quick
python scripts/eval-librispeech.py --language ja --model-arch tiny_streaming --quick

# Every published streaming model, 400 clips each (LibriSpeech for English,
# FLEURS otherwise). This will not match the two-set shipping panels above.
python scripts/eval-librispeech.py --all-streaming --quick
```

`--quick` is a FLEURS-only (or LibriSpeech-only) 400-clip slice, so it will not
reproduce the two-set macros in the table. It is the right tool for "did this
checkpoint still decode Arabic" rather than for updating the published number.

To reproduce the paper's floating-point English reference numbers you also need
a recent version of Transformers (the streaming models were added in
Transformers 5.x) and PyTorch, then pass `--backend hf`:

<!-- doc-test: skip -->
```bash
pip install "transformers>=5.13" torch
python scripts/eval-librispeech.py --backend hf --model-arch tiny_streaming
```

The script disables the VAD by default (`vad_threshold=0` plus a very large
`vad_max_segment_duration`) because the clips are already single utterances, so
any VAD segmentation only adds errors. Pass `--enable-vad` to see the effect of
the segmenter, or `--backend moonshine_c_streaming` to measure the chunked,
real-time streaming path instead of whole-utterance transcription. Use
`--limit N` for a still-shorter smoke test.

## Takeaways

- **8-bit quantization now costs very little accuracy at any English size.** The
  penalty against the floating-point reference is +0.31% WER on Tiny, +0.06% on
  Small and +0.01% on Medium. There is no longer much reason to run the
  floating-point checkpoint in Transformers just for accuracy, even on Tiny.
- **Disable the VAD when evaluating pre-segmented data.** On already-segmented
  clips like LibriSpeech or FLEURS, leaving the default VAD enabled adds roughly
  +1.5–2% WER on Tiny (mostly extra insertions at segment boundaries). The VAD is
  there to chop up continuous live audio, not clean single utterances.
- **Watch which number you're comparing against.** English Open ASR averages,
  English LibriSpeech-clean, and the non-English 400-clip panels are three
  different measurements. Streaming is what each language with a streaming model
  now selects by default; the older Community non-streaming models that have a
  streaming replacement are deprecated.
- **Non-Latin tokenizers need a higher hallucination guard.** Set
  `max_tokens_per_second` to 13 for Arabic, Japanese, Korean, Mandarin and
  Ukrainian. The script does this automatically from `--language`.
