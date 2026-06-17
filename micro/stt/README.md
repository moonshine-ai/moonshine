# stt

On-device **speech-to-text for isolated letters and digits**: a TFLM wrapper
around the int8 mel-mode `SpellingCNN`. Given a normalised log-mel feature plane
(from the [`feature-generation`](../feature-generation) module) it runs the
int8 classifier with CMSIS-NN kernels and returns dequantized logits,
which the included helpers turn into a labelled prediction.

<!--TOC-->

- [Vocabulary](#vocabulary)
- [Public API](#public-api)
- [Memory & compute](#memory--compute)
  - [Latency @ 250 MHz](#latency--250-mhz)
- [Tests](#tests)
- [Generating data](#generating-data)

## Vocabulary

The checked-in SpellingCNN is a **36-way classifier** over isolated spoken
letters and digits. Class labels (from `example-rp2350/generated/classes.*`) are:

- **Letters (26):** `a`, `b`, `c`, …, `z`
- **Digits (10):** `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`

Each class is a single hyperarticulated token in a ~1 s window at 16 kHz. The
model supports isolated letters and digits only — not NATO/ICAO phonetic names,
spelled-out words, or continuous speech. Replacing the embedded `.tflite` and `classes.*` blobs (via
[`scripts/generate_embedded_data.py`](scripts/generate_embedded_data.py)) swaps
the label set, but flash and arena sizing must be revalidated for a different
architecture or class count.

Custom vocabulary models for other deployments are available commercially from
**Moonshine AI**.

## Public API

Single public header [`include/stt/stt.h`](include/stt/stt.h):

```cpp
spelling::Classifier clf(model, model_size, arena, arena_size,
                         n_mels, target_frames, n_classes);
float* feats = clf.feature_scratch();   // borrowed from the arena overlay
log_mel.Compute(waveform, n_samples, feats);
float logits[n_classes];
clf.Run(feats, logits);                 // quantize -> Invoke -> dequantize
int   pred = spelling::Argmax(logits, n_classes);
float prob = spelling::SoftmaxProb(logits, n_classes, pred);
```

The op set is locked to exactly what the exported model uses (`PAD`,
`DEPTHWISE_CONV_2D`, `CONV_2D`, `ADD`, `SUM`, `FULLY_CONNECTED`, `RESHAPE`); a
re-export with new ops fails loudly at `AllocateTensors()`.

## Memory & compute

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (model) | ~2.3 MiB | int8 SpellingCNN weights (`model_data.*`) |
| RAM (arena peak) | ~346 KiB | TFLM working set; app provisions 384 KiB |
| RAM (features) | 0 extra | fp32 log-mel written into idle arena overlay |
| Heap | 0 | interpreter + resolver placement-newed into arena head (~1 KiB) |

Feature generation and inference **share the same bytes** — there is no separate
feature buffer.

### Latency @ 250 MHz

| Operation | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| `Classifier::Run()` (dual-core) | ~535 ms per 1 s audio | ~52 MMAC per 1 s audio (~52 MMAC/s in) | CMSIS-NN int8 SIMD |
| `Classifier::Run()` (single-core) | ~877 ms per 1 s audio | ~52 MMAC per 1 s audio (~52 MMAC/s in) | same model, no core split |

MAC count is from the exported SpellingCNN graph structure (64×128 input). See
the top-level README for the full pipeline breakdown.

## Tests

`tests/predictor_test.cc` (TFLM `micro_test.h`) covers `Argmax` (incl. ties) and
the stable softmax. It runs on the host (helper logic only; the interpreter
wrapper is built for the target). `scripts/desktop_parity.py` reproduces the
on-device embedded-clip test loop on the desktop for regression checks.

## Generating data

`scripts/generate_embedded_data.py` reads the checked-in
`models/spelling_cnn_letters_digits_mel_int8.tflite` and its metadata sidecar,
then emits the example's embedded blobs (`model_data`, `classes`, `mel_tables`,
`audio_config`, `test_clips`):

```bash
python scripts/generate_embedded_data.py                 # 2 clips/class
python scripts/generate_embedded_data.py --clips-per-class 1
```

Output lands in `example-rp2350/generated/` by default. `scripts/desktop_parity.py`
reproduces the on-device run with `ai_edge_litert` and diffs per-clip
predictions against a captured `pico_monitor.log`.
