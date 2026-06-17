# vad

On-device **voice activity detection**: an int8 `TinyVadCNN` plus a moving-average smoother and 1 s clip
extractor. In the always-on path it runs once per ~32 ms hop, turning audio into
speech-segment boundaries that the STT classifier then transcribes.

The module is split into two reusable pieces:

| Class | Role |
| ----- | ---- |
| `Vad` | TFLM wrapper: `(n_mels × window_frames)` log-mel window → one speech probability in `[0, 1]` |
| `VadSegmenter` | smoothing + segmentation: per-frame probabilities → segment `[start, end)` sample indices, storing **no audio** |

The streaming log-mel front-end that feeds `Vad` is the
[`feature-generation`](../feature-generation) module's `MelStreamer` (one FFT per
hop). The application composes `MelStreamer → Vad → VadSegmenter`; see
`example-rp2350`'s audio path. Dependencies are only feature-generation and
TFLM, so the VAD can drop into a different example on another platform unchanged.

<!--TOC-->

- [Public API](#public-api)
- [Memory & compute](#memory--compute)
  - [Latency @ 250 MHz](#latency--250-mhz)
- [Tests](#tests)
- [Generating data](#generating-data)

## Public API

Single public header [`include/vad/vad.h`](include/vad/vad.h):

```cpp
spelling::Vad vad(model, model_size, arena, arena_size, n_mels, window_frames);
float* feats = vad.feature_scratch();
streamer.BuildModelInput(feats);
float p = vad.Predict(feats);            // speech probability

spelling::VadSegmenter seg(threshold, smooth_frames, hop,
                           look_behind_samples, max_segment_samples);
seg.Start();
if (seg.ProcessFrame(p) == spelling::VadEvent::kSpeechEnd) {
  spelling::ExtractClipFrontAligned(audio, n, seg.segment_start_sample(),
                                    seg.segment_end_sample(), clip, clip_len);
}
```

The `Vad` output logit is accepted as int8 (pure-int8 model) or int16 (the
mixed-precision int8-body/int16-head export); `Predict()` dequantizes both and
applies the sigmoid.

## Memory & compute

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (model) | 64 KiB | int8 TinyVadCNN weights (`vad_model_data.*`) |
| Flash (front-end) | ~25 KiB | VAD mel tables (`vad_mel_tables.*`) |
| RAM (arena peak) | ~36 KiB | TFLM working set; shares the STT arena sequentially |
| RAM (segmenter) | ~0.3 KiB | smoothing ring + counters; no audio buffer |
| Heap | 0 | no dynamic allocation |

VAD and STT never `Invoke()` at once, so the VAD arena use adds **no extra static
SRAM** beyond the shared 384 KiB tensor arena provisioned by the app.

### Latency @ 250 MHz

| Operation | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| `Vad::Predict()` | ~3.1 ms per 32 ms audio | ~0.8 MMAC per 32 ms audio (~25 MMAC/s in) | int8 TinyVadCNN `Invoke` |
| VAD step (mel + infer) | ~3.5 ms per 32 ms audio | ~0.8 MMAC per 32 ms audio (~25 MMAC/s in) | streaming mel + `Predict` |

One `Invoke()` per 32 ms of audio — ~11× headroom at 250 MHz. The segmenter is a
moving average (O(window)).

## Tests

`tests/vad_segmenter_test.cc` (TFLM `micro_test.h`) covers segment detection,
the look-behind pre-roll, trailing-segment flush, and front-aligned clip
extraction. It runs on the host (segmenter logic only; the interpreter wrapper
is built for the target).

## Generating data

`scripts/generate_vad_embedded_data.py` emits `vad_config.h`,
`vad_mel_tables.{h,cc}` (front-end, model-independent) and, given a model,
`vad_model_data.{h,cc}`:

```bash
# config + mel tables only (no model file needed)
python scripts/generate_vad_embedded_data.py --config-only

# also embed the checked-in int8 model
python scripts/generate_vad_embedded_data.py \
    --tflite ../models/tinyvad_cnn_speech_mel_head16.tflite
```

Output lands in `example-rp2350/generated/` by default.
