# feature-generation

A portable, **heap-free log-mel spectrogram front-end** for microcontrollers.
It turns 16 kHz PCM into the normalised log-mel features that the VAD and STT
models consume, using single-precision kissfft and a precomputed Slaney mel
filterbank baked into flash. The batch and streaming front-ends produce
bit-identical normalised log-mel features for the same audio.

The module is dependency-light on purpose — **kissfft** (real FFT) and **TFLM's
`micro_log`** (only for a fatal-misconfig `MicroPrintf`) — so it can be reused by
any front-end on any platform that provides those two.

<!--TOC-->

- [Two front-ends](#two-front-ends)
- [Public API](#public-api)
- [Memory & compute](#memory--compute)
  - [Latency @ 250 MHz](#latency--250-mhz)
- [Tests](#tests)
- [Generating flash tables](#generating-flash-tables)

## Two front-ends

| Class | Use | Cost |
| ----- | --- | ---- |
| `LogMelSpectrogram` | one-shot ("batch"): a whole clip → `(n_mels × target_frames)` plane | one FFT per output frame |
| `MelStreamer` | always-on streaming: slides one non-overlapping `n_fft` block per call | exactly **one FFT per hop** |

`MelStreamer` keeps the last `window_frames` log-mel columns in a ring and only
FFTs the newest block, so the streaming VAD front-end costs ~`window_frames`×
less FFT work than recomputing the whole window every hop. With `hop == n_fft`
(`center=false`) the two front-ends are bit-parity (see the unit test).

## Public API

Everything is in the single public header
[`include/feature_generation/feature_generation.h`](include/feature_generation/feature_generation.h):

```cpp
spelling::LogMelParams p{};            // n_fft, hop, n_mels, f_min/f_max, ...
p.precomputed_window = kMelWindow;     // flash tables (no heap, no boot trig)
p.precomputed_nz_off = kMelNzOff;      // CSR Slaney filterbank
// ...
spelling::LogMelSpectrogram lm(p);
lm.Compute(waveform, n_samples, out);  // out: n_mels * target_frames floats

spelling::MelStreamer s(n_mels, window_frames, n_fft, window,
                        nz_off, nz_idx, nz_val, fft);
s.PushHop(block);                      // one FFT
s.BuildModelInput(out);                // normalised (n_mels × window_frames)
```

The filterbank is stored only in compact **CSR** form (`nz_off/idx/val`) — the
dense `(n_mels × n_freq)` matrix is never materialised because that single
allocation (~64 KB at 64 mels) would overflow a small MCU heap.

## Memory & compute

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash | ~26 KiB | precomputed Hann window + CSR Slaney filterbank (`mel_tables.*`) |
| RAM (static) | ~5 KiB | shared `.bss` FFT scratch pool (`src/fft_scratch.h`), not on the stack |
| RAM (streamer) | ≤ 4 KiB | `MelStreamer` ring: `n_mels × window_frames` floats |
| Heap | 0 | deployed path uses flash tables only; host-test fallback allocates once |

The FFT scratch (`frame_buf[n_fft]` + `spectrum` + `power_row`) must live off the
4 KiB core stack — a stack-resident 512-pt frame would corrupt the concurrent
dual-core GEMM on the RP2350.

### Latency @ 250 MHz

Two deployment paths, both at 16 kHz:

| Path | Latency | Compute (approx.) | Notes |
| ---- | ------- | ----------------- | ----- |
| **VAD** — mel for 32 ms of audio | ~0.4 ms per 32 ms audio | ~12 KMAC per 32 ms audio (~0.4 MMAC/s in) | always-on listening; one 512-sample block |
| **STT** — mel for 1 s of audio | ~40 ms per 1 s audio | ~1.5 MMAC per 1 s audio (~1.5 MMAC/s in) | 64×128 log-mel plane after speech ends |

At 250 MHz the VAD path is sub-millisecond per 32 ms of input (dominated by one
real FFT). The STT path processes the full 1 s capture in one shot.

## Tests

`tests/feature_generation_test.cc` (TFLM `micro_test.h`) checks the Hann window,
the Slaney mel round-trip, and the streaming-vs-batch parity invariant. It runs
on the host (logic only, no interpreter).

## Generating flash tables

`scripts/generate_mel_tables.py` emits `mel_tables.{h,cc}` for a given front-end
config, model-independent:

```bash
python scripts/generate_mel_tables.py \
    --sample-rate 16000 --n-fft 512 --win-length 512 \
    --n-mels 64 --f-min 20 --f-max 8000 \
    --prefix kMel --const-prefix kMelTable --basename mel_tables \
    --out-dir ../example-rp2350/generated
```
