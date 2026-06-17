![Moonshine Voice Logo](images/logo.png)

# Moonshine Micro — On-device speech recognition and text to speech for microcontrollers

Build useful voice interfaces in less than 500KB of RAM.

The example demonstrates **useful speech recognition and text-to-speech on a
low-end microcontroller** — a Raspberry Pi Pico 2 (RP2350, 520 KiB SRAM, 4 MiB
flash). Speak a letter or digit, the board detects speech,
classifies it with an int8 CNN, and speaks the result back through synthesized TTS.
The full pipeline — log-mel front-end, VAD, SpellingCNN, and Klatt synth — runs
entirely on-device with no cloud connection.

<!--TOC-->

- [Run the live service with host audio](#run-the-live-service-with-host-audio)
- [What it does](#what-it-does)
- [Vocabulary](#vocabulary)
- [Memory budget](#memory-budget)
- [Components](#components)
- [Building the firmware](#building-the-firmware)
- [Host unit tests](#host-unit-tests)
- [Flashing \& monitoring](#flashing--monitoring)
- [Kernels \& cores](#kernels--cores)
- [Models](#models)
- [Regenerating embedded data](#regenerating-embedded-data)
- [License](#license)

## Run the live service with host audio

The default firmware **is** the live service: the laptop acts as the board's mic
and speaker via `usb_audio_bridge.py`, which streams the laptop microphone to the
device and plays the spoken reply back. Build, flash, then run the bridge:

```bash
cd micro
export PICO_SDK_PATH="$HOME/projects/pico-sdk"
TOOLCHAIN="$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin"

pip install -r requirements.txt

cmake -B build -S . -DPICO_SDK_PATH="$PICO_SDK_PATH" -DPICO_TOOLCHAIN_PATH="$TOOLCHAIN"
cmake --build build -j 8
example-rp2350/scripts/flash.sh

# Bridge the laptop mic/speaker (needs `pip install sounddevice`).
# Do NOT run monitor.sh at the same time — it holds the serial port.
python example-rp2350/scripts/usb_audio_bridge.py
```

Speak a letter or digit; the bridge prints `VAD start` / `VAD end` /
`RESULT <label> <prob>` events and plays the synthesized reply. It is turn-based —
the host pauses sending while the device classifies and speaks, then resumes.

## What it does

On boot the default path runs a streaming **VAD → STT → TTS** loop forever:
`MelStreamer` (one FFT per 32 ms hop) feeds the int8 VAD; when speech ends the
1 s clip is classified by SpellingCNN; the recognized letter/digit is spoken back
via formant TTS using sound-alike words ("bee" for B, "hay" for A, etc.).

Build with `-DSPELLING_TINY_TEST_SWEEP=ON` to run the embedded-clip accuracy sweep
instead (development/CI path; adds ~1.1 MiB of test clips to flash).

## Vocabulary

The embedded SpellingCNN recognizes **36 isolated spoken tokens** — one letter or
digit per ~1 s utterance, not running speech or whole words:

| Category | Labels | Count |
| -------- | ------ | ----- |
| Letters | `a` … `z` | 26 |
| Digits | `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine` | 10 |

NATO/ICAO names ("alpha", "bravo"), multi-character strings, and general
open-vocabulary speech are **out of scope** for this demo model. The firmware
speaks results back using sound-alike words for letters ("bee" for `b`, "hay" for
`a`, etc.); see [`spelling_labels.h`](example-rp2350/src/spelling_labels.h).

Custom vocabulary models — other token sets, product names, command words,
locale-specific letter names, and similar — are available commercially from
**Moonshine AI**.

## Memory budget

The RP2350 has 520 KiB SRAM and 4 MiB flash. Numbers below are from the default
live-audio firmware (`SPELLING_TINY_AUDIO=ON`, dual-core CMSIS-NN).

### Firmware totals

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (`.text`) | ~2.8 MiB | code + rodata (`arm-none-eabi-size`) |
| Static RAM (`.bss`) | ~466 KiB | module + app buffers (see below) |
| Heap (`PICO_HEAP_SIZE`) | 24 KiB | kissfft twiddles, newlib/USB startup |
| Stack banks | 8 KiB | 2 × 4 KiB scratch-bank stacks (see below) |
| **Total provisioned** | **~498 KiB / 520 KiB** | main RAM + scratch SRAM on the RP2350 |

### SRAM by component

Static sizes are from the default live-audio firmware ELF. The 384 KiB TFLM arena
is provisioned once and reused sequentially by VAD → STT → TTS; peak arena use
during each phase is listed in the transient column.

| Component | Static SRAM | Transient / peak | Notes |
| --------- | ----------- | ---------------- | ----- |
| [example-rp2350](example-rp2350/) | 64 KiB | — | `g_waveform[16000]` capture / STT clip buffer |
| [feature-generation](feature-generation/) | ~9 KiB | — | ~5 KiB shared FFT scratch pool + ~4 KiB `MelStreamer` ring |
| [vad](vad/) | — | ~36 KiB arena | `VadSegmenter` state (~0.3 KiB, on stack) |
| [stt](stt/) | 384 KiB arena | ~346 KiB arena | sized for SpellingCNN; fp32 features overlay in arena (0 extra) |
| [tts](tts/) | — | few KiB arena | reuses idle arena after STT |
| Platform (Pico SDK / USB) | ~8 KiB `.bss` | — | USB CDC, runtime globals |
| Platform (heap + stacks) | 24 KiB heap | >4 KiB core-0 stack | see stack note below |

**Stacks:** each core gets a 4 KiB scratch-bank stack (`PICO_STACK_SIZE=4096`), but
core 0 routinely needs **more than 4 KiB** during the deep STT `Invoke()` call chain.
The dual-core build links with [`memmap_dualcore_stack.ld`](example-rp2350/memmap_dualcore_stack.ld),
which swaps the banks so core 0's overflow spills into unused main RAM (~20 KiB slack
below the scratch region) instead of corrupting core 1's live CMSIS-NN worker stack.
FFT working sets are kept in `.bss` for the same reason (see
[`feature-generation/src/fft_scratch.h`](feature-generation/src/fft_scratch.h)).

The log-mel window and Slaney filterbank live in flash; fp32 features are computed
into a slice of the (idle) arena overlay, so feature generation and inference share
the same bytes.

### Latency @ 250 MHz

Per-component latencies on the RP2350 at **250 MHz** (default dual-core firmware).
Compute figures are approximate MAC counts from the model/front-end structure (int8
MACs for the CNNs; float MACs for log-mel and Klatt synth), with inline input/output
rates in MMAC/s.

| Component | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| [feature-generation](feature-generation/) (VAD) | ~0.4 ms per 32 ms audio | ~12 KMAC per 32 ms audio (~0.4 MMAC/s) | always-on streaming mel |
| [feature-generation](feature-generation/) (STT) | ~40 ms per 1 s audio | ~1.5 MMAC per 1 s audio (~1.5 MMAC/s) | batch 64×128 log-mel plane |
| [vad](vad/) | ~3.1 ms per 32 ms audio | ~0.8 MMAC per 32 ms audio (~25 MMAC/s) | TinyVadCNN `Invoke` |
| [stt](stt/) | ~535 ms per 1 s audio | ~52 MMAC per 1 s audio (~52 MMAC/s) | SpellingCNN (dual-core CMSIS-NN) |
| [tts](tts/) | ~400–800 ms reply | ~0.5 MMAC typical (~1 MMAC/s out) | letter/digit at 22.05 kHz; longest ~1 s |
| **Classify + speak** | **~1.0 s typical** | **~54 MMAC in; ~0.5 MMAC out (~1 MMAC/s out)** | 1 s audio in + reply (excludes VAD) |

Single-core STT inference is ~877 ms per 1 s audio; see [Kernels & cores](#kernels--cores).

### Flash by component

| Component | Flash (typical) | README |
| --------- | --------------- | ------ |
| [feature-generation](feature-generation/) | ~26 KiB mel tables | [details](feature-generation/README.md#memory--compute) |
| [vad](vad/) | ~64 KiB model + ~25 KiB mel tables | [details](vad/README.md#memory--compute) |
| [stt](stt/) | ~2.3 MiB SpellingCNN | [details](stt/README.md#memory--compute) |
| [tts](tts/) | ~150 KiB G2P dict + code | [details](tts/README.md#memory--compute) |

## Components

The pipeline is split into reusable modules, each with its own CMake target, public
header, README, and unit tests. [`example-rp2350/`](example-rp2350/) wires them
together for the Pico 2.

| Module | Role | README |
| ------ | ---- | ------ |
| [feature-generation](feature-generation/) | 16 kHz PCM → normalised log-mel features (batch + streaming) | [feature-generation/README.md](feature-generation/README.md) |
| [vad](vad/) | int8 voice-activity detection + segment boundaries | [vad/README.md](vad/README.md) |
| [stt](stt/) | int8 SpellingCNN — isolated letters and digits | [stt/README.md](stt/README.md) |
| [tts](tts/) | formant (Klatt-style) text-to-speech synth | [tts/README.md](tts/README.md) |

The TFLM dependency is decoupled behind a `tflm` CMake INTERFACE target, so modules
do not hard-code which TFLM implementation is linked.

## Building the firmware

Needs **Arm GNU toolchain 13.3.Rel1** (Homebrew's `arm-none-eabi-gcc` 16.x lacks
`nosys.specs`/`nano.specs`) and the **Pico SDK** at `~/projects/pico-sdk`.

```bash
cd cpp-tiny
export PICO_SDK_PATH="$HOME/projects/pico-sdk"
TOOLCHAIN="$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin"

cmake -B build -S . -DPICO_SDK_PATH="$PICO_SDK_PATH" -DPICO_TOOLCHAIN_PATH="$TOOLCHAIN"
cmake --build build -j 8
```

The artifact is `build/example-rp2350/predict_spelling_tiny.uf2`. The first build
compiles TFLM + the SDK (a couple of minutes); rebuilds are seconds.

> **Troubleshooting — `cannot read spec file 'nosys.specs'` or missing headers.**
> CMake caches the compiler on first configure, so changing `PATH` later has no
> effect. Delete `build/` and reconfigure with `-DPICO_TOOLCHAIN_PATH` as above.
> Confirm with `grep CMAKE_C_COMPILER build/CMakeCache.txt` (should point inside
> the Arm toolchain, not `/opt/homebrew/bin`).

### Build options

| Option | Default | Effect |
| ------ | ------- | ------ |
| `SPELLING_TINY_AUDIO` | ON | Live USB mic/speaker service (forces VAD + TTS on) |
| `SPELLING_TINY_TEST_SWEEP` | OFF | Embedded-clip accuracy sweep instead of live service |
| `SPELLING_TINY_MULTICORE` | ON | Dual-core CMSIS-NN (~1.6× speedup, bit-identical) |
| `SPELLING_TINY_TTS` | ON | Formant TTS engine |
| `SPELLING_TINY_VAD` | ON | On-device VAD (forced on in live service) |
| `SPELLING_TINY_PROFILE_OPS` | OFF | Per-op timing (test-sweep build only) |
| `SPELLING_TINY_PRINT_MEMORY_PLAN` | OFF | GreedyMemoryPlanner diagram on `AllocateTensors()` |

## Host unit tests

Module logic is unit-tested on the host with TFLM's `micro_test.h` — no SDK or
device needed:

```bash
cmake -B build-host -S . -DCPP_TINY_HOST_TESTS=ON
cmake --build build-host -j 8
ctest --test-dir build-host --output-on-failure
```

The interpreter wrappers (`stt::Classifier`, `vad::Vad`) are built for the target;
host tests cover the surrounding logic.

## Flashing & monitoring

```bash
example-rp2350/scripts/flash.sh      # waits for BOOTSEL, copies the UF2
example-rp2350/scripts/monitor.sh    # USB CDC serial monitor (tees a log)
```

The firmware uses USB CDC stdio (UART off). Run `monitor.sh` in one terminal and
`flash.sh` in another so the boot banner is not missed.

## Kernels & cores

The int8 classifier runs on **CMSIS-NN with the Cortex-M33 DSP SIMD path**
(`smlad` / `sxtb16`). The default dual-core build splits the SIMD GEMM (~74% of
inference) and the 3×3 depthwise conv (~19%) across both cores for ~1.6× at
250 MHz with bit-identical output. See
[`third-party/pico-tflmicro/PATCHES.md`](third-party/pico-tflmicro/PATCHES.md).

Embedded-clip sweep on the default dual-core firmware at 250 MHz:

| Build | infer | log-mel | avg latency (log-mel + infer) | accuracy |
| ----- | ----- | ------- | ----------------------------- | -------- |
| single-core | ~877 ms | ~40 ms | ~917 ms | 30/36 |
| dual-core | ~535 ms | ~40 ms | ~575 ms | 30/36 |

## Models

Canonical `.tflite` and `.onnx` exports for the two neural models live in
[`models/`](models/). See [`models/README.md`](models/README.md) for tensor
shapes, architecture notes, the pipeline diagram, and MMAC estimates.

| Model | Deployed artifact | Classes |
| ----- | ----------------- | ------- |
| SpellingCNN | `models/spelling_cnn_letters_digits_mel_int8.tflite` | 36 letters + digits |
| TinyVadCNN | `models/tinyvad_cnn_speech_mel_head16.tflite` | speech / non-speech |

## Regenerating embedded data

The `generated/` blobs are checked in but reproducible:

```bash
python stt/scripts/generate_embedded_data.py            # -> example-rp2350/generated/
python vad/scripts/generate_vad_embedded_data.py --config-only
python feature-generation/scripts/generate_mel_tables.py --help
```

## License

This code, apart from the source in `third-party/`, is licensed under the MIT
License — see [LICENSE](LICENSE) in this directory (also at the repository root).

The SpellingCNN and TinyVadCNN models in [`models/`](models/) are released under
the MIT License.

The code in `third-party/` is licensed according to the terms of the open
source projects it originates from, with details in a LICENSE file in each
subfolder.
