# RP2350 Moonshine Micro Example

This sample demonstrates **useful speech recognition and text-to-speech on a
low-end microcontroller** — a Raspberry Pi Pico 2 (RP2350, 520 KiB SRAM, 4 MiB
flash). It demonstrates how you can run a complete WiFi setup process on a resource-constrained device using a voice interface. The full pipeline — log-mel front-end, VAD, SpellingCNN, and Klatt synth — runs entirely on-device with no cloud connection.

<!--TOC-->

- [Quick Start](#quick-start)
- [What it does](#what-it-does)
- [Vocabulary](#vocabulary)
- [Memory budget](#memory-budget)
  - [Firmware totals](#firmware-totals)
  - [SRAM by component](#sram-by-component)
  - [Latency](#latency)
  - [Flash by component](#flash-by-component)
- [Components](#components)
- [Building the firmware](#building-the-firmware)
  - [Build options](#build-options)
  - [Voice-driven WiFi setup (Pico 2 W)](#voice-driven-wifi-setup-pico-2-w)
- [Host unit tests](#host-unit-tests)
- [Flashing & monitoring](#flashing--monitoring)
- [Kernels & cores](#kernels--cores)
- [Models](#models)
- [Regenerating embedded data](#regenerating-embedded-data)

## Quick Start

To make evaluating the example as easy as possible, you can connect your laptop's speaker and microphone to the board as virtual devices, instead of having to wire in physical components.

The default firmware **is** the live service: the laptop acts as the board's mic
and speaker via `usb_audio_bridge.py`, which streams the laptop microphone to the
device and plays the spoken reply back. Build, flash, then run the bridge:

```bash
cd moonshine-micro
export PICO_SDK_PATH="$HOME/projects/pico-sdk"
TOOLCHAIN="$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin"

cmake -B build -S . -DPICO_SDK_PATH="$PICO_SDK_PATH" -DPICO_TOOLCHAIN_PATH="$TOOLCHAIN"
cmake --build build -j 8
examples/rp2350/scripts/flash.sh echo

# Bridge the laptop mic/speaker (needs `pip install sounddevice`).
# Do NOT run monitor.sh at the same time — it holds the serial port.
python examples/rp2350/scripts/usb_audio_bridge.py
```

Speak a letter or digit; the bridge prints `VAD start` / `VAD end` /
`RESULT <label> <prob>` events and plays the synthesized reply. It is turn-based —
the host pauses sending while the device classifies and speaks, then resumes.

## What it does

The default app (`moonshine_micro_echo`) runs a streaming **VAD → STT → TTS**
loop forever: `MelStreamer` (one FFT per 32 ms hop) feeds the int8 VAD; when
speech ends the 1 s clip is classified by SpellingCNN; the recognized letter/digit
is spoken back via formant TTS using sound-alike words ("bee" for B, "hay" for A,
etc.).

WiFi setup is triggered by passing `wifi` as the argument to `flash.sh`, instead of `echo`. This application listens out for the trigger word 'wifi' and then runs the user through a conversation to pick a network, enter the password, and connect. You can also say 'ip' to get the local IP address if you are connected.

Flash the `test` variant (`moonshine_micro_echo_test`) instead to run the embedded-clip
accuracy sweep (development/CI path; adds ~1.1 MiB of test clips to flash).

## Vocabulary

The embedded SpellingCNN recognizes **51 isolated spoken tokens** — one letter,
digit, or command word per ~1 s utterance, not running speech or whole words:

| Category | Labels | Count |
| -------- | ------ | ----- |
| Letters | `a` … `z` | 26 |
| Digits | `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine` | 10 |
| Command / symbol words | `capital`, `uppercase`, `star`, `dollar`, `underscore`, `exclamation`, `percent`, `delete`, `done`, `cancel`, `wifi`, `ip`, `yes`, `no`, `hey rp` | 15 |

The default echo app uses the letters and digits; the command/symbol words drive
the [voice WiFi-setup flow](#voice-driven-wifi-setup-pico-2-w). NATO/ICAO names
("alpha", "bravo"), multi-character strings, and general open-vocabulary speech
are **out of scope** for this demo model. The firmware speaks letter results back
using sound-alike words ("bee" for `b`, "hay" for `a`, etc.); see
[`spelling_labels.h`](src/spelling_labels.h).

Custom vocabulary models — other token sets, product names, command words,
locale-specific letter names, and similar — are available commercially from
**Moonshine AI**.

## Memory budget

The RP2350 has 520 KiB SRAM and 4 MiB flash. Numbers below are from the default
live-audio firmware (the `moonshine_micro_echo` target, dual-core CMSIS-NN).

### Firmware totals

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (`.text`) | ~1.8 MiB | code + rodata + model (`arm-none-eabi-size`) |
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
| [examples/rp2350](./) | 64 KiB | — | `g_waveform[16000]` capture / STT clip buffer |
| [feature-generation](../../feature-generation/) | ~9 KiB | — | ~5 KiB shared FFT scratch pool + ~4 KiB `MelStreamer` ring |
| [vad](../../vad/) | — | ~36 KiB arena | `VadSegmenter` state (~0.3 KiB, on stack) |
| [stt](../../stt/) | 384 KiB arena | ~346 KiB arena | sized for SpellingCNN; fp32 features overlay in arena (0 extra) |
| [tts](../../tts/) | — | few KiB arena | reuses idle arena after STT |
| Platform (Pico SDK / USB) | ~8 KiB `.bss` | — | USB CDC, runtime globals |
| Platform (heap + stacks) | 24 KiB heap | >4 KiB core-0 stack | see stack note below |

**Stacks:** each core gets a 4 KiB scratch-bank stack (`PICO_STACK_SIZE=4096`), but
core 0 routinely needs **more than 4 KiB** during the deep STT `Invoke()` call chain.
The dual-core build links with [`memmap_dualcore_stack.ld`](memmap_dualcore_stack.ld),
which swaps the banks so core 0's overflow spills into unused main RAM (~20 KiB slack
below the scratch region) instead of corrupting core 1's live CMSIS-NN worker stack.
FFT working sets are kept in `.bss` for the same reason (see
[`feature-generation/src/fft_scratch.h`](../../feature-generation/src/fft_scratch.h)).

The log-mel window and Slaney filterbank live in flash; fp32 features are computed
into a slice of the (idle) arena overlay, so feature generation and inference share
the same bytes.

### Latency

Per-component latencies on the RP2350 at **250 MHz** (default dual-core firmware).
Compute figures are approximate MAC counts from the model/front-end structure (int8
MACs for the CNNs; float MACs for log-mel and Klatt synth), with inline input/output
rates in MMAC/s.

| Component | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| [feature-generation](../../feature-generation/) (VAD) | ~0.4 ms per 32 ms audio | ~12 KMAC per 32 ms audio (~0.4 MMAC/s) | always-on streaming mel |
| [feature-generation](../../feature-generation/) (STT) | ~40 ms per 1 s audio | ~1.5 MMAC per 1 s audio (~1.5 MMAC/s) | batch 64×128 log-mel plane |
| [vad](../../vad/) | ~3.1 ms per 32 ms audio | ~0.8 MMAC per 32 ms audio (~25 MMAC/s) | TinyVadCNN `Invoke` |
| [stt](../../stt/) | ~314 ms per 1 s audio | ~36 MMAC per 1 s audio (~36 MMAC/s) | SpellingCNN (dual-core CMSIS-NN) |
| [tts](../../tts/) | ~400–800 ms reply | ~0.5 MMAC typical (~1 MMAC/s out) | letter/digit at 22.05 kHz; longest ~1 s |
| **Classify + speak** | **~1.0 s typical** | **~38 MMAC in; ~0.5 MMAC out (~1 MMAC/s out)** | 1 s audio in + reply (excludes VAD) |

Single-core STT inference is ~507 ms per 1 s audio; see [Kernels & cores](#kernels--cores).

### Flash by component

| Component | Flash (typical) | README |
| --------- | --------------- | ------ |
| [feature-generation](../../feature-generation/) | ~26 KiB mel tables | [details](../../feature-generation/README.md#memory--compute) |
| [vad](../../vad/) | ~64 KiB model + ~25 KiB mel tables | [details](../../vad/README.md#memory--compute) |
| [stt](../../stt/) | ~1.3 MiB SpellingCNN | [details](../../stt/README.md#memory--compute) |
| [tts](../../tts/) | ~150 KiB G2P dict + code | [details](../../tts/README.md#memory--compute) |

## Components

The pipeline is split into reusable modules, each with its own CMake target, public
header, README, and unit tests. [`examples/rp2350/`](./) wires them
together for the Pico 2.

| Module | Role | README |
| ------ | ---- | ------ |
| [feature-generation](../../feature-generation/) | 16 kHz PCM → normalised log-mel features (batch + streaming) | [feature-generation/README.md](../../feature-generation/README.md) |
| [vad](../../vad/) | int8 voice-activity detection + segment boundaries | [vad/README.md](../../vad/README.md) |
| [stt](../../stt/) | int8 SpellingCNN — isolated letters, digits, and command words | [stt/README.md](../../stt/README.md) |
| [tts](../../tts/) | formant (Klatt-style) text-to-speech synth | [tts/README.md](../../tts/README.md) |

The TFLM dependency is decoupled behind a `tflm` CMake INTERFACE target, so modules
do not hard-code which TFLM implementation is linked.

## Building the firmware

Needs **Arm GNU toolchain 13.3.Rel1** (Homebrew's `arm-none-eabi-gcc` 16.x lacks
`nosys.specs`/`nano.specs`) and the **Pico SDK** at `~/projects/pico-sdk`.

```bash
cd moonshine-micro
export PICO_SDK_PATH="$HOME/projects/pico-sdk"
TOOLCHAIN="$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin"

cmake -B build -S . -DPICO_SDK_PATH="$PICO_SDK_PATH" -DPICO_TOOLCHAIN_PATH="$TOOLCHAIN"
cmake --build build -j 8
```

This builds one `.uf2` per app under `build/examples/rp2350/` (the first build
compiles TFLM + the SDK -- a couple of minutes; rebuilds are seconds):

| Target / artifact | App |
| ----------------- | --- |
| `moonshine_micro_echo.uf2` | live mic/speaker echo service (default) |
| `moonshine_micro_echo_test.uf2` | embedded-clip accuracy sweep |
| `moonshine_micro_echo_wifi.uf2` | voice WiFi setup (only with `-DPICO_BOARD=pico2_w`) |

Which app runs is chosen by which target you build and flash -- not by a
compile-time macro. Build one with `cmake --build build --target <name>`, or all
of them with `cmake --build build`. Each entry point is a tiny `src/main_*.cc`.

> **Troubleshooting — `cannot read spec file 'nosys.specs'` or missing headers.**
> CMake caches the compiler on first configure, so changing `PATH` later has no
> effect. Delete `build/` and reconfigure with `-DPICO_TOOLCHAIN_PATH` as above.
> Confirm with `grep CMAKE_C_COMPILER build/CMakeCache.txt` (should point inside
> the Arm toolchain, not `/opt/homebrew/bin`).

### Build options

These tune *how* targets are built (not which app runs -- that's the target you
flash):

| Option | Default | Effect |
| ------ | ------- | ------ |
| `PICO_BOARD` | `pico2` | Set to `pico2_w` to also build the WiFi target |
| `SPELLING_TINY_MULTICORE` | ON | Dual-core CMSIS-NN (~1.6× speedup, bit-identical) |
| `SPELLING_TINY_TTS` | ON | Add the TTS USB speak-loop demo to the **test** target |
| `SPELLING_TINY_VAD` | OFF | Add the VAD demo to the **test** target |
| `SPELLING_TINY_PROFILE_OPS` | OFF | Per-op timing in the **test** target |
| `SPELLING_TINY_PRINT_MEMORY_PLAN` | OFF | GreedyMemoryPlanner diagram on `AllocateTensors()` |

The live + WiFi targets always link the full VAD + STT + TTS pipeline regardless
of `SPELLING_TINY_VAD` / `SPELLING_TINY_TTS` (those only gate the test target's
optional demos).

### Voice-driven WiFi setup (Pico 2 W)

The `moonshine_micro_echo_wifi` target ([examples/rp2350/src/wifi_app.cc](src/wifi_app.cc))
reuses the same VAD → STT → TTS pipeline but, instead of echoing single letters,
walks a small state machine to join a network: say `wifi`, spell the SSID and
password character-by-character (letters, digit words, the symbol words
`star`/`dollar`/`underscore`/`exclamation`/`percent`, `capital` to upcase the
next letter, `delete`/`done`/`cancel`, and `yes`/`no` to confirm), and the
device associates over CYW43 and reads its DHCP address back when you say `ip`.

It is only created when the board has a CYW43 radio, so configure for the
Pico 2 W and build that target:

```bash
cmake -B build -S . -DPICO_SDK_PATH=$HOME/projects/pico-sdk \
  -DPICO_TOOLCHAIN_PATH=$HOME/projects/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi \
  -DPICO_BOARD=pico2_w
cmake --build build -j 8 --target moonshine_micro_echo_wifi
```

It links a minimized lwIP ([examples/rp2350/lwipopts.h](lwipopts.h):
poll mode, no TCP/DNS, small pools) tuned for a one-shot WPA2 join. To fit
alongside the radio stack the SpellingCNN arena is trimmed from 384 KiB to
360 KiB (still clears the ~346 KiB working set, via `SPELLING_TINY_ARENA_BYTES`);
the firmware links with ~14 KiB of SRAM headroom.

Audio still flows over the USB tether for the demo (the laptop is the mic +
speaker via [examples/rp2350/scripts/usb_audio_bridge.py](scripts/usb_audio_bridge.py))
while the Pico actually joins WiFi. For a standalone product, swap the
`UsbAudioInput`/`UsbAudioOutput` for an I2S mic + DAC behind
[examples/rp2350/src/audio_io.h](src/audio_io.h) — the recognition
and setup logic are unchanged. Only WPA2-PSK and the model's character set
(letters, digits, `* $ _ ! %`) are supported in v1.

## Host unit tests

Module logic is unit-tested on the host with TFLM's `micro_test.h` — no SDK or
device needed:

```bash
cmake -B build-host -S . -DMOONSHINE_MICRO_HOST_TESTS=ON
cmake --build build-host -j 8
ctest --test-dir build-host --output-on-failure
```

The interpreter wrappers (`stt::Classifier`, `vad::Vad`) are built for the target;
host tests cover the surrounding logic.

## Flashing & monitoring

```bash
examples/rp2350/scripts/flash.sh      # waits for BOOTSEL, copies the UF2
examples/rp2350/scripts/monitor.sh    # USB CDC serial monitor (tees a log)
```

The firmware uses USB CDC stdio (UART off). Run `monitor.sh` in one terminal and
`flash.sh` in another so the boot banner is not missed.

## Kernels & cores

The int8 classifier runs on **CMSIS-NN with the Cortex-M33 DSP SIMD path**
(`smlad` / `sxtb16`). The default dual-core build splits the SIMD GEMM (~74% of
inference) and the 3×3 depthwise conv (~19%) across both cores for ~1.6× at
250 MHz with bit-identical output. See
[`third-party/pico-tflmicro/PATCHES.md`](../../third-party/pico-tflmicro/PATCHES.md).

Embedded-clip sweep on the default dual-core firmware at 250 MHz:

| Build | infer | log-mel | avg latency (log-mel + infer) | accuracy |
| ----- | ----- | ------- | ----------------------------- | -------- |
| single-core | ~507 ms | ~40 ms | ~549 ms | 65/72 |
| dual-core | ~314 ms | ~40 ms | ~354 ms | 65/72 |

## Models

Canonical `.tflite` and `.onnx` exports for the two neural models live in
[`models/`](../../models/). See [`models/README.md`](../../models/README.md) for tensor
shapes, architecture notes, the pipeline diagram, and MMAC estimates.

| Model | Deployed artifact | Classes |
| ----- | ----------------- | ------- |
| SpellingCNN | `models/spelling_cnn_mel_int8.tflite` | 51 letters + digits + command words |
| TinyVadCNN | `models/tinyvad_cnn_speech_mel_head16.tflite` | speech / non-speech |

## Regenerating embedded data

The `generated/` blobs are checked in but reproducible:

```bash
python stt/scripts/generate_embedded_data.py            # -> examples/rp2350/generated/
python vad/scripts/generate_vad_embedded_data.py --config-only
python feature-generation/scripts/generate_mel_tables.py --help
```