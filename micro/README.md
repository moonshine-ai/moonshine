![Moonshine Voice Logo](images/logo.png)

# Moonshine Micro — Voice Interfaces for Microcontrollers

[Moonshine Voice](https://github.com/moonshine-ai/moonshine) is an open source AI toolkit for developers building real-time voice agents and applications. Moonshine Micro is a version of that designed specifically for embedded system processors like microcontrollers and DSPs. It includes [voice-activity detection](vad/README.md), [command recognition](stt/README.md), and [speech synthesis](tts/README.md) and can run in as little as 500KB RAM.

The memory and compute requirements are designed to fit resource-constrained
systems. Figures below are for [the RP2350 demo](examples/rp2350/README.md); the detailed [memory budget](#memory-budget) breaks each one down:

| Component | Flash | SRAM (arena peak) | Compute |
|-----------|-------|-------------------|---------|
| VAD (Voice Activity Detection) | ~89 KiB | ~36 KiB | ~0.8 MMAC/frame (~25 MMAC/second) |
| STT (SpellingCNN Speech-to-Text) | ~1.3 MiB | ~346 KiB | ~36 MMAC/second |
| TTS (Klatt synth, waveform synthesis etc.) | ~150 KiB | few KiB | ~1 MMAC/second |
| **TOTAL (Demo pipeline)** | **~1.8 MiB** | **~498 KiB provisioned\*** | — |

*Notes:*
- *Flash is the `.text` measured with `arm-none-eabi-size` (model + code + rodata); SRAM is `.bss` + heap + stacks.*
- *\*The three stages run sequentially and time-share one ~384 KiB TFLM arena, so SRAM is not additive — ~498 KiB is the total RAM provisioned on the 520 KiB RP2350.*
- *A MAC is one multiply-accumulate; MMAC/s = millions per second during the active (non-idle) stage.*

The code is released under [the permissive MIT License](#license), usable for commercial applications.

There's a [complete end-to-end example](examples/rp2350/README.md) showing how to set up a wifi connection on a microcontroller using voice on an RP2350 MCU.

The VAD, STT, and TTS libraries can be used independently of each other, relying on the included [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) library for the neural computations.

## Documentation

 - [Voice Activity Detection](vad/README.md)
 - [Speech to Text](stt/README.md)
 - [Text to Speech](tts/README.md)
 - [Wifi Setup Example](examples/rp2350/README.md)

## License

This code, apart from the source in `third-party/`, is licensed under the MIT
License — see [LICENSE](LICENSE) in this directory (also at the repository root).

The SpellingCNN and TinyVadCNN models in [`models/`](models/) are released under
the MIT License.

The code in `third-party/` is licensed according to the terms of the open
source projects it originates from, with details in a LICENSE file in each
subfolder.
