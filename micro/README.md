![Moonshine Voice Logo](images/logo.png)

# Moonshine Micro — Voice Interfaces for Microcontrollers

[Moonshine Voice](https://github.com/moonshine-ai/moonshine) is an open source AI toolkit for developers building real-time voice agents and applications. Moonshine Micro is a version designed specifically for embedded system processors like microcontrollers and DSPs, and uses the Raspberry Pi RP2350, which retails for just 80 cents, as its reference platform. It includes [voice-activity detection](vad/README.md), [command recognition](stt/README.md), and [neural speech synthesis](neural-tts/README.md) and can run in as little as 470 KB of RAM.

You can see a full walkthrough in the video below:

[![Moonshine Micro demo video](video-thumbnail.gif)](https://www.youtube.com/watch?v=kMliOFYBiz4)

The memory and compute requirements are designed to fit resource-constrained
systems. Figures below are for [the RP2350 demo](examples/rp2350/README.md); the
detailed [memory budget](examples/rp2350/README.md#memory-budget) breaks each one down:

| Component | Flash | SRAM (arena peak) | Compute |
|-----------|-------|-------------------|---------|
| VAD (Voice Activity Detection) | ~89 KiB | ~36 KiB | ~0.8 MMAC/frame (~25 MMAC/s) |
| STT (SpellingCNN Speech-to-Text) | ~1.3 MiB | ~346 KiB | ~36 MMAC/s |
| TTS (neural diphone synth @ 16 kHz) | ~1.8 MiB voice pack | ~340 KiB | ~37 MMAC typical reply (~65 MMAC/s out) |
| **TOTAL (Demo pipeline)** | **~3.6 MiB** | **~468 KiB provisioned\*** | classify + speak ~0.7–1.0 s |

*Notes:*
- *Flash is `.text` + `.rodata` measured with `arm-none-eabi-size` on the default `moonshine_micro_echo` firmware (includes the embedded neural voice pack); SRAM is `.bss` + heap + stacks.*
- *\*VAD, STT, and neural TTS run sequentially and time-share one ~384 KiB TFLM arena, so SRAM is not additive — ~468 KiB is the total RAM provisioned on the 520 KiB RP2350 (`wifi_hardware` ~491 KiB).*
- *A MAC is one multiply-accumulate; MMAC/s = millions per second during the active (non-idle) stage.*

The code is released under [the permissive MIT License](#license), suitable for commercial applications.

There's a [complete end-to-end example](examples/rp2350/README.md) showing how to set up a wifi connection on a microcontroller using voice on an RP2350 MCU.

The VAD, STT, and TTS libraries can be used independently of each other, relying on the included [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) library for the neural computations.

## Documentation

 - [Voice Activity Detection](vad/README.md)
 - [Speech to Text](stt/README.md)
 - [Neural Text to Speech](neural-tts/README.md)
 - [Wifi Setup Example](examples/rp2350/README.md)

## License

This code, apart from the source in `third-party/`, is licensed under the MIT
License — see [LICENSE](LICENSE) in this directory (also at the repository root).

The SpellingCNN and TinyVadCNN models in [`models/`](models/) are released under
the MIT License.

The code in `third-party/` is licensed according to the terms of the open
source projects it originates from, with details in a LICENSE file in each
subfolder.
