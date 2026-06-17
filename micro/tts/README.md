# tts

A portable, dependency-free **formant (Klatt-style) text-to-speech** synth core.
The goal is *"robotic but understandable"* speech that fits the RP2350's resource
envelope. On-device it speaks back the recognized letter/digit after STT.

The core is C++17, **float-only math, no allocation in the audio inner loop, and
no third-party dependency** (not even kissfft or TFLM).

<!--TOC-->

- [Public API](#public-api)
- [Memory & compute](#memory--compute)
  - [Latency @ 250 MHz](#latency--250-mhz)
- [Tests](#tests)
- [Tuning](#tuning)

## Public API

Single public entry header [`include/tts/tts.h`](include/tts/tts.h) — include
only this:

```cpp
tts::VoiceParams voice = tts::DefaultVoiceParams();
tts::StreamSynth synth(voice, arena, arena_size);   // caller-supplied arena
tts::StreamOptions opts; opts.sample_rate = 22050.0f;
synth.BeginText("hello world", opts);               // on-device G2P
// or synth.BeginIpa("h\u0259lo\u028a", opts);      // raw IPA, bypass G2P
float buf[256];
for (int n; (n = synth.Read(buf, 256)) > 0; )       // streams; never buffers
  emit(buf, n);                                       // ... the whole utterance
```

The remaining headers under `include/tts/` (`synth_stream.h`, `config.h`,
`klatt.h`, `synth_internal.h`, `phonemes.h`, `g2p_dict.h`) are transitive types
that `StreamSynth` exposes by value; the G2P internals are private in `src/`.

## Memory & compute

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (dictionary) | ~150 KiB | baked common-word G2P table (`src/g2p_dict_data.h`) |
| Flash (code) | ~tens KiB | Klatt synth + G2P rules (linked into firmware `.text`) |
| RAM (static) | ~0 extra | per-frame tracks live in the caller-supplied arena |
| RAM (arena peak) | few KiB | reuses idle TFLM tensor arena after STT finishes |
| Heap (transient) | few KiB | tokenizer / segment list in `BeginText` only; none in `Read()` |

`Read()` renders one frame at a time; the full utterance PCM is never buffered.
Recognition and synthesis never run at once, so the synth adds essentially **no
static SRAM** beyond the shared arena the app already provisions.

### Latency @ 250 MHz

| Operation | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| `StreamSynth` letter/digit | ~400–800 ms reply | ~0.5 MMAC typical (~1 MMAC/s out) | e.g. "bee" ~430 ms, "zero" ~830 ms @ 22.05 kHz |
| `Read()` inner loop | faster than real-time | ~5 KMAC per 5 ms frame (~1 MMAC/s out) | six biquad resonators × ~110 samples/frame |

Measured output lengths with `DefaultVoiceParams()` at 22.05 kHz: "ee" ~320 ms,
"bee" ~430 ms, "seven" ~760 ms, "double u" ~1.0 s. `Read()` renders one frame at
a time; the full utterance PCM is never buffered. Recognition and synthesis never
run at once.

## Tests

`tests/tts_test.cc` (TFLM `micro_test.h`) covers the G2P front-end (text → phone
tokens, number normalization) and an end-to-end `StreamSynth` smoke test
(non-empty, in-range PCM). It runs entirely on the host.

## Tuning

Voice parameters ship as compiled-in defaults in `DefaultVoiceParams()`
(`src/config.cc`). Optional `.cfg` files can override them at runtime on the
desktop build for listening tests; the firmware uses the defaults only.
