# Neural Text to Speech

On-device **neural text-to-speech** at 16 kHz: a black-box synthesizer that turns
plain English (or IPA) into streaming mono int16 PCM, at [a much higher quality](#example-audio) than other TTS systems able to run on sub-$1 systems. Behind the public API it
runs the shared [`g2p`](../g2p/) front end, Klatt duration rules from
[`klatt-tts`](../klatt-tts/), diphone/word unit selection, RVQ decode through an
s16x8 TFLM graph, and a float32 WORLD-lite vocoder — all driven by one
flash-resident voice pack (`g_neural_tts_pack`).

## Example audio

| Phrase | |
| ------ | - |
| say wifi to set up a network | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/say_wifi.wav) |
| bee | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/bee.wav) |
| zero | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/zero.wav) |
| double u | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/double_u.wav) |
| You are connected | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/connected.wav) |

<!--TOC-->

- [Public API](#public-api)
- [Memory \& compute](#memory--compute)
- [Tests](#tests)
- [Generating data](#generating-data)

## Public API

Single public entry header [`include/neural_tts/neural_tts.h`](include/neural_tts/neural_tts.h):

```cpp
extern "C" const uint8_t g_neural_tts_pack[];  // flash pack (generated/)

neural_tts::NeuralTts tts(g_neural_tts_pack, arena, arena_size);
if (!tts.ok()) { /* pack corrupt or arena < kMinArenaBytes */ }

const int n = tts.EstimateSamples("bee");   // plan-only sample count
output.Begin(neural_tts::NeuralTts::kSampleRate, n);
tts.Synthesize("bee", EmitPcm, &sink);      // streams int16 chunks as rendered
output.End();

const neural_tts::NeuralTts::Stats& st = tts.stats();
// st.first_pcm_us, st.decode_us, st.render_us, st.tiles, ...
```

`Synthesize()` calls your `EmitFn` repeatedly with consecutive 16 kHz mono int16
chunks as the vocoder renders them — there is no full-utterance PCM buffer inside
the engine to minimize RAM usage. A minimal sink looks like:

```cpp
struct MySink {
  AudioOutput* out;   // I2S, USB CDC, DAC, ring buffer, ...
  int written = 0;
};

void EmitPcm(void* user, const int16_t* samples, int n) {
  auto* sink = static_cast<MySink*>(user);
  // `samples` is only valid for this call; copy if you queue asynchronously.
  sink->out->Write(samples, n);
  sink->written += n;
}

MySink sink{&output};
const int rc = tts.Synthesize("bee", EmitPcm, &sink);
```

The live apps copy through a small stack buffer and write in ≤256-sample slices
because `Write()` may block while a DMA ring drains (see `audio_service.cc` and
`main_i2s_audio_test.cc`). Feed the watchdog in long `Synthesize()` calls on
device builds.

`Synthesize()` and `SynthesizeIpa()` return the total samples emitted, or a
negative error code (`-1` G2P/plan failure, `-2` arena bump overflow,
`-3` vocoder/decode failure, `-4` chunk too long for the arena — the engine
retries with a smaller split). Long inputs are clause/chunk queued at silence
boundaries in `examples/rp2350`'s `Speak()`.

### Raw IPA for proper nouns

`Synthesize()` runs the on-device G2P rules (plus the baked dictionary) before
planning. That is enough for spelling readback and short prompts, but proper nouns
and other exceptions often need curated IPA. Pass the phone string directly with
`SynthesizeIpa()` — it skips the word front end and feeds `TokenizeIpa()` instead.

**Reading** (Pennsylvania) is a concrete example. The name is not in the baked
dictionary, so `Synthesize()` falls through to the rule engine, which reads the
spelling as `ɹˈiːdɪŋ` (like the verb, “REE-ding”). Locals say `ɹˈɛdɪŋ`
(“RED-ing”). Same graphemes, different phones — and `Synthesize()` takes the
wrong path here:

```cpp
// Default G2P path — rules misread this place name.
tts.Synthesize("Reading", EmitPcm, &sink);

// Curated IPA — bypasses G2P; use for place names, product names, etc.
tts.SynthesizeIpa("ɹˈɛdɪŋ", EmitPcm, &sink);
```

| Path | IPA | Listen |
| ---- | --- | ------ |
| `Synthesize("Reading")` (rules) | `ɹˈiːdɪŋ` | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/ipa/reading_g2p.wav) |
| `SynthesizeIpa("ɹˈɛdɪŋ")` | `ɹˈɛdɪŋ` | [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine-spelling@main/moonshine-micro/neural-tts/examples/ipa/reading_ipa.wav) |

Clips live under [`examples/ipa/`](examples/ipa/) (not the main demo table above).
Regenerate with:

```bash
cd moonshine-micro
PACK=examples/rp2350/generated/neural_tts_pack.bin
CLI=neural-tts/host/build/tts_cli
$CLI --pack "$PACK" Reading -o neural-tts/examples/ipa/reading_g2p.wav
$CLI --pack "$PACK" --ipa 'ɹˈɛdɪŋ' -o neural-tts/examples/ipa/reading_ipa.wav
```

Other common exceptions such as **Illinois** are already covered by the baked
dictionary; **Reading** shows the case where only `SynthesizeIpa()` or a
[`g2p::Lexicon`](../g2p/include/g2p/g2p_dict.h) override can fix the readback.
For a small recurring set at runtime, Lexicon (word → IPA TSV) is the other hook;
`SynthesizeIpa()` is simplest when you already have the phone string.

The remaining headers under `include/neural_tts/` (`pb_decoder.h`,
`worldlite_synth.h`, `pack_format.h`) are transitive types used by bring-up
and the engine; callers should include only `neural_tts.h`.

## Memory & compute

| Resource | Size | Notes |
| -------- | ---- | ----- |
| Flash (voice pack) | ~1.8 MiB | decoder .tflite, RVQ codebooks, diphone/word units, prosody tables (`neural_tts_pack.bin`) |
| Flash (code) | ~tens KiB | engine + vocoder linked into firmware `.text` |
| RAM (arena peak) | ~340 KiB | reuses idle STT arena; PbDecoder TFLM sub-arena ~144 KiB inside the bump |
| RAM (static) | ~0 extra | kissfft plans live in the arena bump during `Synthesize()` |
| Heap (transient) | few KiB | G2P `std::string` per word on the desktop path; PICO uses fixed token lists |

Recognition and synthesis never run at once, so neural TTS adds **no extra
static SRAM** beyond the shared tensor arena the app already provisions (~384 KiB
on the default RP2350 echo target; trimmed on WiFi/hardware variants — see
[`examples/rp2350/README.md`](../examples/rp2350/README.md#memory-budget)).

### Latency @ 250 MHz

| Operation | Latency | Compute (approx.) | Notes |
| --------- | ------- | ----------------- | ----- |
| `NeuralTts::Synthesize()` letter reply | ~0.4–0.7 s | ~37 MMAC typical (~65 MMAC/s out) | e.g. "bee" 0.37 s audio, 1 tile |
| `PbDecoder` tile `Invoke()` | ~0.1–0.3 s per tile | ~29 MMAC per tile | TL=32 latents → 128 frames (640 ms audio); dual-core CMSIS-NN GEMM |
| `WorldLiteSynth` render | interleaved with decode | ~8 MMAC per 0.37 s reply | float32 kissfft; ~1× real-time overall |

Wall time tracks reply audio length at roughly real-time synthesis speed.
`NeuralTts::Stats` breaks out G2P, planning, tiled decode, and vocoder time on
device (`PICO_BUILD`). See the [RP2350 example latency table](../examples/rp2350/README.md#latency)
for the full pipeline.

## Tests

There is no standalone `tests/*_test.cc` in this module yet. Validation instead
uses:

- **`worldlite_synth_cli`** (host, `MOONSHINE_MICRO_BUILD_TESTS=ON`) — raw
  `[T,61]` float32 controls on stdin → int16 PCM on stdout; driven by
  `scripts/test_worldlite_c.py` against pyworld.
- **`host/tts_cli`** — the full C++ pipeline on the desktop with the portable
  TFLM reference kernels; same bytes in/out as the `moonshine_micro_tts` firmware.
- **`scripts/hw_tts_eval/`** — captures on-device PCM over USB and scores
  synthesis quality against a host reference corpus.

The `examples/rp2350` bring-up ladder (`step6_decoder`, `step7_synthesize`, …)
exercises `PbDecoder` and `WorldLiteSynth` incrementally on hardware.

## Generating data

The voice pack is built at the repository root and embedded into the RP2350
example:

```bash
# 1. Export the s16x8 decoder graph (once per codec revision)
.venv/bin/python scripts/export_pb_decoder_litert.py \
    --codec data/pb_codec_s3d --fixup data/pb_fixup_s3d \
    --out data/pb_deploy

# 2. Bundle decoder + codebooks + unit inventory into one flash blob
.venv/bin/python scripts/export_neural_tts_pack.py \
    --deploy data/pb_deploy \
    --out moonshine-micro/examples/rp2350/generated
```

This writes `neural_tts_pack.bin`, `neural_tts_pack.S` (incbin for
`g_neural_tts_pack`), and `neural_tts_pack_report.json`. Re-export
`scripts/export_pb_decoder_litert.py` when the RVQ codec or fixup filter
changes; bump `kNeuralTtsPackVersion` in
[`include/neural_tts/pack_format.h`](include/neural_tts/pack_format.h) whenever
the pack layout changes.
