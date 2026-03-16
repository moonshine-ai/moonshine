## Summary

Add optional word-level timestamps to the Moonshine transcription API. When enabled via `{"word_timestamps", "true"}`, each transcript line includes per-word `start`, `end`, and `confidence` values derived from the model's own cross-attention alignment.

Works with both non-streaming and streaming models. Addresses #145.

## Example output

```json
{
  "text": "Ever tried.",
  "start_time": 0.096,
  "duration": 1.376,
  "words": [
    {"word": "Ever", "start": 0.096, "end": 0.563, "confidence": 1.00},
    {"word": "tried.", "start": 0.563, "end": 1.423, "confidence": 1.00}
  ]
}
```

Full example from `beckett.wav` (10s, "Ever tried, ever failed, no matter, try again, fail again, fail better"):

```
[0.096s - 0.563s] Ever
[0.563s - 1.423s] tried.
[1.632s - 2.129s] Ever
[2.129s - 2.824s] failed.
[3.296s - 3.691s] No
[3.691s - 4.307s] matter.
[4.640s - 5.087s] Try
[5.087s - 5.806s] again.
[6.400s - 6.794s] Fair
[6.794s - 7.582s] again.
[7.904s - 8.441s] Fell
[8.441s - 9.102s] better.
```

## How it works

The Moonshine decoder already computes cross-attention weights (`softmax(Q·K^T/√d)`) at each layer during decoding. These weights aren't exposed as ONNX outputs in the standard model. We modify the ONNX graph to surface them by adding `Identity` nodes from the internal `Softmax` outputs to new graph outputs — same model, same weights, same computation.

During the autoregressive decode loop, the C++ code reads these 6 extra tensors per step, stacks them into a `[layers×heads, tokens, encoder_frames]` matrix, then applies Dynamic Time Warping (DTW) to find the monotonic alignment mapping each token to audio frames. Sub-word tokens are grouped into words using SentencePiece boundary markers.

A script (`scripts/export-decoder-with-attention.py`) generates the attention-enabled decoder from any existing model directory. It auto-detects streaming vs non-streaming format.

## Performance

Benchmarked on Apple M3, `two_cities_16k.wav` (44s, ~120 words):

### Speed

| Model | Without | With | Overhead |
|-------|---------|------|----------|
| Non-streaming (tiny) | 769ms | 903ms | +17% |
| Streaming (tiny) | 1030ms | 1179ms | +14% |

The overhead comes from reading 6 extra attention tensors per decode step and running DTW post-processing. The modified decoder model itself is the same size and per-step speed as the original.

### Memory (peak RSS)

| Model | Without | With | Delta |
|-------|---------|------|-------|
| Non-streaming (tiny) | 270 MB | 275 MB | +5 MB (+2%) |
| Streaming (tiny) | 349 MB | 376 MB | +27 MB (+8%) |

The extra memory is the accumulated attention buffer (~6 × heads × tokens × encoder_frames × 4 bytes per segment).

### Model size

| File | Original | With attention | Delta |
|------|----------|---------------|-------|
| `decoder_model_merged.ort` / `decoder_with_attention.ort` | 29.0 MB | 28.7 MB | -0.3 MB |
| `decoder_kv.ort` / `decoder_kv_with_attention.ort` | 30.9 MB | 30.9 MB | +0.0 MB |

### WER

No systematic change. The modified decoder uses identical weights and computation. In testing, 2 of 4 test cases produced byte-identical transcriptions. The 2 differences were single-word substitutions on words the model was already getting wrong ("Fair"→"Fare"), caused by floating-point rounding differences from the graph modification pushing a borderline token over the argmax threshold.

## API changes

New struct added to `moonshine-c-api.h`:

```c
struct transcript_word_t {
    const char *text;
    float start;
    float end;
    float confidence;
};
```

New fields on `transcript_line_t`:

```c
const struct transcript_word_t *words;  /* NULL if not enabled */
uint64_t word_count;                    /* 0 if not enabled */
```

Equivalent types added to the C++ wrapper (`WordTiming`), Python (`WordTiming` dataclass), Swift (`WordTiming` struct), Android/Java (`WordTiming` class), and JavaScript/TypeScript (`WordTiming` interface).

When `word_timestamps` is not enabled (the default), `words` is `NULL` and `word_count` is `0`. There is zero overhead — the standard decoder loads and runs unchanged.

## Test plan

- [x] `word-alignment-test` (doctest): loads model with `word_timestamps=true`, transcribes `beckett.wav`, verifies `word_count > 0` on all lines, checks monotonic ordering
- [x] `word-alignment-benchmark`: measures latency with/without timestamps for both streaming and non-streaming models
- [x] C++ builds clean with `-Wall -Wextra -pedantic -Werror`
- [x] `scripts/run-core-tests.sh` updated to include the new test
- [x] Manual testing: Python ctypes, Swift direct C interop, both produce correct word timestamps

## Files changed

**New files:**
- `core/word-alignment.h/.cpp` — DTW, median filter, token-to-word grouping
- `core/word-alignment-test.cpp` — doctest test
- `core/word-alignment-benchmark.cpp` — latency benchmark
- `scripts/export-decoder-with-attention.py` — generate attention-enabled decoder
- `test-assets/tiny-en/decoder_with_attention.ort` — non-streaming model (LFS)
- `test-assets/tiny-streaming-en/decoder_kv_with_attention.ort` — streaming model (LFS)
- `android/.../WordTiming.java` — Android data class
- `docs/word-level-timestamps.md` — feature documentation

**Modified files:**
- `core/moonshine-c-api.h/.cpp` — `transcript_word_t` struct, `word_timestamps` option
- `core/moonshine-model.h/.cpp` — attention collection in non-streaming decode loop
- `core/moonshine-streaming-model.h/.cpp` — attention collection in streaming decode loop
- `core/transcriber.h/.cpp` — load attention-enabled decoder, wire word timestamps
- `core/moonshine-cpp.h` — `WordTiming` struct in C++ wrapper
- `core/CMakeLists.txt` — build targets
- `python/src/moonshine_voice/moonshine_api.py` / `transcriber.py` — Python bindings
- `swift/Sources/MoonshineVoice/Transcript.swift` / `MoonshineAPI.swift` — Swift bindings
- `android/.../TranscriptLine.java`, `moonshine-jni.cpp` — Android bindings
- `scripts/run-core-tests.sh` — add word-alignment-test
