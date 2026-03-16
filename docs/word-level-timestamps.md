# Word-Level Timestamps for Moonshine ASR

## Overview

Moonshine now supports optional word-level timestamps on transcription output. When enabled, each transcript line includes an array of words with `start`, `end`, and `confidence` values derived from the model's own cross-attention alignment — no external alignment model or second pass required.

Works with both non-streaming and streaming models.

```json
{
  "text": "And God said, Let there be light, and there was light.",
  "start_time": 3.045,
  "duration": 3.535,
  "words": [
    {"word": "And",    "start": 3.057, "end": 3.395, "confidence": 0.73},
    {"word": "God",    "start": 3.395, "end": 3.758, "confidence": 0.96},
    {"word": "said,",  "start": 3.758, "end": 4.025, "confidence": 0.86},
    {"word": "Let",    "start": 4.025, "end": 4.169, "confidence": 0.68},
    {"word": "there",  "start": 4.169, "end": 4.315, "confidence": 0.99},
    {"word": "be",     "start": 4.315, "end": 4.871, "confidence": 1.00},
    {"word": "light,", "start": 4.871, "end": 5.258, "confidence": 0.88},
    {"word": "and",    "start": 5.258, "end": 5.402, "confidence": 0.96},
    {"word": "there",  "start": 5.402, "end": 5.548, "confidence": 1.00},
    {"word": "was",    "start": 5.548, "end": 6.030, "confidence": 1.00},
    {"word": "light.", "start": 6.030, "end": 6.575, "confidence": 0.97}
  ]
}
```

## Enabling Word Timestamps

Pass `word_timestamps=true` as a transcriber option. When disabled (the default), transcript lines have `words=NULL` / `word_count=0` and there is zero overhead.

### C API

```c
transcriber_option_t options[] = {
    { "word_timestamps", "true" },
};
int32_t handle = moonshine_load_transcriber_from_files(
    model_path, MOONSHINE_MODEL_ARCH_TINY, // or MOONSHINE_MODEL_ARCH_TINY_STREAMING
    options, 1, moonshine_get_version());
```

Then read words from each transcript line:

```c
for (uint64_t i = 0; i < transcript->line_count; i++) {
    transcript_line_t *line = &transcript->lines[i];
    for (uint64_t j = 0; j < line->word_count; j++) {
        transcript_word_t *word = &line->words[j];
        printf("[%.3fs - %.3fs] %s (conf: %.2f)\n",
               word->start, word->end, word->text, word->confidence);
    }
}
```

### C++ Wrapper

```cpp
moonshine::Transcriber transcriber(modelPath, moonshine::ModelArch::TINY,
    {{"word_timestamps", "true"}});

auto transcript = transcriber.transcribeWithoutStreaming(audioData, 16000);
for (auto &line : transcript.lines) {
    for (auto &word : line.words) {
        std::cout << "[" << word.start << "s - " << word.end << "s] "
                  << word.word << " (conf: " << word.confidence << ")\n";
    }
}
```

### Python

```python
from moonshine_voice import Transcriber, ModelArch

transcriber = Transcriber(model_path, ModelArch.TINY,
    options={"word_timestamps": "true"})
transcript = transcriber.transcribe_without_streaming(audio_data)

for line in transcript.lines:
    if line.words:
        for word in line.words:
            print(f"[{word.start:.3f}s - {word.end:.3f}s] {word.word}")
```

### Swift

```swift
let transcriber = try Transcriber(
    modelPath: modelPath,
    modelArch: .tiny,
    options: [TranscriberOption(name: "word_timestamps", value: "true")])

let transcript = try transcriber.transcribeWithoutStreaming(audioData: audio)
for line in transcript.lines {
    for word in line.words {
        print("[\(word.start)s - \(word.end)s] \(word.word)")
    }
}
```

## Model Files Required

When `word_timestamps=true`, the transcriber looks for an attention-enabled decoder alongside the existing model files. The file name depends on the model type:

| Model Type | Standard Decoder | Attention-Enabled Decoder |
|-----------|-----------------|--------------------------|
| Non-streaming | `decoder_model_merged.ort` | `decoder_with_attention.ort` |
| Streaming | `decoder_kv.ort` | `decoder_kv_with_attention.ort` |

These are the same decoder models with 6 additional outputs that surface the cross-attention weights already computed internally. Same weights, same size, same speed.

Generate them with the included script:

```bash
# Non-streaming model
python scripts/export-decoder-with-attention.py path/to/tiny-en/

# Streaming model
python scripts/export-decoder-with-attention.py path/to/tiny-streaming-en/
```

The script auto-detects the model type and produces the correct output file.

## How It Works

### The Technique: Cross-Attention DTW

Moonshine is an encoder-decoder transformer. The encoder converts audio into a sequence of frames. The decoder autoregressively generates text tokens, attending to the encoder frames via cross-attention at each layer.

At each decoder layer, cross-attention computes:

```
attention_weights = softmax(Q @ K^T / sqrt(d))
```

where Q comes from the decoder's hidden state and K comes from the encoder output. These weights form a `[heads, 1, encoder_frames]` distribution showing which audio frames the decoder attends to when producing each token.

We collect these weights at every decode step, stack them into a `[layers × heads, tokens, encoder_frames]` matrix, then apply Dynamic Time Warping (DTW) to find the optimal monotonic alignment path mapping each token to a range of audio frames. Finally, sub-word tokens are grouped into words using SentencePiece boundary markers (`▁`).

### Single-Pass Implementation

The decoder already computes these attention weights internally — they just aren't exposed as model outputs. The export script modifies the ONNX graph to surface them:

1. Load the original optimized `.ort` decoder
2. Save it as standard ONNX format
3. Find the `encoder_attn/Softmax` output tensors (one per decoder layer)
4. Add `Identity` nodes wiring them to new graph outputs named `cross_attentions.0` through `cross_attentions.5`
5. Convert back to ORT format

For non-streaming models, the Softmax nodes are inside `If` subgraphs (optimum's cache branching). For streaming models, the graph is flat and the cross-attention Softmax nodes are identified by their `mul_*` inputs (vs `masked_fill*` for self-attention).

The result is the same model with the same weights and optimizations — just 6 additional outputs. No retraining, no re-export from PyTorch, no accuracy change.

### Processing Pipeline

**Non-streaming:**
```
Audio
  → encoder_model.ort → encoder_hidden_states
  → decoder_with_attention.ort (autoregressive decode loop)
      → logits, KV cache (same as before)
      → cross_attentions.* (NEW: attention weights per step)
  → DTW alignment → word timestamps
```

**Streaming:**
```
Audio chunks
  → frontend.ort → encoder.ort → adapter.ort → memory
  → cross_kv.ort → precomputed K/V
  → decoder_kv_with_attention.ort (per-step decode)
      → logits, KV cache (same as before)
      → cross_attentions.* (NEW: attention weights per step)
  → DTW alignment → word timestamps
```

The DTW post-processing is the same for both paths:
```
stack per-step attention into [layers*heads, tokens, encoder_frames]
→ z-score normalize per head
→ median filter (width 7) to smooth
→ average across heads/layers → [tokens, encoder_frames]
→ DTW on negated matrix → monotonic alignment path
→ group tokens into words (SentencePiece ▁ boundaries)
→ fix overlapping word boundaries
→ TranscriberWord { text, start, end, confidence }
```

## Performance

The modified decoders are the same size and speed as the originals. Word timestamp overhead comes from reading 6 extra tensors per decode step and running DTW after decoding.

### Non-streaming (tiny)

| Audio | Duration | Without | With | Overhead |
|-------|----------|---------|------|----------|
| beckett.wav | 10s | 161ms | 143ms | ~0% (within noise) |
| two_cities | 44s | 813ms | 914ms | +12% |

### Streaming (tiny-streaming)

| Audio | Duration | Without | With | Overhead |
|-------|----------|---------|------|----------|
| beckett.wav | 10s | 203ms | 200ms | ~0% (within noise) |
| two_cities | 44s | 1111ms | 1173ms | +6% |

WER is unchanged — the transcription text is identical (the model weights and computation are the same).

## Data Structures

### C API (`moonshine-c-api.h`)

```c
struct transcript_word_t {
    const char *text;    /* UTF-8 word text */
    float start;         /* Start time in seconds (absolute) */
    float end;           /* End time in seconds */
    float confidence;    /* 0.0 to 1.0 */
};

struct transcript_line_t {
    /* ... existing fields ... */
    const struct transcript_word_t *words;  /* NULL if not enabled */
    uint64_t word_count;                    /* 0 if not enabled */
};
```

### C++ Wrapper (`moonshine-cpp.h`)

```cpp
struct WordTiming {
    std::string word;
    float start;
    float end;
    float confidence;
};

struct TranscriptLine {
    /* ... existing fields ... */
    std::vector<WordTiming> words;  // empty if not enabled
};
```

### Python (`moonshine_api.py`)

```python
@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float

@dataclass
class TranscriptLine:
    # ... existing fields ...
    words: Optional[List[WordTiming]] = None
```

### Swift (`Transcript.swift`)

```swift
public struct WordTiming {
    public let word: String
    public let start: Float
    public let end: Float
    public let confidence: Float
}

public struct TranscriptLine {
    // ... existing fields ...
    public let words: [WordTiming]  // empty if not enabled
}
```

### Android/Java

```java
public class WordTiming {
    public String word;
    public float start;
    public float end;
    public float confidence;
}

public class TranscriptLine {
    // ... existing fields ...
    public List<WordTiming> words;
}
```

### JavaScript/TypeScript

```typescript
export interface WordTiming {
    word: string;
    start: number;
    end: number;
    confidence: number;
}
```

## Files Changed

### New Files
- `core/word-alignment.h` / `core/word-alignment.cpp` — DTW algorithm, median filter, token-to-word grouping
- `core/word-alignment-test.cpp` — C++ correctness test
- `core/word-alignment-benchmark.cpp` — Latency benchmark (supports both model types)
- `scripts/export-decoder-with-attention.py` — Generate attention-enabled decoder from existing model
- `test-assets/tiny-en/decoder_with_attention.ort` — Non-streaming decoder with attention outputs
- `test-assets/tiny-streaming-en/decoder_kv_with_attention.ort` — Streaming decoder with attention outputs
- `android/java/main/java/ai/moonshine/voice/WordTiming.java` — Android word timing class
- `docs/word-level-timestamps.md` — This document

### Modified Files
- `core/moonshine-c-api.h` — Added `transcript_word_t`, `words`/`word_count` on `transcript_line_t`
- `core/moonshine-c-api.cpp` — Parse `word_timestamps` option
- `core/moonshine-model.h` / `core/moonshine-model.cpp` — Collect cross-attention during non-streaming decode, `compute_word_timestamps()` method
- `core/moonshine-streaming-model.h` / `core/moonshine-streaming-model.cpp` — Collect cross-attention during streaming decode
- `core/moonshine-cpp.h` — Added `WordTiming`, `words` on `TranscriptLine`
- `core/transcriber.h` / `core/transcriber.cpp` — Load attention-enabled decoder (both paths), compute word timestamps after transcription
- `core/CMakeLists.txt` — Added `word-alignment.cpp`, test and benchmark targets
- `python/src/moonshine_voice/moonshine_api.py` — Added `TranscriptWordC`, `WordTiming`, `words` field
- `python/src/moonshine_voice/transcriber.py` — Parse words from C struct
- `swift/Sources/MoonshineVoice/Transcript.swift` — Added `WordTiming`, `words` field
- `swift/Sources/MoonshineVoice/MoonshineAPI.swift` — Parse words from C struct
- `android/java/main/java/ai/moonshine/voice/TranscriptLine.java` — Added `words` field
- `android/moonshine-jni/moonshine-jni.cpp` — Populate word timestamps via JNI
