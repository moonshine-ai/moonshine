# Word-Level Timestamps for Moonshine ASR

## Overview

Moonshine now supports optional word-level timestamps on transcription output. When enabled, each transcript line includes an array of words with `start`, `end`, and `confidence` values derived from the model's own cross-attention alignment — no external alignment model or second pass required.

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
    model_path, MOONSHINE_MODEL_ARCH_TINY,
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

When `word_timestamps=true`, the transcriber looks for `decoder_with_attention.ort` in the model directory (alongside the existing `encoder_model.ort` and `decoder_model_merged.ort`). This is the same decoder model with 6 additional outputs that surface the cross-attention weights already computed internally.

If `decoder_with_attention.ort` is not found, it falls back to `alignment_model.ort` (a separate teacher-forced decoder) which runs as a second pass after transcription.

| Model File | Purpose | Size (tiny) |
|-----------|---------|-------------|
| `decoder_with_attention.ort` | Single-pass decoder with attention outputs (preferred) | ~29 MB |
| `alignment_model.ort` | Two-pass fallback alignment model | ~110 MB |

## How It Works

### The Technique: Cross-Attention DTW

Moonshine is an encoder-decoder transformer. The encoder converts audio into a sequence of frames (one per ~24ms for V1 tiny). The decoder autoregressively generates text tokens, attending to the encoder frames via cross-attention at each layer.

At each decoder layer, cross-attention computes:

```
attention_weights = softmax(Q @ K^T / sqrt(d))
```

where Q comes from the decoder's hidden state and K comes from the encoder output. These weights form a `[heads, 1, encoder_frames]` distribution showing which audio frames the decoder attends to when producing each token.

We collect these weights at every decode step, stack them into a `[layers × heads, tokens, encoder_frames]` matrix, then apply Dynamic Time Warping (DTW) to find the optimal monotonic alignment path mapping each token to a range of audio frames. Finally, sub-word tokens are grouped into words using SentencePiece boundary markers (`▁`).

### Single-Pass Implementation

The key insight: the decoder already computes these attention weights internally — they just aren't exposed as model outputs. We modify the ONNX graph to surface them:

1. Load the original optimized `decoder_model_merged.ort`
2. Save it as standard ONNX format
3. Inside the `If` node's subgraphs, find the 6 `encoder_attn/Softmax` output tensors (one per decoder layer)
4. Add `Identity` nodes wiring them to new graph outputs named `cross_attentions.0` through `cross_attentions.5`
5. Convert back to ORT format

The result is the same model with the same weights and optimizations — just 6 additional outputs. No retraining, no re-export from PyTorch, no accuracy change.

### Processing Pipeline

```
Audio
  → encoder_model.ort → encoder_hidden_states
  → decoder_with_attention.ort (autoregressive decode loop)
      → logits (for next token selection, same as before)
      → present.* (KV cache updates, same as before)
      → cross_attentions.* (NEW: attention weights per step)
  → stack per-step attention into [layers*heads, tokens, encoder_frames]
  → z-score normalize per head
  → median filter (width 7) to smooth
  → average across heads/layers → [tokens, encoder_frames]
  → DTW on negated matrix → monotonic alignment path
  → group tokens into words (SentencePiece ▁ boundaries)
  → fix overlapping word boundaries
  → TranscriberWord { text, start, end, confidence }
```

## Performance

The modified decoder is the same size and speed as the original. Word timestamp overhead comes only from reading 6 extra tensors per decode step (memcpy) and running DTW after decoding (microseconds for typical segment lengths).

| Audio | Duration | Without | With | Overhead |
|-------|----------|---------|------|----------|
| beckett.wav | 10s | 158ms | 149ms | ~0% (within noise) |
| two_cities | 44s | 843ms | 919ms | +9% |

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

## Generating the Attention-Enabled Decoder

To create `decoder_with_attention.ort` for a model:

```python
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto

# Step 1: Load the original .ort and save as ONNX
so = ort.SessionOptions()
so.optimized_model_filepath = "/tmp/decoder.onnx"
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
ort.InferenceSession("decoder_model_merged.ort", so)

# Step 2: Add attention outputs to the ONNX graph
model = onnx.load("/tmp/decoder.onnx")
top_node = [n for n in model.graph.node if n.op_type == "If"][0]
NUM_LAYERS = 6  # 6 for tiny, 8 for base

for attr in top_node.attribute:
    if attr.type != onnx.AttributeProto.GRAPH:
        continue
    g = attr.g
    if not g.name:
        g.name = attr.name
    for node in g.node:
        if node.op_type == "Softmax":
            for inp in node.input:
                if "encoder_attn" in inp:
                    layer_idx = int(inp.split("layers.")[1].split("/")[0])
                    out_name = f"cross_attentions.{layer_idx}"
                    g.node.append(helper.make_node(
                        "Identity", [node.output[0]], [out_name],
                        name=f"attn_id_{attr.name}_{layer_idx}"))
                    g.output.append(helper.make_tensor_value_info(
                        out_name, TensorProto.FLOAT, None))

for i in range(NUM_LAYERS):
    top_node.output.append(f"cross_attentions.{i}")
    model.graph.output.append(helper.make_tensor_value_info(
        f"cross_attentions.{i}", TensorProto.FLOAT, None))

onnx.save(model, "decoder_with_attention.onnx")

# Step 3: Convert to ORT format
so2 = ort.SessionOptions()
so2.optimized_model_filepath = "decoder_with_attention.ort"
so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
ort.InferenceSession("decoder_with_attention.onnx", so2)
```

## Files Changed

### New Files
- `core/word-alignment.h` / `core/word-alignment.cpp` — DTW algorithm, median filter, token-to-word grouping
- `core/word-alignment-test.cpp` — C++ test
- `core/word-alignment-benchmark.cpp` — Latency benchmark
- `test-assets/tiny-en/decoder_with_attention.onnx` — Modified decoder model
- `export_decoder_with_attention.py` — Alignment model export script (fallback path)
- `word_timestamps.py` / `word_timestamps_v2.py` — Python prototypes
- `benchmark_timestamps.py` — Python benchmark
- `test_python_api.py` / `test_swift_api.swift` — API tests
- `plans/word-timestamps-api.md` — Implementation plan
- `docs/word-level-timestamps.md` — This document

### Modified Files
- `core/moonshine-c-api.h` — Added `transcript_word_t`, `words`/`word_count` on `transcript_line_t`
- `core/moonshine-c-api.cpp` — Parse `word_timestamps` option
- `core/moonshine-model.h` / `core/moonshine-model.cpp` — Save encoder states and tokens, collect cross-attention during decode, `compute_word_timestamps()` method
- `core/moonshine-cpp.h` — Added `WordTiming`, `words` on `TranscriptLine`
- `core/transcriber.h` — Added `word_timestamps` option, `#include "word-alignment.h"`
- `core/transcriber.cpp` — Load attention-enabled decoder, call `compute_word_timestamps()` after transcription
- `core/CMakeLists.txt` — Added `word-alignment.cpp`, test and benchmark targets
- `python/src/moonshine_voice/moonshine_api.py` — Added `TranscriptWordC`, `WordTiming`, `words` field
- `python/src/moonshine_voice/transcriber.py` — Parse words from C struct
- `swift/Sources/MoonshineVoice/Transcript.swift` — Added `WordTiming`, `words` field
- `swift/Sources/MoonshineVoice/MoonshineAPI.swift` — Parse words from C struct
