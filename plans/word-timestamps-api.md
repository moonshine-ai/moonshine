# Plan: Add Word-Level Timestamps to the Moonshine API

**Goal:** Expose per-word `start`, `end`, and `confidence` timestamps as an optional field on transcript lines, across all API layers — matching the request in [#145](https://github.com/moonshine-ai/moonshine/issues/145).

**Approach:** The cross-attention DTW alignment (proven in `word_timestamps_v2.py`) runs inside the C++ decode loop with near-zero overhead (~2-3%). We add new structs to carry word timing data through the existing API surface.

---

## New Data Structures

### C API (`moonshine-c-api.h`)

```c
/* A single word with timing information */
struct transcript_word_t {
    const char *text;       /* UTF-8 word text */
    float start;            /* Start time in seconds (relative to line start or audio start — TBD) */
    float end;              /* End time in seconds */
    float confidence;       /* Model confidence 0.0–1.0 */
};
```

Add to existing `transcript_line_t`:

```c
struct transcript_line_t {
    /* ... existing fields unchanged ... */

    /* Word-level timestamps (NULL if not requested) */
    const struct transcript_word_t *words;
    uint64_t word_count;
};
```

### C++ Internal (`transcriber.h` — `TranscriberLine`)

```cpp
struct TranscriberWord {
    std::string text;
    float start;        // seconds, absolute (from audio start)
    float end;
    float confidence;
};

struct TranscriberLine {
    /* ... existing fields unchanged ... */
    std::vector<TranscriberWord> words;
};
```

### C++ Wrapper (`moonshine-cpp.h` — `TranscriptLine`)

```cpp
struct WordTiming {
    std::string word;
    float start;
    float end;
    float confidence;
};

struct TranscriptLine {
    /* ... existing fields unchanged ... */
    std::vector<WordTiming> words;
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
    # ... existing fields unchanged ...
    words: list[WordTiming]   # empty list if not requested
```

ctypes struct:

```python
class TranscriptWordC(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.POINTER(ctypes.c_char)),
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]
```

Add to `TranscriptLineC`:

```python
("words", ctypes.POINTER(TranscriptWordC)),
("word_count", ctypes.c_uint64),
```

### Swift (`TranscriptLine`)

```swift
public struct WordTiming {
    public let word: String
    public let start: Float
    public let end: Float
    public let confidence: Float
}

public struct TranscriptLine {
    // ... existing fields unchanged ...
    public let words: [WordTiming]
}
```

### Android/Java (`TranscriptLine`)

```java
public class WordTiming {
    public String word;
    public float start;
    public float end;
    public float confidence;
}

public class TranscriptLine {
    // ... existing fields unchanged ...
    public List<WordTiming> words;
}
```

---

## Opt-In Mechanism

Word timestamps are **off by default** (zero overhead for existing users). Enabled via the existing options system:

```c
transcriber_option_t options[] = {
    { "word_timestamps", "true" }
};
```

This flows through the existing `transcriber_option_t` array that `moonshine_load_transcriber_from_files()` already accepts.

When disabled:
- `transcript_line_t.words = NULL`
- `transcript_line_t.word_count = 0`

When enabled:
- Words are populated after each decode pass
- Cross-attention weights are collected during the decode loop (single-pass, ~2-3% overhead)
- DTW + word grouping runs post-decode (microseconds)

---

## Implementation Plan — Layer by Layer (Bottom Up)

### Phase 1: C++ Core — Collect Cross-Attention & Run DTW

**Files:** `moonshine-model.cpp`, `moonshine-streaming-model.cpp`

**Problem:** The ONNX `.ort` models don't expose cross-attention weights as outputs. The decoder computes attention internally and only outputs logits + KV cache.

**Options:**

**Option A — Modify ONNX models** to output cross-attention weights as additional tensors. Requires re-exporting from PyTorch with attention outputs wired to graph outputs. This is the cleanest path but requires model re-export tooling.

**Option B — Compute Q·K externally.** We already have `k_cross` / `v_cross` cached in `MoonshineStreamingState`. If we also extract the Q projection weights from the ONNX model at load time, we can compute `softmax(Q @ K^T / sqrt(d))` externally at each decode step. Q would be derived from the decoder hidden states — but those aren't exposed either.

**Option C — Use HuggingFace Transformers for the alignment pass.** Keep the ONNX pipeline for fast transcription, but optionally run a second alignment-only pass through a PyTorch model. This is what `word_timestamps_v2.py` does. Downside: requires PyTorch as a dependency.

**Option D — Re-export ONNX decoder with attention outputs.** Export a variant `decoder_kv_with_attn.ort` that includes cross-attention weights as additional output tensors. Load this variant when `word_timestamps=true`. The standard model stays unchanged. This is the most practical path for the C++ core:

1. Export a new ONNX decoder variant from PyTorch that adds cross-attention weights to outputs
2. At load time, if `word_timestamps=true`, load `decoder_kv_with_attn.ort` instead of `decoder_kv.ort`
3. At each decode step, read the additional attention output tensors
4. After decoding completes, run DTW in C++ and populate `TranscriberWord` structs

**Recommendation: Option D** — it keeps the C++ code simple (just reading extra ONNX outputs), doesn't require PyTorch at runtime, and the attention-enabled model variant is only loaded when needed.

#### DTW Implementation in C++

Port the DTW algorithm from `word_timestamps_v2.py` to C++:

```
New files:
  core/word-alignment.h    — DTW, median filter, word grouping functions
  core/word-alignment.cpp  — Implementation
```

Functions:
- `dtw(cost_matrix) → (text_indices, time_indices)` — O(N×M) dynamic programming
- `median_filter(matrix, width)` — Smoothing
- `align_words(cross_attention, tokens, encoder_frames, time_per_frame, tokenizer) → vector<TranscriberWord>`

This is pure math, no external dependencies.

#### Tokenizer Word Grouping

The existing `BinTokenizer` needs a method to detect word boundaries:
- Check if a token's raw text starts with `▁` (SentencePiece word boundary marker)
- Group consecutive sub-word tokens into words
- Merge timing from token-level to word-level

### Phase 2: C API — Expose New Struct

**File:** `moonshine-c-api.h`, `moonshine-c-api.cpp`

1. Add `transcript_word_t` struct definition
2. Add `words` and `word_count` fields to `transcript_line_t`
3. In `update_transcript_from_lines()` (in `transcriber.cpp`): copy `TranscriberWord` data into `transcript_word_t` array
4. Memory management: word arrays are owned by `TranscriptStreamOutput`, freed on next update or teardown — same lifetime as existing `transcript_line_t` data

### Phase 3: C++ Wrapper

**File:** `moonshine-cpp.h`

1. Add `WordTiming` struct
2. Add `std::vector<WordTiming> words` to `TranscriptLine`
3. Update `TranscriptLine(const transcript_line_t&)` constructor to copy word data
4. Zero code change needed in event system — events already carry `TranscriptLine`

### Phase 4: Python Bindings

**File:** `python/src/moonshine_voice/moonshine_api.py`, `transcriber.py`

1. Add `TranscriptWordC` ctypes struct
2. Add fields to `TranscriptLineC`
3. Add `WordTiming` dataclass
4. Update `TranscriptLine` dataclass with `words: list[WordTiming]`
5. Update conversion code that maps `TranscriptLineC` → `TranscriptLine`

### Phase 5: Swift Bindings

**File:** `swift/Sources/MoonshineVoice/`

1. Add `WordTiming` struct
2. Add `words: [WordTiming]` to `TranscriptLine`
3. Update C-to-Swift conversion

### Phase 6: Android/Java Bindings

**File:** `android/java/`

1. Add `WordTiming` class
2. Add `words` field to `TranscriptLine`
3. Update JNI bridge to copy word data

---

## Streaming vs Non-Streaming

### Non-Streaming

Straightforward. The full audio is encoded, then decoded in one autoregressive pass. Cross-attention weights are collected during the decode loop, DTW runs once at the end. This is exactly what the prototype does.

### Streaming

The streaming model decodes incrementally as audio chunks arrive. Two aspects to consider:

**1. Collecting cross-attention weights during streaming decode:**

Each call to `decode_step()` / `decode_tokens()` produces one token. With the attention-enabled ONNX variant, each step also emits a `[layers, heads, 1, cross_len]` attention tensor. We append this to a per-line accumulator in `MoonshineStreamingState`.

Cost per step: one tensor copy. Measured at ~0.2ms — negligible vs the decode step itself.

**2. When to run DTW:**

- **On `is_complete` lines** — after VAD detects end of speech and the line is finalized. DTW runs once on the complete token sequence. Zero impact on per-chunk streaming latency.
- **On every `transcribe_stream()` call** — gives live word timestamps on provisional text. DTW cost for a typical in-progress segment (10-20 tokens, 50-100 frames) is < 0.1ms.

**Recommendation:** Run DTW on every `transcribe_stream()` call. The cost is microseconds for in-progress segments, and consumers already expect provisional data to change (via `has_text_changed`). Word timestamps on provisional lines follow the same pattern — they refine as more audio arrives.

**Latency impact summary:**

| Operation | Cost | When |
|-----------|------|------|
| Attention tensor copy per decode step | ~0.2ms | During decode |
| DTW on completed line (30 tokens, 150 frames) | < 0.1ms | After line completion |
| DTW on provisional line (10 tokens, 50 frames) | < 0.05ms | Each transcribe_stream() call |
| **Total overhead per streaming update** | **< 0.5ms** | — |

Moonshine V2 streaming achieves 50ms latency on tiny. Adding < 0.5ms is < 1% overhead — no meaningful impact on the latency goal.

### Cross-attention state management

When a streaming line is updated (new audio → re-encode → re-decode):
- The cross-attention accumulator for that line is **reset** when the line's tokens change
- Re-populated during the new decode pass
- DTW re-runs on the updated attention matrix

This mirrors how `k_self` / `v_self` caches are already managed in `MoonshineStreamingState`.

---

## Open Questions

1. **Word timestamps: absolute or relative?**
   - **Absolute** (from start of audio): simpler, matches Whisper/WhisperX convention
   - **Relative** (from start of line): matches `transcript_line_t.start_time` + `duration` pattern
   - **Recommendation:** Absolute — easier for consumers, and the line's `start_time` is already absolute

2. **ONNX model re-export:** Who exports the attention-enabled variant? Is there an existing export script in the repo, or does this need to be created from the HuggingFace model?

3. **Non-streaming model path:** The non-streaming `decoder_model_merged.ort` has a different structure from the streaming `decoder_kv.ort`. Both need attention-enabled variants.

---

## Phasing

| Phase | Scope | Dependency |
|-------|-------|------------|
| 1a | Export attention-enabled ONNX model variants | HuggingFace model access |
| 1b | C++ DTW + word alignment (`word-alignment.h/cpp`) | None |
| 1c | Integrate into decode loop + Transcriber | 1a + 1b |
| 2 | C API struct additions | 1c |
| 3 | C++ wrapper additions | 2 |
| 4 | Python bindings | 2 |
| 5 | Swift bindings | 2 |
| 6 | Android/Java bindings | 2 |

Phases 3-6 are independent of each other and can be done in parallel once Phase 2 is complete.

---

## Testing

- Reuse the audio chop technique from the prototype: cut audio at word boundaries and verify clips sound correct
- Compare word timestamps against `word_timestamps_v2.py` output on the same audio — should match
- Benchmark: verify `word_timestamps=true` adds < 5% overhead vs `word_timestamps=false`
- Monotonicity check: verify no word's start time is before the previous word's start time
