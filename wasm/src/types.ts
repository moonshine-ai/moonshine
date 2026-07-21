/**
 * Plain, readonly value types returned by the binding. These mirror the C
 * structs in `core/moonshine-c-api.h` and the value types the Python/Swift/
 * Android bindings expose, but as idiomatic JS objects (no heap views).
 */

/** A single word with timing information (when word timestamps are enabled). */
export interface WordTiming {
  readonly text: string;
  /** Start time in seconds (absolute, from the start of the audio/stream). */
  readonly start: number;
  /** End time in seconds. */
  readonly end: number;
  /** Model confidence score, 0.0–1.0. */
  readonly confidence: number;
}

/** One contiguous span of a line attributed to a single speaker. */
export interface SpeakerSpan {
  readonly startTime: number;
  readonly duration: number;
  /** Stable speaker id within the stream (opaque; safe for keying/display). */
  readonly speakerId: number;
  /** Order the speaker first appeared, starting at 0. */
  readonly speakerIndex: number;
  /** UTF-8 byte offset into the line text where this span begins. */
  readonly startChar: number;
  /** UTF-8 byte offset into the line text where this span ends. */
  readonly endChar: number;
}

/** One "line" of a transcript — a phrase or sentence. */
export interface TranscriptLine {
  readonly text: string;
  readonly startTime: number;
  readonly duration: number;
  /** Stable identifier for the line. */
  readonly id: number;
  readonly isComplete: boolean;
  readonly isUpdated: boolean;
  readonly isNew: boolean;
  readonly hasTextChanged: boolean;
  readonly haveSpeakersChanged: boolean;
  readonly lastTranscriptionLatencyMs: number;
  readonly words: readonly WordTiming[];
  readonly speakerSpans: readonly SpeakerSpan[];
}

/** A full transcript: an ordered list of lines. */
export interface Transcript {
  readonly lines: readonly TranscriptLine[];
}

/** A ranked intent match from {@link IntentRecognizer.closestIntents}. */
export interface IntentMatch {
  readonly canonicalPhrase: string;
  readonly similarity: number;
}

/** Synthesized audio returned by {@link TextToSpeech.say}. */
export interface TtsSynthesisResult {
  /** Mono float PCM in [-1, 1]. */
  readonly audio: Float32Array;
  readonly sampleRate: number;
}

// --- Internal helpers: normalize raw embind objects into the clean types. ---

/** Reads an embind value that may be a native array or a bound vector. */
function readVector<T>(raw: unknown): T[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) return raw as T[];
  // Embind register_vector<> object exposes size()/get().
  const vec = raw as { size?: () => number; get?: (i: number) => T };
  if (typeof vec.size === 'function' && typeof vec.get === 'function') {
    const out: T[] = [];
    const n = vec.size();
    for (let i = 0; i < n; i++) out.push(vec.get(i) as T);
    // Bound vectors must be explicitly freed to avoid leaking heap memory.
    (raw as { delete?: () => void }).delete?.();
    return out;
  }
  return [];
}

/** @internal Normalizes a raw embind transcript into the public shape. */
export function normalizeTranscript(raw: any): Transcript {
  const rawLines = readVector<any>(raw?.lines);
  const lines: TranscriptLine[] = rawLines.map((line) => ({
    text: line.text ?? '',
    startTime: line.startTime ?? 0,
    duration: line.duration ?? 0,
    id: line.id ?? 0,
    isComplete: !!line.isComplete,
    isUpdated: !!line.isUpdated,
    isNew: !!line.isNew,
    hasTextChanged: !!line.hasTextChanged,
    haveSpeakersChanged: !!line.haveSpeakersChanged,
    lastTranscriptionLatencyMs: line.lastTranscriptionLatencyMs ?? 0,
    words: readVector<any>(line.words).map((w) => ({
      text: w.text ?? '',
      start: w.start ?? 0,
      end: w.end ?? 0,
      confidence: w.confidence ?? 0,
    })),
    speakerSpans: readVector<any>(line.speakerSpans).map((s) => ({
      startTime: s.startTime ?? 0,
      duration: s.duration ?? 0,
      speakerId: s.speakerId ?? 0,
      speakerIndex: s.speakerIndex ?? 0,
      startChar: s.startChar ?? 0,
      endChar: s.endChar ?? 0,
    })),
  }));
  return { lines };
}

/** @internal Normalizes a raw embind intent-match vector. */
export function normalizeIntentMatches(raw: any): IntentMatch[] {
  return readVector<any>(raw).map((m) => ({
    canonicalPhrase: m.canonicalPhrase ?? '',
    similarity: m.similarity ?? 0,
  }));
}
