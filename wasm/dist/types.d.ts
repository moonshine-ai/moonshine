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
/** @internal Normalizes a raw embind transcript into the public shape. */
export declare function normalizeTranscript(raw: any): Transcript;
/** @internal Normalizes a raw embind intent-match vector. */
export declare function normalizeIntentMatches(raw: any): IntentMatch[];
//# sourceMappingURL=types.d.ts.map