/**
 * Streaming transcription session. Wraps the embind Stream and turns raw
 * transcript snapshots into {@link TranscriptEventListener} callbacks, matching
 * the event-driven API of the other bindings.
 */
import { type TranscriptEventListener } from './events.js';
import { TranscribeFlags } from './enums.js';
import type { RawStream } from './module.js';
import { type Transcript } from './types.js';
export declare class Stream {
    private readonly raw;
    private readonly listeners;
    private readonly diff;
    private closed;
    /** @internal Constructed via {@link Transcriber.createStream}. */
    constructor(raw: RawStream);
    addListener(listener: TranscriptEventListener): void;
    removeListener(listener: TranscriptEventListener): void;
    removeAllListeners(): void;
    start(): void;
    stop(): void;
    /**
     * Feeds PCM audio (mono float in [-1, 1]) into the stream buffer. Cheap; call
     * as often as your audio source produces chunks.
     */
    addAudio(audio: Float32Array, sampleRate: number, flags?: TranscribeFlags): void;
    /**
     * Runs a transcription pass over the buffered audio, dispatches diffed events
     * to listeners, and returns the current transcript snapshot.
     */
    transcribe(flags?: TranscribeFlags): Transcript;
    close(): void;
    /** Enables `using` (explicit resource management). */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=stream.d.ts.map