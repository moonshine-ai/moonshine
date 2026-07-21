/**
 * Live microphone transcription, mirroring the `MicrophoneTranscriber` helpers
 * in the other bindings. Captures mic audio via WebAudio, downmixes to mono +
 * resamples to 16 kHz, and feeds a streaming {@link Stream}.
 *
 * Uses an AudioWorklet when available (off the main thread), falling back to a
 * ScriptProcessorNode. All model work still happens through the WASM binding.
 */
import { TranscribeFlags } from './enums.js';
import type { TranscriptEventListener } from './events.js';
import { Transcriber, type TranscriberLoadOptions } from './transcriber.js';
export interface MicrophoneTranscriberOptions {
    /** An already-loaded transcriber. If omitted, provide {@link load}. */
    transcriber?: Transcriber;
    /** Constraints passed to getUserMedia({ audio }). */
    audioConstraints?: MediaTrackConstraints | boolean;
    flags?: TranscribeFlags;
    listeners?: TranscriptEventListener[];
}
export declare class MicrophoneTranscriber {
    private readonly transcriber;
    private readonly ownsTranscriber;
    private readonly flags;
    private readonly listeners;
    private readonly audioConstraints;
    private mediaStream?;
    private audioContext?;
    private sourceNode?;
    private workletNode?;
    private scriptNode?;
    private stream?;
    private running;
    private constructor();
    /**
     * Creates a MicrophoneTranscriber. Either pass an existing `transcriber`, or
     * pass {@link TranscriberLoadOptions} to load one automatically.
     */
    static load(options: MicrophoneTranscriberOptions & Partial<TranscriberLoadOptions>): Promise<MicrophoneTranscriber>;
    addListener(listener: TranscriptEventListener): void;
    /** Starts capturing and transcribing. Resolves once audio is flowing. */
    start(): Promise<void>;
    /** Stops capture, flushes a final transcript, and releases audio resources. */
    stop(): Promise<void>;
    /** Releases the stream (and the transcriber, if this instance loaded it). */
    close(): void;
    private setupWorklet;
    private setupScriptProcessor;
}
//# sourceMappingURL=microphone-transcriber.d.ts.map