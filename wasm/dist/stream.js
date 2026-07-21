/**
 * Streaming transcription session. Wraps the embind Stream and turns raw
 * transcript snapshots into {@link TranscriptEventListener} callbacks, matching
 * the event-driven API of the other bindings.
 */
import { createDiffState, diffTranscript, dispatchError, } from './events.js';
import { toMoonshineError, wrapErrors } from './errors.js';
import { TranscribeFlags } from './enums.js';
import { normalizeTranscript } from './types.js';
export class Stream {
    raw;
    listeners = [];
    diff = createDiffState();
    closed = false;
    /** @internal Constructed via {@link Transcriber.createStream}. */
    constructor(raw) {
        this.raw = raw;
    }
    addListener(listener) {
        this.listeners.push(listener);
    }
    removeListener(listener) {
        const i = this.listeners.indexOf(listener);
        if (i >= 0)
            this.listeners.splice(i, 1);
    }
    removeAllListeners() {
        this.listeners.length = 0;
    }
    start() {
        wrapErrors(() => this.raw.start());
    }
    stop() {
        wrapErrors(() => this.raw.stop());
        // Flush a final transcription so any trailing line is completed/emitted.
        this.transcribe(TranscribeFlags.ForceUpdate);
    }
    /**
     * Feeds PCM audio (mono float in [-1, 1]) into the stream buffer. Cheap; call
     * as often as your audio source produces chunks.
     */
    addAudio(audio, sampleRate, flags = TranscribeFlags.None) {
        wrapErrors(() => this.raw.addAudio(audio, sampleRate, flags));
    }
    /**
     * Runs a transcription pass over the buffered audio, dispatches diffed events
     * to listeners, and returns the current transcript snapshot.
     */
    transcribe(flags = TranscribeFlags.None) {
        let transcript;
        try {
            transcript = normalizeTranscript(this.raw.transcribe(flags));
        }
        catch (err) {
            const wrapped = toMoonshineError(err);
            dispatchError(this.listeners, wrapped);
            throw wrapped;
        }
        diffTranscript(transcript, this.diff, this.listeners);
        return transcript;
    }
    close() {
        if (!this.closed) {
            this.closed = true;
            wrapErrors(() => this.raw.close());
        }
    }
    /** Enables `using` (explicit resource management). */
    [Symbol.dispose]() {
        this.close();
    }
}
//# sourceMappingURL=stream.js.map