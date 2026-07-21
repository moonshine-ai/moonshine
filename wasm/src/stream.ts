/**
 * Streaming transcription session. Wraps the embind Stream and turns raw
 * transcript snapshots into {@link TranscriptEventListener} callbacks, matching
 * the event-driven API of the other bindings.
 */

import {
  createDiffState,
  diffTranscript,
  dispatchError,
  type DiffState,
  type TranscriptEventListener,
} from './events.js';
import { toMoonshineError, wrapErrors } from './errors.js';
import { TranscribeFlags } from './enums.js';
import type { RawStream } from './module.js';
import { normalizeTranscript, type Transcript } from './types.js';

export class Stream {
  private readonly raw: RawStream;
  private readonly listeners: TranscriptEventListener[] = [];
  private readonly diff: DiffState = createDiffState();
  private closed = false;

  /** @internal Constructed via {@link Transcriber.createStream}. */
  constructor(raw: RawStream) {
    this.raw = raw;
  }

  addListener(listener: TranscriptEventListener): void {
    this.listeners.push(listener);
  }

  removeListener(listener: TranscriptEventListener): void {
    const i = this.listeners.indexOf(listener);
    if (i >= 0) this.listeners.splice(i, 1);
  }

  removeAllListeners(): void {
    this.listeners.length = 0;
  }

  start(): void {
    wrapErrors(() => this.raw.start());
  }

  stop(): void {
    wrapErrors(() => this.raw.stop());
    // Flush a final transcription so any trailing line is completed/emitted.
    this.transcribe(TranscribeFlags.ForceUpdate);
  }

  /**
   * Feeds PCM audio (mono float in [-1, 1]) into the stream buffer. Cheap; call
   * as often as your audio source produces chunks.
   */
  addAudio(
    audio: Float32Array,
    sampleRate: number,
    flags: TranscribeFlags = TranscribeFlags.None,
  ): void {
    wrapErrors(() => this.raw.addAudio(audio, sampleRate, flags));
  }

  /**
   * Runs a transcription pass over the buffered audio, dispatches diffed events
   * to listeners, and returns the current transcript snapshot.
   */
  transcribe(flags: TranscribeFlags = TranscribeFlags.None): Transcript {
    let transcript: Transcript;
    try {
      transcript = normalizeTranscript(this.raw.transcribe(flags));
    } catch (err) {
      const wrapped = toMoonshineError(err);
      dispatchError(this.listeners, wrapped);
      throw wrapped;
    }
    diffTranscript(transcript, this.diff, this.listeners);
    return transcript;
  }

  close(): void {
    if (!this.closed) {
      this.closed = true;
      wrapErrors(() => this.raw.close());
    }
  }

  /** Enables `using` (explicit resource management). */
  [Symbol.dispose](): void {
    this.close();
  }
}
