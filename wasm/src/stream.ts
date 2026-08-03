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

/**
 * Seconds of new audio a stream collects before it will run another pass.
 *
 * Matches the C++ `Stream`'s own default and the core's
 * `transcription_interval`, which is the interval below which the engine
 * declines to update anyway.
 */
export const DEFAULT_UPDATE_INTERVAL = 0.5;

/**
 * The most the interval will stretch to under load, as a multiple of itself.
 *
 * A pass that takes longer than the audio it covers widens the gate (see
 * {@link Stream.transcribe}), and one freak pass -- a garbage collection, a
 * machine that went to sleep mid-call -- must not be able to leave a live
 * transcript silent for a minute afterwards. Ten intervals is five seconds by
 * default: long enough for the widening to do its work, short enough that a page
 * still looks alive while it does.
 */
const MAX_INTERVAL_FACTOR = 10;

export class Stream {
  private readonly raw: RawStream;
  private readonly listeners: TranscriptEventListener[] = [];
  private readonly diff: DiffState = createDiffState();
  private readonly updateInterval: number;
  /** Seconds of audio added since the last pass over the engine. */
  private pending = 0;
  /** Wall-clock seconds the last pass took, which is what the next one must earn. */
  private lastPass = 0;
  private latest: Transcript = { lines: [] };
  private closed = false;

  /** @internal Constructed via {@link Transcriber.createStream}. */
  constructor(raw: RawStream, updateInterval: number = DEFAULT_UPDATE_INTERVAL) {
    this.raw = raw;
    this.updateInterval = Math.max(0, updateInterval);
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
    // A restarted stream begins from an empty transcript, so the snapshot held
    // back for callers who ask too soon must not be the last session's.
    this.latest = { lines: [] };
    this.pending = 0;
    this.lastPass = 0;
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
    if (sampleRate > 0) this.pending += audio.length / sampleRate;
  }

  /**
   * Runs a transcription pass over the buffered audio, dispatches diffed events
   * to listeners, and returns the current transcript snapshot.
   *
   * A pass is only worth making once there is enough new audio to say something
   * new, so one that comes too soon returns the last snapshot instead of
   * entering the engine. Callers are expected to ask on every chunk their audio
   * source produces — an AudioWorklet hands over 128 frames at a time, which is
   * some 375 times a second — and a pass costs far more than that budget: the
   * engine holds back below `transcription_interval` anyway, but with speakers
   * enabled it still re-clips every diarization turn onto every line before it
   * does so, which is work that grows for as long as the meeting does. Left
   * ungoverned that is what puts a live transcript further and further behind
   * the audio. Pass {@link TranscribeFlags.ForceUpdate} to insist.
   *
   * The interval is a floor, not a cadence: a pass has to cover at least as much
   * audio as the last one took to make. Most of what a pass costs is not the
   * audio in it — measured on the tiny model with speakers, 102ms of a pass goes
   * on getting started and 269ms on each second of audio it looks at — so asking
   * twice a second pays that overhead twice a second, and a machine that cannot
   * quite afford it does not fall behind by a fixed amount, it falls behind
   * further every pass. Replayed as a live session, a machine three times slower
   * than the one those numbers came from ends a three-minute meeting 82 seconds
   * behind and is still delivering lines 80 seconds after the audio stopped.
   * Making a pass earn its keep turns that into batch behaviour instead: passes
   * grow until each one covers its own cost, which is 1.6s of audio at a time on
   * that same machine, and the transcript stays within a pass or two of the
   * speaker. It buys headroom rather than working miracles — nothing can be done
   * for a machine that cannot transcribe a second of audio in a second — and
   * where there is headroom to spare the floor governs and nothing changes.
   */
  transcribe(flags: TranscribeFlags = TranscribeFlags.None): Transcript {
    const forced = (flags & TranscribeFlags.ForceUpdate) !== 0;
    const needed = Math.min(
      Math.max(this.updateInterval, this.lastPass),
      this.updateInterval * MAX_INTERVAL_FACTOR,
    );
    if (!forced && this.pending < needed) return this.latest;
    this.pending = 0;

    let transcript: Transcript;
    const started = performance.now();
    try {
      transcript = normalizeTranscript(this.raw.transcribe(flags));
    } catch (err) {
      const wrapped = toMoonshineError(err);
      dispatchError(this.listeners, wrapped);
      throw wrapped;
    } finally {
      // What the engine cost, not what the listeners went on to do with it:
      // drawing the words is the caller's own budget to keep, and a page that
      // spends it unwisely should not be answered by transcribing less often.
      this.lastPass = (performance.now() - started) / 1000;
    }
    this.latest = transcript;
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
