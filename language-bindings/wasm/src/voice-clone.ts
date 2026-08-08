/**
 * Captures the short reference clip that zero-shot voice cloning needs.
 *
 * ```ts
 * const clone = tts.startCloning();
 * clone.onReady(() => status.textContent = 'Got it — you can stop talking.');
 * await clone.fromMicrophone();
 * await tts.cloneFrom(clone);
 * ```
 *
 * Finding a usable clip means locating a window of the recording that is mostly
 * speech rather than silence or breathing. That search runs in the core (see
 * `moonshine_extract_speech_clip`), so the browser, iOS and Android bindings
 * all agree on what a good clip looks like. Extract stays VAD-only; ZipVoice
 * clone ASR refine + transcript happen once inside {@link TextToSpeech.cloneFrom}.
 */

import { resampleTo16k } from './mic-transcriber.js';
import { loadMoonshineModule, type MoonshineModule } from './module.js';

const TARGET_SAMPLE_RATE = 16000;
/** How much new audio to accumulate between speech searches. */
const SEARCH_INTERVAL_SECONDS = 0.25;
/** Give up looking for a good window after this much recording. */
const DEFAULT_MAX_RECORD_SECONDS = 20;

export interface VoiceCloneOptions {
  /** Length of the reference clip in seconds. Defaults to 4. */
  clipSeconds?: number;
  /** How much of the clip has to be speech. Defaults to 2. */
  minimumSpeechSeconds?: number;
}

export class VoiceClone {
  private readonly module: MoonshineModule;
  private readonly ttsHandle: number;
  private readonly clipSeconds: number;
  private readonly minimumSpeechSeconds: number;

  private chunks: Float32Array[] = [];
  private sampleCount = 0;
  private samplesSinceSearch = 0;
  private clip?: Float32Array;
  private clipTranscript?: string;
  private speech = 0;
  private readyCallbacks: Array<() => void> = [];
  private progressCallbacks: Array<
    (recordedSeconds: number, speechSeconds: number) => void
  > = [];
  private stopCapture?: () => Promise<void>;

  constructor(
    module: MoonshineModule,
    ttsHandle: number,
    options: VoiceCloneOptions = {},
  ) {
    this.module = module;
    this.ttsHandle = ttsHandle;
    this.clipSeconds = options.clipSeconds ?? 4;
    this.minimumSpeechSeconds = options.minimumSpeechSeconds ?? 2;
    if (!module.extractSpeechClip) {
      throw new Error(
        'This Moonshine WASM build does not support speech clip extraction.',
      );
    }
  }

  /** Fires once, as soon as enough speech has been captured. */
  onReady(callback: () => void): this {
    if (this.clip) {
      callback();
    } else {
      this.readyCallbacks.push(callback);
    }
    return this;
  }

  /** Reports how long the caller has been recording and how much was speech. */
  onProgress(
    callback: (recordedSeconds: number, speechSeconds: number) => void,
  ): this {
    this.progressCallbacks.push(callback);
    return this;
  }

  /** True once {@link audio} holds a usable reference clip. */
  get isReady(): boolean {
    return this.clip !== undefined;
  }

  /** Speech found in the best window so far, in seconds. */
  get speechSeconds(): number {
    return this.speech;
  }

  /** Transcript is unused for VAD capture; cloneFrom fills it via create-time ASR. */
  get transcript(): string | undefined {
    return this.clipTranscript;
  }

  get recordedSeconds(): number {
    return this.sampleCount / TARGET_SAMPLE_RATE;
  }

  /** The captured clip (16 kHz mono), or `undefined` until {@link isReady}. */
  get audio(): Float32Array | undefined {
    return this.clip;
  }

  get sampleRate(): number {
    return TARGET_SAMPLE_RATE;
  }

  /**
   * Feeds captured audio in. Call this from your own audio pipeline; the search
   * for a usable window runs a few times a second rather than on every chunk.
   */
  addAudio(pcm: Float32Array, sampleRate: number): void {
    if (this.clip || pcm.length === 0) return;
    const resampled = resampleTo16k(pcm, sampleRate);
    this.chunks.push(resampled);
    this.sampleCount += resampled.length;
    this.samplesSinceSearch += resampled.length;
    if (
      this.samplesSinceSearch <
      SEARCH_INTERVAL_SECONDS * TARGET_SAMPLE_RATE
    ) {
      return;
    }
    this.samplesSinceSearch = 0;
    this.search();
  }

  /**
   * Opens the microphone and records until there is enough speech, or until
   * `maxSeconds` have passed. Resolves with the clip, which is also available
   * from {@link audio}.
   */
  async fromMicrophone(
    options: {
      maxSeconds?: number;
      audioConstraints?: MediaTrackConstraints | boolean;
    } = {},
  ): Promise<Float32Array> {
    const maxSeconds = options.maxSeconds ?? DEFAULT_MAX_RECORD_SECONDS;
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: options.audioConstraints ?? true,
    });
    const context = new AudioContext();
    // A context created without user activation starts suspended, and never
    // pulls audio: recording would appear to run and capture nothing but
    // silence until maxSeconds expired.
    if (context.state === 'suspended') await context.resume();
    const inputRate = context.sampleRate;
    const source = context.createMediaStreamSource(mediaStream);
    const node = context.createScriptProcessor(4096, 1, 1);

    let finish: (clip: Float32Array) => void;
    let fail: (err: Error) => void;
    const done = new Promise<Float32Array>((resolve, reject) => {
      finish = resolve;
      fail = reject;
    });

    this.stopCapture = async () => {
      node.onaudioprocess = null;
      node.disconnect();
      source.disconnect();
      mediaStream.getTracks().forEach((t) => t.stop());
      await context.close();
      this.stopCapture = undefined;
    };

    node.onaudioprocess = (event) => {
      this.addAudio(
        new Float32Array(event.inputBuffer.getChannelData(0)),
        inputRate,
      );
      if (this.clip) {
        finish(this.clip);
      } else if (this.recordedSeconds >= maxSeconds) {
        // Out of patience: take the best window we have, even a quiet one.
        this.search({ acceptAnything: true });
        if (this.clip) {
          finish(this.clip);
        } else {
          fail(
            new Error(
              `No speech detected in ${maxSeconds}s of recording. Try again somewhere quieter.`,
            ),
          );
        }
      }
    };

    source.connect(node);
    node.connect(context.destination);

    try {
      return await done;
    } finally {
      await this.stopCapture?.();
    }
  }

  /** Stops an in-flight {@link fromMicrophone} capture. */
  async cancel(): Promise<void> {
    await this.stopCapture?.();
  }

  /** Throws away everything captured so far. */
  reset(): void {
    this.chunks = [];
    this.sampleCount = 0;
    this.samplesSinceSearch = 0;
    this.clip = undefined;
    this.clipTranscript = undefined;
    this.speech = 0;
  }

  private search(options: { acceptAnything?: boolean } = {}): void {
    const recording = concat(this.chunks, this.sampleCount);
    const result = this.module.extractSpeechClip!(
      recording,
      TARGET_SAMPLE_RATE,
      this.ttsHandle,
      this.clipSeconds,
      options.acceptAnything ? 0 : this.minimumSpeechSeconds,
    );
    this.speech = result.speechDuration;
    for (const cb of this.progressCallbacks) {
      cb(this.recordedSeconds, this.speech);
    }
    if (result.isComplete && result.audio && result.audio.length > 0) {
      this.clip = result.audio;
      this.clipTranscript = result.transcript || undefined;
      const callbacks = this.readyCallbacks;
      this.readyCallbacks = [];
      for (const cb of callbacks) cb();
    }
  }
}

/**
 * Pulls a reference clip out of already-recorded audio, without any capture
 * loop. Used by `TextToSpeech.cloneFrom` when given a file or buffer.
 */
export async function extractSpeechClip(
  audio: Float32Array,
  sampleRate: number,
  options: VoiceCloneOptions & {
    module?: MoonshineModule;
    ttsHandle: number;
  },
): Promise<Float32Array> {
  const module = options.module ?? (await loadMoonshineModule());
  if (!module.extractSpeechClip) {
    throw new Error(
      'This Moonshine WASM build does not support speech clip extraction.',
    );
  }
  const clipSeconds = options.clipSeconds ?? 4;
  const result = module.extractSpeechClip(
    audio,
    sampleRate,
    options.ttsHandle,
    clipSeconds,
    options.minimumSpeechSeconds ?? 2,
  );
  if (result.isComplete && result.audio && result.audio.length > 0) {
    return result.audio;
  }
  // Nothing clearly speech-like. Rather than fail outright, fall back to the
  // best window the detector found, or the start of the recording if it found
  // nothing at all — a poor clone beats no clone for a caller who explicitly
  // handed us this audio.
  const resampled = resampleTo16k(audio, sampleRate);
  const wanted = Math.round(clipSeconds * TARGET_SAMPLE_RATE);
  if (resampled.length <= wanted) return resampled;
  const from = Math.min(
    Math.max(0, Math.round(result.startTime * TARGET_SAMPLE_RATE)),
    resampled.length - wanted,
  );
  return resampled.slice(from, from + wanted);
}

function concat(chunks: Float32Array[], total: number): Float32Array {
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
