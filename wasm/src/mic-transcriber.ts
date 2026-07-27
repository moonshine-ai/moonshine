/**
 * Live microphone transcription.
 *
 * ```ts
 * const mic = new MicTranscriber()
 *   .onText((text) => renderLive(text))
 *   .onLine((line) => appendFinal(line.text));
 *
 * await mic.load();
 * await mic.start();
 * ```
 *
 * Captures microphone audio via WebAudio (an AudioWorklet where available,
 * falling back to a ScriptProcessorNode), downmixes to mono, resamples to
 * 16 kHz, and feeds a streaming {@link Stream}. Everything except the choice of
 * language is optional.
 */

import { AssetDownloader } from './asset-downloader.js';
import { ModelArch, TranscribeFlags } from './enums.js';
import type { TranscriptEventListener } from './events.js';
import { Stream } from './stream.js';
import { Transcriber, type TranscriberLoadOptions } from './transcriber.js';
import type { TranscriptLine } from './types.js';

const TARGET_SAMPLE_RATE = 16000;

/** Download progress, as a `0..1` fraction plus the file being fetched. */
export type ProgressCallback = (fraction: number, file: string) => void;

export class MicTranscriber {
  private transcriber?: Transcriber;
  private ownsTranscriber = true;

  private languageCode = 'en';
  private arch: ModelArch = ModelArch.MediumStreaming;
  private modelSource?: string | Record<string, string>;
  private constraints: MediaTrackConstraints | boolean = true;
  private flags: TranscribeFlags = TranscribeFlags.None;

  private readonly listeners: TranscriptEventListener[] = [];
  private textCallbacks: Array<(text: string) => void> = [];
  private lineCallbacks: Array<(line: TranscriptLine) => void> = [];
  private errorCallbacks: Array<(error: Error) => void> = [];
  private progressCallback?: ProgressCallback;

  private mediaStream?: MediaStream;
  private audioContext?: AudioContext;
  private sourceNode?: MediaStreamAudioSourceNode;
  private workletNode?: AudioWorkletNode;
  private scriptNode?: ScriptProcessorNode;
  private stream?: Stream;
  private running = false;
  private muted = false;

  /** Speech-to-text language. Defaults to `"en"`. */
  language(code: string): this {
    this.languageCode = code;
    return this;
  }

  /** Overrides the streaming model. Defaults to the best one for the language. */
  modelArch(arch: ModelArch): this {
    this.arch = arch;
    return this;
  }

  /**
   * Loads the model from somewhere other than the Moonshine CDN: either a base
   * URL that the canonical filenames are appended to (`'/models/'`), or one URL
   * per file (`{'encoder_model.ort': '/models/enc.ort'}`).
   */
  modelsFrom(source: string | Record<string, string>): this {
    this.modelSource = source;
    return this;
  }

  /** Reuses an already-loaded transcriber rather than loading another. */
  useTranscriber(transcriber: Transcriber): this {
    this.transcriber = transcriber;
    this.ownsTranscriber = false;
    return this;
  }

  /** Constraints passed to `getUserMedia({ audio })`. */
  audioConstraints(constraints: MediaTrackConstraints | boolean): this {
    this.constraints = constraints;
    return this;
  }

  /** Enables alphanumeric spelling mode for dictating codes and serial numbers. */
  spellingMode(enabled = true): this {
    this.flags = enabled
      ? this.flags | TranscribeFlags.SpellingMode
      : this.flags & ~TranscribeFlags.SpellingMode;
    return this;
  }

  /** Called with the in-progress text of the line currently being spoken. */
  onText(callback: (text: string) => void): this {
    this.textCallbacks.push(callback);
    return this;
  }

  /** Called once per finished line. */
  onLine(callback: (line: TranscriptLine) => void): this {
    this.lineCallbacks.push(callback);
    return this;
  }

  onError(callback: (error: Error) => void): this {
    this.errorCallbacks.push(callback);
    return this;
  }

  /** Model download progress, as a `0..1` fraction. */
  onProgress(callback: ProgressCallback): this {
    this.progressCallback = callback;
    return this;
  }

  /**
   * Attaches a full {@link TranscriptEventListener}, for applications that need
   * line ids, speaker spans, or word timings rather than just the text.
   */
  addListener(listener: TranscriptEventListener): this {
    this.listeners.push(listener);
    this.stream?.addListener(listener);
    return this;
  }

  /** Downloads the model if needed and prepares the transcriber. */
  async load(): Promise<this> {
    if (this.transcriber) return this;
    const onProgress = wrapProgress(this.progressCallback);
    if (this.modelSource && typeof this.modelSource === 'object') {
      this.transcriber = await Transcriber.loadFromUrls(this.modelSource, {
        modelArch: this.arch,
        onProgress,
      });
      return this;
    }
    this.transcriber = await Transcriber.load({
      language: this.languageCode,
      modelArch: this.arch,
      // A base URL is applied by pointing the downloader somewhere else; the
      // catalog still decides which files this architecture needs.
      downloader: this.modelSource
        ? new AssetDownloader({ onProgress, baseUrl: this.modelSource })
        : undefined,
      onProgress,
    } as TranscriberLoadOptions);
    return this;
  }

  /**
   * Opens the microphone and starts transcribing. Loads the model first if
   * {@link load} has not already been called.
   */
  async start(): Promise<void> {
    if (this.running) return;
    await this.load();
    this.running = true;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: this.constraints,
      });
    } catch (err) {
      this.running = false;
      this.emitError(err);
      throw err;
    }

    this.audioContext = new AudioContext();
    const inputSampleRate = this.audioContext.sampleRate;
    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    this.stream = this.transcriber!.createStream({ flags: this.flags });
    this.stream.addListener(this.namedCallbackListener());
    for (const listener of this.listeners) this.stream.addListener(listener);
    this.stream.start();

    const onChunk = (chunk: Float32Array) => {
      if (!this.running || this.muted || !this.stream) return;
      const resampled = resampleTo16k(chunk, inputSampleRate);
      this.stream.addAudio(resampled, TARGET_SAMPLE_RATE, this.flags);
      this.stream.transcribe(this.flags);
    };

    if (this.audioContext.audioWorklet) {
      await this.setupWorklet(onChunk);
    } else {
      this.setupScriptProcessor(onChunk);
    }
  }

  /** Stops capture, flushes a final transcript, and releases audio resources. */
  async stop(): Promise<void> {
    if (!this.running) return;
    this.running = false;

    this.workletNode?.disconnect();
    this.scriptNode?.disconnect();
    this.sourceNode?.disconnect();
    this.mediaStream?.getTracks().forEach((t) => t.stop());
    this.stream?.stop();
    await this.audioContext?.close();

    this.workletNode = undefined;
    this.scriptNode = undefined;
    this.sourceNode = undefined;
    this.mediaStream = undefined;
    this.audioContext = undefined;
  }

  /**
   * Drops incoming audio without tearing down the microphone. Used to stop the
   * assistant transcribing its own synthesized speech.
   */
  mute(muted = true): void {
    this.muted = muted;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /** Releases the stream, and the transcriber unless one was supplied. */
  close(): void {
    this.stream?.close();
    this.stream = undefined;
    if (this.ownsTranscriber) this.transcriber?.close();
    this.transcriber = undefined;
  }

  private namedCallbackListener(): TranscriptEventListener {
    return {
      onLineTextChanged: (event) => {
        for (const cb of this.textCallbacks) cb(event.line.text);
      },
      onLineCompleted: (event) => {
        for (const cb of this.lineCallbacks) cb(event.line);
      },
      onError: (event) => this.emitError(event.error),
    };
  }

  private emitError(err: unknown): void {
    const error = err instanceof Error ? err : new Error(String(err));
    for (const cb of this.errorCallbacks) cb(error);
  }

  private async setupWorklet(onChunk: (c: Float32Array) => void): Promise<void> {
    const ctx = this.audioContext!;
    const url = URL.createObjectURL(
      new Blob([CAPTURE_WORKLET_SOURCE], { type: 'application/javascript' }),
    );
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    this.workletNode = new AudioWorkletNode(ctx, 'moonshine-capture');
    this.workletNode.port.onmessage = (event) => onChunk(event.data as Float32Array);
    this.sourceNode!.connect(this.workletNode);
    // Keep the graph alive without producing output.
    this.workletNode.connect(ctx.destination);
  }

  private setupScriptProcessor(onChunk: (c: Float32Array) => void): void {
    const ctx = this.audioContext!;
    this.scriptNode = ctx.createScriptProcessor(4096, 1, 1);
    this.scriptNode.onaudioprocess = (event) => {
      onChunk(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    this.sourceNode!.connect(this.scriptNode);
    this.scriptNode.connect(ctx.destination);
  }
}

/**
 * Adapts the `0..1` fraction callbacks the public API uses onto the
 * `(loaded, total, file)` shape the downloader reports.
 */
export function wrapProgress(
  callback: ProgressCallback | undefined,
): ((loaded: number, total: number | undefined, file: string) => void) | undefined {
  if (!callback) return undefined;
  return (loaded, total, file) => {
    callback(total ? Math.min(1, loaded / total) : 0, file);
  };
}

/** Simple linear resampler to 16 kHz mono. */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const outLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? 0;
    const b = input[idx + 1] ?? a;
    output[i] = a + (b - a) * frac;
  }
  return output;
}

/** AudioWorklet that forwards mono input frames to the main thread. */
const CAPTURE_WORKLET_SOURCE = `
class MoonshineCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // Copy: the underlying buffer is reused by the engine.
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}
registerProcessor('moonshine-capture', MoonshineCaptureProcessor);
`;
