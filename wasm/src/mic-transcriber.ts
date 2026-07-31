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

/** Byte counts behind a progress report, covering the whole model. */
export interface DownloadProgress {
  /** Bytes fetched so far across every file the model needs. */
  loaded: number;
  /**
   * Total bytes the model needs, from the sizes its manifest declares.
   * Undefined when fetching files of unknown size, in which case `fraction`
   * is meaningless and the download should be shown as indeterminate.
   */
  total?: number;
}

/**
 * Download progress: a `0..1` fraction of the *entire* model, the file
 * currently in flight, and the underlying byte counts.
 */
export type ProgressCallback = (
  fraction: number,
  file: string,
  progress?: DownloadProgress,
) => void;

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
    // A context created without user activation starts suspended, and a
    // suspended context never pulls audio: the worklet is installed, the graph
    // is connected, no error is raised, and not a single sample arrives. Since
    // start() is reached through several awaits (model load, getUserMedia) the
    // originating gesture can be far enough back for Chrome to withhold it, so
    // resume explicitly rather than assuming.
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
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
    // Asking for one input channel makes WebAudio downmix a multi-channel
    // source for us, so this path needs no equivalent of the worklet's
    // downmixToMono.
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
 * `(loaded, total, file)` shape the downloader reports. The downloader counts
 * bytes across the whole model, so the fraction is a true overall percentage
 * whenever the manifest declared its sizes.
 */
export function wrapProgress(
  callback: ProgressCallback | undefined,
): ((loaded: number, total: number | undefined, file: string) => void) | undefined {
  if (!callback) return undefined;
  return (loaded, total, file) => {
    callback(total ? Math.min(1, loaded / total) : 0, file, { loaded, total });
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

/**
 * Averages capture channels into one mono buffer.
 *
 * Averaging rather than taking the first channel matters: plenty of capture
 * devices (USB headsets, audio interfaces, dock passthroughs) open as stereo
 * with the microphone on the right channel and digital silence on the left.
 * Reading only channel 0 there yields a stream of zeroes, so everything appears
 * to run and nothing is ever transcribed.
 *
 * Always returns a copy, because the worklet reuses its input buffers.
 */
export function downmixToMono(channels: readonly Float32Array[]): Float32Array {
  if (channels.length === 1) return new Float32Array(channels[0]);
  const frames = channels[0].length;
  const mixed = new Float32Array(frames);
  for (let c = 0; c < channels.length; c++) {
    const channel = channels[c];
    for (let i = 0; i < frames; i++) mixed[i] += channel[i];
  }
  for (let i = 0; i < frames; i++) mixed[i] /= channels.length;
  return mixed;
}

/**
 * AudioWorklet that downmixes the input to mono and forwards it to the main
 * thread. A worklet gets its own global scope, so {@link downmixToMono} is
 * inlined by source rather than imported — that way the function under test is
 * literally the one that runs.
 */
const CAPTURE_WORKLET_SOURCE = `
${downmixToMono.toString()}

class MoonshineCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length && input[0]) {
      this.port.postMessage(downmixToMono(input));
    }
    return true;
  }
}
registerProcessor('moonshine-capture', MoonshineCaptureProcessor);
`;
