/**
 * Loads and caches the Emscripten module produced by the core build
 * (`moonshine.mjs` + `moonshine.wasm`). Everything else in the binding goes
 * through the singleton returned by {@link loadMoonshineModule}.
 */

import { toMoonshineError } from './errors.js';

/** Raw embind class/function surface exported by moonshine.mjs. */
export interface MoonshineModule {
  Transcriber: new (
    keys: string[],
    buffers: Uint8Array[],
    modelArch: number,
    optionNames: string[],
    optionValues: string[],
  ) => RawTranscriber;
  Stream: new (transcriber: RawTranscriber, flags: number) => RawStream;
  EmbeddingModel: new (
    keys: string[],
    buffers: Uint8Array[],
    modelArch: number,
    modelVariant: string,
  ) => RawEmbeddingModel;
  TextToSpeech?: new (
    language: string,
    keys: string[],
    buffers: Uint8Array[],
    optionNames: string[],
    optionValues: string[],
  ) => RawTextToSpeech;
  GraphemeToPhonemizer?: new (
    language: string,
    keys: string[],
    buffers: Uint8Array[],
  ) => RawGraphemeToPhonemizer;
  version(): number;
  sttDependencies(
    language: string,
    modelArch: string,
    includeSpelling: boolean,
  ): string;
  embeddingDependencies(modelName: string, variant: string): string;
  ttsDependencies?(languages: string, voice: string): string;
  ttsVoices?(
    languages: string,
    optionNames: string[],
    optionValues: string[],
  ): string;
  g2pDependencies?(languages: string): string;
  extractSpeechClip?(
    audio: Float32Array,
    sampleRate: number,
    clipDurationSeconds: number,
    minimumSpeechSeconds: number,
  ): RawSpeechClip;
}

/** Result of {@link MoonshineModule.extractSpeechClip}. */
export interface RawSpeechClip {
  /** 16 kHz mono PCM; `undefined` until `isComplete`. */
  audio?: Float32Array;
  startTime: number;
  speechDuration: number;
  isComplete: boolean;
}

export interface RawTranscriber {
  transcribe(audio: Float32Array, sampleRate: number, flags: number): any;
  close(): void;
}

export interface RawStream {
  start(): void;
  stop(): void;
  addAudio(audio: Float32Array, sampleRate: number, flags: number): void;
  transcribe(flags: number): any;
  close(): void;
}

export interface RawEmbeddingModel {
  calculateEmbedding(sentence: string): Float32Array;
  distance(embeddingA: Float32Array, embeddingB: Float32Array): number;
  close(): void;
}

export interface RawTextToSpeech {
  say(text: string): { audio: Float32Array; sampleRate: number };
  close(): void;
}

export interface RawGraphemeToPhonemizer {
  textToPhonemes(text: string): string;
  close(): void;
}

/** Options for {@link loadMoonshineModule}. */
export interface LoadModuleOptions {
  /**
   * Override how the `.wasm` (and worker) files are located. Useful when the
   * generated `moonshine.mjs` is served from a different path than the `.wasm`.
   */
  locateFile?: (path: string, scriptDirectory: string) => string;
  /** Provide the Emscripten factory directly (e.g. a custom bundling setup). */
  factory?: EmscriptenFactory;
}

type EmscriptenFactory = (opts?: Record<string, unknown>) => Promise<MoonshineModule>;

let cached: Promise<MoonshineModule> | undefined;

/**
 * ONNX Runtime warns "Unknown CPU vendor" the first time it builds a session
 * because browsers don't expose the host CPU vendor. The value only feeds
 * execution-provider device metadata, which the wasm CPU backend ignores.
 * Upstream stopped emitting it for wasm in ORT 1.24.3
 * (microsoft/onnxruntime#27399); drop this once the vendored ORT is newer.
 */
const SUPPRESSED_STDERR = /Unknown CPU vendor\. cpuinfo_vendor value:/;

function printErr(...args: unknown[]): void {
  if (typeof args[0] === 'string' && SUPPRESSED_STDERR.test(args[0])) return;
  console.error(...args);
}

/**
 * Resolves the Emscripten factory. By default it dynamically imports the
 * sibling `./moonshine.mjs` emitted by the build; callers can inject their own
 * via {@link LoadModuleOptions.factory} for non-standard bundling.
 */
async function resolveFactory(
  options: LoadModuleOptions,
): Promise<EmscriptenFactory> {
  if (options.factory) return options.factory;
  // The generated ES module lives next to this file after bundling.
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore - generated at build time, no types.
  const mod = await import('./moonshine.mjs');
  return (mod.default ?? mod) as EmscriptenFactory;
}

/**
 * Loads (and memoizes) the Moonshine WASM module. Safe to call repeatedly; the
 * heavy compile happens once.
 */
export function loadMoonshineModule(
  options: LoadModuleOptions = {},
): Promise<MoonshineModule> {
  if (!cached) {
    cached = (async () => {
      try {
        const factory = await resolveFactory(options);
        const moduleArgs: Record<string, unknown> = { printErr };
        if (options.locateFile) moduleArgs.locateFile = options.locateFile;
        return await factory(moduleArgs);
      } catch (err) {
        cached = undefined; // allow retry on failure
        throw toMoonshineError(err);
      }
    })();
  }
  return cached;
}

/** Clears the cached module (mainly for tests). */
export function resetMoonshineModule(): void {
  cached = undefined;
}
