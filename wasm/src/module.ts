/**
 * Loads and caches the Emscripten module produced by the core build
 * (`moonshine.mjs` + `moonshine.wasm`). Everything else in the binding goes
 * through the singleton returned by {@link loadMoonshineModule}.
 */

import { toMoonshineError } from './errors.js';

/** Raw embind class/function surface exported by moonshine.mjs. */
export interface MoonshineModule {
  Transcriber: new (
    encoder: Uint8Array,
    decoder: Uint8Array,
    tokenizer: Uint8Array,
    spelling: Uint8Array | undefined,
    modelArch: number,
  ) => RawTranscriber;
  Stream: new (transcriber: RawTranscriber, flags: number) => RawStream;
  IntentRecognizer: new (
    modelPath: string,
    modelArch: number,
    modelVariant: string,
  ) => RawIntentRecognizer;
  TextToSpeech?: new (
    language: string,
    keys: string[],
    buffers: Uint8Array[],
  ) => RawTextToSpeech;
  GraphemeToPhonemizer?: new (
    language: string,
    keys: string[],
    buffers: Uint8Array[],
  ) => RawGraphemeToPhonemizer;
  /** Emscripten virtual filesystem (exported via -sEXPORTED_RUNTIME_METHODS). */
  FS?: EmscriptenFS;
  version(): number;
  sttDependencies(language: string, modelArch: string): string;
  intentDependencies(modelName: string, variant: string): string;
  ttsDependencies?(languages: string, voice: string): string;
  ttsVoices?(languages: string): string;
  g2pDependencies?(languages: string): string;
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

export interface RawIntentRecognizer {
  registerIntent(phrase: string, priority: number): void;
  unregisterIntent(phrase: string): void;
  clearIntents(): void;
  closestIntents(utterance: string, threshold: number): any;
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

/** Minimal subset of the Emscripten FS API we use to stage model files. */
export interface EmscriptenFS {
  mkdir(path: string): void;
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
  analyzePath(path: string): { exists: boolean };
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
        const moduleArgs: Record<string, unknown> = {};
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
