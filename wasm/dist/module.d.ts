/**
 * Loads and caches the Emscripten module produced by the core build
 * (`moonshine.mjs` + `moonshine.wasm`). Everything else in the binding goes
 * through the singleton returned by {@link loadMoonshineModule}.
 */
/** Raw embind class/function surface exported by moonshine.mjs. */
export interface MoonshineModule {
    Transcriber: new (keys: string[], buffers: Uint8Array[], modelArch: number) => RawTranscriber;
    Stream: new (transcriber: RawTranscriber, flags: number) => RawStream;
    IntentRecognizer: new (keys: string[], buffers: Uint8Array[], modelArch: number, modelVariant: string) => RawIntentRecognizer;
    TextToSpeech?: new (language: string, keys: string[], buffers: Uint8Array[]) => RawTextToSpeech;
    GraphemeToPhonemizer?: new (language: string, keys: string[], buffers: Uint8Array[]) => RawGraphemeToPhonemizer;
    version(): number;
    sttDependencies(language: string, modelArch: string, includeSpelling: boolean): string;
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
    say(text: string): {
        audio: Float32Array;
        sampleRate: number;
    };
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
/**
 * Loads (and memoizes) the Moonshine WASM module. Safe to call repeatedly; the
 * heavy compile happens once.
 */
export declare function loadMoonshineModule(options?: LoadModuleOptions): Promise<MoonshineModule>;
/** Clears the cached module (mainly for tests). */
export declare function resetMoonshineModule(): void;
export {};
//# sourceMappingURL=module.d.ts.map