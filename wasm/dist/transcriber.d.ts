/**
 * High-level speech-to-text entry point, mirroring the Python/Swift/Android
 * `Transcriber`. Load it with the async {@link Transcriber.load} factory (which
 * fetches the model from the CDN), then either transcribe a whole buffer or
 * drive a streaming {@link Stream}.
 */
import { AssetDownloader } from './asset-downloader.js';
import { ModelArch, TranscribeFlags } from './enums.js';
import { type LoadModuleOptions, type MoonshineModule } from './module.js';
import { Stream } from './stream.js';
import type { TranscriptEventListener } from './events.js';
import { type Transcript } from './types.js';
/** Load a transcriber from raw in-memory model bytes. */
export interface TranscriberFromBytes {
    encoder: Uint8Array;
    decoder: Uint8Array;
    tokenizer: Uint8Array;
    /** Optional spelling-CNN for alphanumeric fusion (SpellingMode). */
    spelling?: Uint8Array;
    modelArch?: ModelArch;
}
/** Load a transcriber by fetching a model from the CDN by language. */
export interface TranscriberFromCatalog {
    /** Language code (e.g. `"en"`) or English name (e.g. `"English"`). */
    language: string;
    modelArch?: ModelArch;
    /** Also fetch + load the spelling model if one is published. */
    includeSpelling?: boolean;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}
export type TranscriberLoadOptions = (TranscriberFromBytes | TranscriberFromCatalog) & {
    /** Options forwarded to the WASM module loader. */
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
};
export declare class Transcriber {
    private readonly raw;
    private readonly module;
    private defaultStream;
    private closed;
    private constructor();
    /**
     * Loads a transcriber. Either pass raw model bytes, or a `language` to fetch
     * the model from the Moonshine CDN (cached for next time).
     */
    static load(options: TranscriberLoadOptions): Promise<Transcriber>;
    /** Transcribes a complete buffer of PCM audio (non-streaming). */
    transcribe(audio: Float32Array, options?: {
        sampleRate?: number;
        flags?: TranscribeFlags;
    }): Transcript;
    /** Creates a new streaming session. */
    createStream(options?: {
        flags?: TranscribeFlags;
    }): Stream;
    private ensureDefaultStream;
    addListener(listener: TranscriptEventListener): void;
    removeAllListeners(): void;
    start(): void;
    addAudio(audio: Float32Array, sampleRate: number, flags?: TranscribeFlags): void;
    stop(): void;
    /** Architecture-name helper for logging/UX. */
    archName(arch: ModelArch): string;
    close(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=transcriber.d.ts.map