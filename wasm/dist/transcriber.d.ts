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
/**
 * Load a transcriber from raw in-memory model bytes for a non-streaming model.
 * For streaming models (or word-timestamp/spelling extras) use
 * {@link TranscriberFromFiles} and key each buffer by its canonical filename.
 */
export interface TranscriberFromBytes {
    encoder: Uint8Array;
    decoder: Uint8Array;
    tokenizer: Uint8Array;
    /** Optional spelling-CNN for alphanumeric fusion (SpellingMode). */
    spelling?: Uint8Array;
    modelArch?: ModelArch;
}
/**
 * Load a transcriber from in-memory model buffers keyed by their canonical
 * manifest filename (e.g. `encoder_model.ort`, `decoder_model_merged.ort`,
 * `tokenizer.bin` for non-streaming; `frontend.ort`, `encoder.ort`,
 * `adapter.ort`, `cross_kv.ort`, `decoder_kv.ort`, `streaming_config.json`,
 * `tokenizer.bin` for streaming; plus optional `decoder_with_attention.ort` /
 * `decoder_kv_with_attention.ort` and `spelling_cnn.ort`). This is the general
 * in-memory loader and works for every architecture.
 */
export interface TranscriberFromFiles {
    files: Record<string, Uint8Array> | Map<string, Uint8Array>;
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
export type TranscriberLoadOptions = (TranscriberFromBytes | TranscriberFromFiles | TranscriberFromCatalog) & {
    /** Options forwarded to the WASM module loader. */
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
};
/** Options for {@link Transcriber.loadFromUrls}. */
export interface TranscriberFromUrlsOptions {
    modelArch?: ModelArch;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
}
export declare class Transcriber {
    private readonly raw;
    private readonly module;
    private defaultStream;
    private closed;
    private constructor();
    /**
     * Loads a transcriber. Pass raw non-streaming bytes ({@link
     * TranscriberFromBytes}), a keyed map of model files ({@link
     * TranscriberFromFiles}, which also supports streaming), or a `language` to
     * fetch the model from the Moonshine CDN ({@link TranscriberFromCatalog},
     * cached for next time). All paths load the model purely in memory — the
     * browser has no natural filesystem.
     */
    static load(options: TranscriberLoadOptions): Promise<Transcriber>;
    /**
     * Loads a transcriber from a map of canonical filename -> URL. Downloads each
     * remote file into a buffer (with caching, via {@link AssetDownloader}) and
     * feeds the buffers through the in-memory loader. Convenient when you host the
     * model files yourself instead of using the Moonshine CDN catalog.
     *
     * @example
     * const t = await Transcriber.loadFromUrls({
     *   'encoder_model.ort': '/models/encoder_model.ort',
     *   'decoder_model_merged.ort': '/models/decoder_model_merged.ort',
     *   'tokenizer.bin': '/models/tokenizer.bin',
     * }, { modelArch: ModelArch.Base });
     */
    static loadFromUrls(files: Record<string, string> | Map<string, string>, options?: TranscriberFromUrlsOptions): Promise<Transcriber>;
    /** Builds the raw WASM transcriber from a keyed, in-memory file map. */
    private static construct;
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