/**
 * Intent recognition, mirroring the Python/Swift `IntentRecognizer`. Registers
 * canonical phrases and finds the closest match to an utterance using the
 * embedding model.
 *
 * The embedding model ships as a single all-in-one `.ort` file (plus
 * `tokenizer.bin`) and is loaded entirely from in-memory buffers via the
 * `moonshine_create_intent_recognizer_from_memory` C ABI — the browser has no
 * natural filesystem, so nothing is staged to disk.
 */
import { AssetDownloader } from './asset-downloader.js';
import { EmbeddingModelArch } from './enums.js';
import { type LoadModuleOptions, type MoonshineModule } from './module.js';
import { type IntentMatch } from './types.js';
export interface IntentFromCatalog {
    /** Embedding model id (e.g. `"embeddinggemma-300m"`). Empty = default. */
    modelName?: string;
    modelArch?: EmbeddingModelArch;
    /** One of "q4", "q8", "fp16", "fp32", "q4f16". Empty = model default. */
    variant?: string;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}
export type IntentRecognizerOptions = IntentFromCatalog & {
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
};
/** Options for {@link IntentRecognizer.loadFromUrls} (self-hosted model files). */
export interface IntentFromUrlsOptions {
    modelArch?: EmbeddingModelArch;
    /** One of "q4", "q8", "fp16", "fp32", "q4f16". Empty = "q4". */
    variant?: string;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
}
/** A phrase to register, with an optional priority for tie-breaking. */
export interface IntentPhrase {
    phrase: string;
    priority?: number;
}
export declare class IntentRecognizer {
    private readonly raw;
    private constructor();
    static load(options?: IntentRecognizerOptions): Promise<IntentRecognizer>;
    /**
     * Loads the embedding model from a caller-supplied map of canonical filename
     * -> URL (e.g. `{ 'model_q4.ort': '...', 'tokenizer.bin': '...' }`), for
     * self-hosting the model files instead of using the Moonshine CDN.
     */
    static loadFromUrls(files: Record<string, string> | Map<string, string>, options?: IntentFromUrlsOptions): Promise<IntentRecognizer>;
    private static construct;
    /** Registers a phrase (optionally many). */
    register(phrases: string | IntentPhrase | Array<string | IntentPhrase>): void;
    unregister(phrase: string): void;
    clear(): void;
    /** Returns registered phrases ranked by similarity to `utterance`. */
    closestIntents(utterance: string, threshold?: number): IntentMatch[];
    /** Convenience: the single best match above `threshold`, or null. */
    bestIntent(utterance: string, threshold?: number): IntentMatch | null;
    close(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=intent-recognizer.d.ts.map