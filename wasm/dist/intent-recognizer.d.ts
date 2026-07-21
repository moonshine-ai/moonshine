/**
 * Intent recognition (Phase 3), mirroring the Python/Swift `IntentRecognizer`.
 * Registers canonical phrases and finds the closest match to an utterance using
 * the embedding model.
 *
 * The C ABI only exposes a from-files constructor, so in WASM we stage the
 * downloaded model into Emscripten's in-memory filesystem (MEMFS) and load from
 * that path.
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
/** A phrase to register, with an optional priority for tie-breaking. */
export interface IntentPhrase {
    phrase: string;
    priority?: number;
}
export declare class IntentRecognizer {
    private readonly raw;
    private constructor();
    static load(options?: IntentRecognizerOptions): Promise<IntentRecognizer>;
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