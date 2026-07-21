/**
 * Grapheme-to-phoneme conversion (Phase 2), mirroring the Python/Swift
 * `GraphemeToPhonemizer`. Turns text into IPA phonemes using the WASM binding.
 */
import { AssetDownloader } from './asset-downloader.js';
import { type LoadModuleOptions, type MoonshineModule } from './module.js';
export interface G2pFromAssets {
    language: string;
    assets: Map<string, Uint8Array>;
}
export interface G2pFromCatalog {
    language: string;
    languages?: string;
    assetBaseUrl?: string;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}
export type GraphemeToPhonemizerOptions = (G2pFromAssets | G2pFromCatalog) & {
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
};
export declare class GraphemeToPhonemizer {
    private readonly raw;
    private constructor();
    static load(options: GraphemeToPhonemizerOptions): Promise<GraphemeToPhonemizer>;
    /** Converts `text` into IPA phonemes. */
    textToPhonemes(text: string): string;
    close(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=grapheme-to-phonemizer.d.ts.map