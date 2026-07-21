/**
 * Text-to-speech (Phase 2), mirroring the Python/Swift `TextToSpeech`. Loads
 * vocoder + G2P assets into the WASM synthesizer and returns / plays mono PCM.
 *
 * The TTS dependency manifest (`ttsDependencies`) is a flat list of canonical
 * asset *keys* (e.g. `kokoro/model.onnx`) without a base URL, so callers either
 * supply assets in memory directly, or provide an `assetBaseUrl` we prefix each
 * key with to fetch from the CDN.
 */
import { AssetDownloader } from './asset-downloader.js';
import { type LoadModuleOptions, type MoonshineModule } from './module.js';
import type { TtsSynthesisResult } from './types.js';
export interface TtsFromAssets {
    language: string;
    /** Map of canonical key (e.g. `kokoro/model.onnx`) -> bytes. */
    assets: Map<string, Uint8Array>;
}
export interface TtsFromCatalog {
    language: string;
    /** Comma-separated languages passed to the manifest helper (default: language). */
    languages?: string;
    voice?: string;
    /** Base URL prepended to each canonical asset key. */
    assetBaseUrl?: string;
    downloader?: AssetDownloader;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}
export type TextToSpeechOptions = (TtsFromAssets | TtsFromCatalog) & {
    moduleOptions?: LoadModuleOptions;
    module?: MoonshineModule;
};
export declare class TextToSpeech {
    private readonly raw;
    private constructor();
    static load(options: TextToSpeechOptions): Promise<TextToSpeech>;
    /** Synthesizes `text` to mono PCM. */
    say(text: string): TtsSynthesisResult;
    /**
     * Synthesizes and plays `text` through WebAudio, resolving when playback
     * finishes. Pass an existing AudioContext to reuse one.
     */
    speak(text: string, audioContext?: AudioContext): Promise<void>;
    close(): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=text-to-speech.d.ts.map