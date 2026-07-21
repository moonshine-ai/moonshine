/**
 * Grapheme-to-phoneme conversion (Phase 2), mirroring the Python/Swift
 * `GraphemeToPhonemizer`. Turns text into IPA phonemes using the WASM binding.
 */
import { AssetDownloader } from './asset-downloader.js';
import { wrapErrors } from './errors.js';
import { loadMoonshineModule, } from './module.js';
const DEFAULT_G2P_ASSET_BASE = 'https://download.moonshine.ai/tts';
function isFromAssets(o) {
    return 'assets' in o;
}
export class GraphemeToPhonemizer {
    raw;
    constructor(raw) {
        this.raw = raw;
    }
    static async load(options) {
        const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
        if (!module.GraphemeToPhonemizer) {
            throw new Error('This Moonshine WASM build was compiled without G2P support.');
        }
        let assets;
        if (isFromAssets(options)) {
            assets = options.assets;
        }
        else {
            if (!module.g2pDependencies) {
                throw new Error('G2P manifests are unavailable in this build.');
            }
            const languages = options.languages ?? options.language;
            const keys = JSON.parse(module.g2pDependencies(languages));
            const base = (options.assetBaseUrl ?? DEFAULT_G2P_ASSET_BASE).replace(/\/+$/, '');
            const downloader = options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
            assets = new Map();
            for (const key of keys) {
                assets.set(key, await downloader.fetchFile(`${base}/${key.replace(/^\/+/, '')}`));
            }
        }
        const raw = wrapErrors(() => new module.GraphemeToPhonemizer(options.language, [...assets.keys()], [...assets.values()]));
        return new GraphemeToPhonemizer(raw);
    }
    /** Converts `text` into IPA phonemes. */
    textToPhonemes(text) {
        return wrapErrors(() => this.raw.textToPhonemes(text));
    }
    close() {
        wrapErrors(() => this.raw.close());
    }
    [Symbol.dispose]() {
        this.close();
    }
}
//# sourceMappingURL=grapheme-to-phonemizer.js.map