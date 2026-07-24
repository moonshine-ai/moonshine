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
import { wrapErrors } from './errors.js';
import { loadMoonshineModule, } from './module.js';
import { normalizeIntentMatches } from './types.js';
export class IntentRecognizer {
    raw;
    constructor(raw) {
        this.raw = raw;
    }
    static async load(options = {}) {
        const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
        const arch = options.modelArch ?? EmbeddingModelArch.Gemma300M;
        const variant = options.variant ?? '';
        const downloader = options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
        const manifest = module.intentDependencies(options.modelName ?? '', variant);
        const files = await downloader.downloadManifest(manifest);
        return IntentRecognizer.construct(module, files, arch, variant);
    }
    /**
     * Loads the embedding model from a caller-supplied map of canonical filename
     * -> URL (e.g. `{ 'model_q4.ort': '...', 'tokenizer.bin': '...' }`), for
     * self-hosting the model files instead of using the Moonshine CDN.
     */
    static async loadFromUrls(files, options = {}) {
        const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
        const arch = options.modelArch ?? EmbeddingModelArch.Gemma300M;
        const downloader = options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
        const bytes = await downloader.downloadNamedFiles(files);
        return IntentRecognizer.construct(module, bytes, arch, options.variant ?? '');
    }
    static construct(module, files, arch, variant) {
        const keys = [...files.keys()];
        const buffers = keys.map((k) => files.get(k));
        const raw = wrapErrors(() => new module.IntentRecognizer(keys, buffers, arch, variant));
        return new IntentRecognizer(raw);
    }
    /** Registers a phrase (optionally many). */
    register(phrases) {
        const list = Array.isArray(phrases) ? phrases : [phrases];
        for (const item of list) {
            const phrase = typeof item === 'string' ? item : item.phrase;
            const priority = typeof item === 'string' ? 0 : (item.priority ?? 0);
            wrapErrors(() => this.raw.registerIntent(phrase, priority));
        }
    }
    unregister(phrase) {
        wrapErrors(() => this.raw.unregisterIntent(phrase));
    }
    clear() {
        wrapErrors(() => this.raw.clearIntents());
    }
    /** Returns registered phrases ranked by similarity to `utterance`. */
    closestIntents(utterance, threshold = 0) {
        return wrapErrors(() => normalizeIntentMatches(this.raw.closestIntents(utterance, threshold)));
    }
    /** Convenience: the single best match above `threshold`, or null. */
    bestIntent(utterance, threshold = 0) {
        const matches = this.closestIntents(utterance, threshold);
        return matches.length > 0 ? matches[0] : null;
    }
    close() {
        wrapErrors(() => this.raw.close());
    }
    [Symbol.dispose]() {
        this.close();
    }
}
//# sourceMappingURL=intent-recognizer.js.map