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
import { wrapErrors } from './errors.js';
import { loadMoonshineModule, } from './module.js';
import { normalizeIntentMatches } from './types.js';
let stagedModelCounter = 0;
export class IntentRecognizer {
    raw;
    constructor(raw) {
        this.raw = raw;
    }
    static async load(options = {}) {
        const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
        if (!module.FS) {
            throw new Error('Intent recognition needs the Emscripten filesystem; rebuild with -sFORCE_FILESYSTEM=1.');
        }
        const arch = options.modelArch ?? EmbeddingModelArch.Gemma300M;
        const downloader = options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
        const manifest = module.intentDependencies(options.modelName ?? '', options.variant ?? '');
        const files = await downloader.downloadManifest(manifest);
        const dir = `/moonshine-intent-${stagedModelCounter++}`;
        stageFiles(module, dir, files);
        const raw = wrapErrors(() => new module.IntentRecognizer(dir, arch, options.variant ?? ''));
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
/** Writes files into a fresh MEMFS directory (basename keys only). */
function stageFiles(module, dir, files) {
    const fs = module.FS;
    if (!fs.analyzePath(dir).exists)
        fs.mkdir(dir);
    for (const [name, bytes] of files) {
        fs.writeFile(`${dir}/${name}`, bytes);
    }
}
//# sourceMappingURL=intent-recognizer.js.map