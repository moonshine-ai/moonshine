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
import { wrapErrors } from './errors.js';
import { loadMoonshineModule, } from './module.js';
const DEFAULT_TTS_ASSET_BASE = 'https://download.moonshine.ai/tts';
function isFromAssets(o) {
    return 'assets' in o;
}
export class TextToSpeech {
    raw;
    constructor(raw) {
        this.raw = raw;
    }
    static async load(options) {
        const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
        if (!module.TextToSpeech) {
            throw new Error('This Moonshine WASM build was compiled without TTS support.');
        }
        let assets;
        if (isFromAssets(options)) {
            assets = options.assets;
        }
        else {
            if (!module.ttsDependencies) {
                throw new Error('TTS manifests are unavailable in this build.');
            }
            const languages = options.languages ?? options.language;
            const keysJson = module.ttsDependencies(languages, options.voice ?? '');
            const keys = JSON.parse(keysJson);
            const base = (options.assetBaseUrl ?? DEFAULT_TTS_ASSET_BASE).replace(/\/+$/, '');
            const downloader = options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
            assets = new Map();
            for (const key of keys) {
                const url = `${base}/${key.replace(/^\/+/, '')}`;
                assets.set(key, await downloader.fetchFile(url));
            }
        }
        const keys = [...assets.keys()];
        const buffers = [...assets.values()];
        const raw = wrapErrors(() => new module.TextToSpeech(options.language, keys, buffers));
        return new TextToSpeech(raw);
    }
    /** Synthesizes `text` to mono PCM. */
    say(text) {
        const result = wrapErrors(() => this.raw.say(text));
        return { audio: result.audio, sampleRate: result.sampleRate };
    }
    /**
     * Synthesizes and plays `text` through WebAudio, resolving when playback
     * finishes. Pass an existing AudioContext to reuse one.
     */
    async speak(text, audioContext) {
        const { audio, sampleRate } = this.say(text);
        const ctx = audioContext ?? new AudioContext();
        try {
            const buffer = ctx.createBuffer(1, audio.length, sampleRate);
            buffer.copyToChannel(audio, 0);
            const source = ctx.createBufferSource();
            source.buffer = buffer;
            source.connect(ctx.destination);
            await new Promise((resolve) => {
                source.onended = () => resolve();
                source.start();
            });
        }
        finally {
            if (!audioContext)
                await ctx.close();
        }
    }
    close() {
        wrapErrors(() => this.raw.close());
    }
    [Symbol.dispose]() {
        this.close();
    }
}
//# sourceMappingURL=text-to-speech.js.map