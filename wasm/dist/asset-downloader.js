/**
 * Fetches model assets from the Moonshine CDN and caches them in the browser,
 * driven by the JSON manifest helpers in the C ABI (so we never re-implement
 * the file/URL layout in JS). Mirrors the download flow of the Python/Swift/
 * Android bindings, adapted to `fetch` + the Cache API.
 */
import { MoonshineDownloadError } from './errors.js';
const DEFAULT_CACHE = 'moonshine-models-v1';
/**
 * Downloads model files with transparent caching. A single instance can be
 * reused across models; entries are keyed by absolute URL.
 */
export class AssetDownloader {
    cacheName;
    onProgress;
    constructor(options = {}) {
        this.cacheName = options.cacheName ?? DEFAULT_CACHE;
        this.onProgress = options.onProgress;
    }
    /**
     * Downloads every file listed in a `{groups:[...]}` manifest (STT / intent),
     * returning them keyed by canonical filename.
     */
    async downloadManifest(manifestJson) {
        let manifest;
        try {
            manifest = JSON.parse(manifestJson);
        }
        catch (err) {
            throw new MoonshineDownloadError(`Failed to parse model manifest: ${err.message}`);
        }
        const out = new Map();
        for (const group of manifest.groups ?? []) {
            for (const file of group.files) {
                const url = joinUrl(group.base_url, file);
                out.set(file, await this.fetchFile(url));
            }
        }
        return out;
    }
    /** Downloads a flat list of URLs, returning bytes keyed by basename. */
    async downloadFiles(urls) {
        const out = new Map();
        for (const url of urls) {
            out.set(basename(url), await this.fetchFile(url));
        }
        return out;
    }
    /** Fetches a single URL, using the Cache API when available. */
    async fetchFile(url) {
        const cache = await this.openCache();
        if (cache) {
            const hit = await cache.match(url);
            if (hit) {
                const buf = await hit.arrayBuffer();
                this.onProgress?.(buf.byteLength, buf.byteLength, basename(url));
                return new Uint8Array(buf);
            }
        }
        const response = await fetch(url);
        if (!response.ok) {
            throw new MoonshineDownloadError(`Failed to download ${url}: ${response.status} ${response.statusText}`);
        }
        if (cache) {
            // Store a clone so the body below can still be read.
            await cache.put(url, response.clone());
        }
        const buf = await this.readWithProgress(response, basename(url));
        return new Uint8Array(buf);
    }
    async readWithProgress(response, file) {
        const total = Number(response.headers.get('content-length')) || undefined;
        if (!response.body || !this.onProgress) {
            const buf = await response.arrayBuffer();
            this.onProgress?.(buf.byteLength, total, file);
            return buf;
        }
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (done)
                break;
            if (value) {
                chunks.push(value);
                loaded += value.byteLength;
                this.onProgress(loaded, total, file);
            }
        }
        const merged = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            merged.set(chunk, offset);
            offset += chunk.byteLength;
        }
        return merged.buffer;
    }
    async openCache() {
        try {
            if (typeof caches !== 'undefined') {
                return await caches.open(this.cacheName);
            }
        }
        catch {
            // Cache API not available (e.g. non-secure context / Node) — skip.
        }
        return undefined;
    }
}
function joinUrl(base, file) {
    return `${base.replace(/\/+$/, '')}/${file.replace(/^\/+/, '')}`;
}
function basename(url) {
    const clean = url.split(/[?#]/)[0];
    const idx = clean.lastIndexOf('/');
    return idx >= 0 ? clean.slice(idx + 1) : clean;
}
//# sourceMappingURL=asset-downloader.js.map