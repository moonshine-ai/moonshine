/**
 * Fetches model assets from the Moonshine CDN and caches them in the browser,
 * driven by the JSON manifest helpers in the C ABI (so we never re-implement
 * the file/URL layout in JS). Mirrors the download flow of the Python/Swift/
 * Android bindings, adapted to `fetch` + the Cache API.
 */
/** One downloaded asset. */
export interface DownloadedAsset {
    /** The canonical filename (basename), e.g. `encoder_model.ort`. */
    readonly name: string;
    readonly bytes: Uint8Array;
}
export interface AssetDownloaderOptions {
    /** Cache name used with the browser Cache API. */
    cacheName?: string;
    /** Called with (loadedBytes, totalBytes|undefined, currentFile). */
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}
/**
 * Downloads model files with transparent caching. A single instance can be
 * reused across models; entries are keyed by absolute URL.
 */
export declare class AssetDownloader {
    private readonly cacheName;
    private readonly onProgress?;
    constructor(options?: AssetDownloaderOptions);
    /**
     * Downloads every file listed in a `{groups:[...]}` manifest (STT / intent),
     * returning them keyed by canonical filename.
     */
    downloadManifest(manifestJson: string): Promise<Map<string, Uint8Array>>;
    /** Downloads a flat list of URLs, returning bytes keyed by basename. */
    downloadFiles(urls: string[]): Promise<Map<string, Uint8Array>>;
    /** Fetches a single URL, using the Cache API when available. */
    fetchFile(url: string): Promise<Uint8Array>;
    private readWithProgress;
    private openCache;
}
//# sourceMappingURL=asset-downloader.d.ts.map