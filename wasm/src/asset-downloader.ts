/**
 * Fetches model assets from the Moonshine CDN and caches them in the browser,
 * driven by the JSON manifest helpers in the C ABI (so we never re-implement
 * the file/URL layout in JS). Mirrors the download flow of the Python/Swift/
 * Android bindings, adapted to `fetch` + the Cache API.
 */

import { MoonshineDownloadError } from './errors.js';

/** One downloaded asset. */
export interface DownloadedAsset {
  /** The canonical filename (basename), e.g. `encoder_model.ort`. */
  readonly name: string;
  readonly bytes: Uint8Array;
}

/**
 * One file in a manifest group. The C ABI emits each `files` entry as an object
 * carrying everything needed to fetch and verify it (not a bare string).
 */
interface ManifestFile {
  /** Canonical filename (basename), e.g. `encoder_model.ort`. */
  name: string;
  /** Fully-qualified download URL (`base_url + "/" + name`). */
  url: string;
  /** Expected size in bytes, or null/absent when unknown. */
  size?: number | null;
  /** Base64 checksum digest, or "" when unknown. */
  checksum?: string;
  /** Checksum algorithm, e.g. "crc32c", or "" when unknown. */
  checksum_type?: string;
}

/** A group of files sharing a base URL, matching the C ABI manifest shape. */
interface ManifestGroup {
  base_url: string;
  files: ManifestFile[];
}

interface Manifest {
  groups: ManifestGroup[];
}

export interface AssetDownloaderOptions {
  /** Cache name used with the browser Cache API. */
  cacheName?: string;
  /** Called with (loadedBytes, totalBytes|undefined, currentFile). */
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
  /**
   * Fetches manifest files from here instead of the base URL the manifest names,
   * so applications can host the assets themselves.
   */
  baseUrl?: string;
}

const DEFAULT_CACHE = 'moonshine-models-v1';

/**
 * Downloads model files with transparent caching. A single instance can be
 * reused across models; entries are keyed by absolute URL.
 */
export class AssetDownloader {
  private readonly cacheName: string;
  private readonly onProgress?: AssetDownloaderOptions['onProgress'];
  private readonly baseUrl?: string;

  constructor(options: AssetDownloaderOptions = {}) {
    this.cacheName = options.cacheName ?? DEFAULT_CACHE;
    this.onProgress = options.onProgress;
    this.baseUrl = options.baseUrl;
  }

  /**
   * Downloads every file listed in a `{groups:[...]}` manifest (STT / intent),
   * returning them keyed by canonical filename.
   */
  async downloadManifest(manifestJson: string): Promise<Map<string, Uint8Array>> {
    let manifest: Manifest;
    try {
      manifest = JSON.parse(manifestJson) as Manifest;
    } catch (err) {
      throw new MoonshineDownloadError(
        `Failed to parse model manifest: ${(err as Error).message}`,
      );
    }
    const out = new Map<string, Uint8Array>();
    for (const group of manifest.groups ?? []) {
      for (const file of group.files) {
        const url = this.baseUrl
          ? joinUrl(this.baseUrl, file.name)
          : (file.url ?? joinUrl(group.base_url, file.name));
        const bytes = await this.fetchFile(url);
        if (
          typeof file.size === 'number' &&
          file.size >= 0 &&
          bytes.byteLength !== file.size
        ) {
          throw new MoonshineDownloadError(
            `Size mismatch for ${file.name}: expected ${file.size} bytes, ` +
              `got ${bytes.byteLength} (from ${url})`,
          );
        }
        out.set(file.name, bytes);
      }
    }
    return out;
  }

  /** Downloads a flat list of URLs, returning bytes keyed by basename. */
  async downloadFiles(urls: string[]): Promise<Map<string, Uint8Array>> {
    const out = new Map<string, Uint8Array>();
    for (const url of urls) {
      out.set(basename(url), await this.fetchFile(url));
    }
    return out;
  }

  /**
   * Downloads a map of canonical filename -> URL, returning bytes keyed by the
   * supplied filename (not the URL basename). Use this when the caller controls
   * the canonical keys, e.g. feeding a transcriber's in-memory loader.
   */
  async downloadNamedFiles(
    files: Record<string, string> | Map<string, string>,
  ): Promise<Map<string, Uint8Array>> {
    const entries =
      files instanceof Map ? [...files.entries()] : Object.entries(files);
    const out = new Map<string, Uint8Array>();
    for (const [name, url] of entries) {
      out.set(name, await this.fetchFile(url));
    }
    return out;
  }

  /** Fetches a single URL, using the Cache API when available. */
  async fetchFile(url: string): Promise<Uint8Array> {
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
      throw new MoonshineDownloadError(
        `Failed to download ${url}: ${response.status} ${response.statusText}`,
      );
    }
    if (cache) {
      // Store a clone so the body below can still be read.
      await cache.put(url, response.clone());
    }
    const buf = await this.readWithProgress(response, basename(url));
    return new Uint8Array(buf);
  }

  private async readWithProgress(
    response: Response,
    file: string,
  ): Promise<ArrayBuffer> {
    const total = Number(response.headers.get('content-length')) || undefined;
    if (!response.body || !this.onProgress) {
      const buf = await response.arrayBuffer();
      this.onProgress?.(buf.byteLength, total, file);
      return buf;
    }
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
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

  private async openCache(): Promise<Cache | undefined> {
    try {
      if (typeof caches !== 'undefined') {
        return await caches.open(this.cacheName);
      }
    } catch {
      // Cache API not available (e.g. non-secure context / Node) — skip.
    }
    return undefined;
  }
}

function joinUrl(base: string, file: string): string {
  return `${base.replace(/\/+$/, '')}/${file.replace(/^\/+/, '')}`;
}

function basename(url: string): string {
  const clean = url.split(/[?#]/)[0];
  const idx = clean.lastIndexOf('/');
  return idx >= 0 ? clean.slice(idx + 1) : clean;
}
