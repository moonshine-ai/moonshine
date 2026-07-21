/**
 * Grapheme-to-phoneme conversion (Phase 2), mirroring the Python/Swift
 * `GraphemeToPhonemizer`. Turns text into IPA phonemes using the WASM binding.
 */

import { AssetDownloader } from './asset-downloader.js';
import { wrapErrors } from './errors.js';
import {
  loadMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawGraphemeToPhonemizer,
} from './module.js';

const DEFAULT_G2P_ASSET_BASE = 'https://download.moonshine.ai/tts';

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

function isFromAssets(o: GraphemeToPhonemizerOptions): o is G2pFromAssets & object {
  return 'assets' in o;
}

export class GraphemeToPhonemizer {
  private readonly raw: RawGraphemeToPhonemizer;

  private constructor(raw: RawGraphemeToPhonemizer) {
    this.raw = raw;
  }

  static async load(
    options: GraphemeToPhonemizerOptions,
  ): Promise<GraphemeToPhonemizer> {
    const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
    if (!module.GraphemeToPhonemizer) {
      throw new Error(
        'This Moonshine WASM build was compiled without G2P support.',
      );
    }

    let assets: Map<string, Uint8Array>;
    if (isFromAssets(options)) {
      assets = options.assets;
    } else {
      if (!module.g2pDependencies) {
        throw new Error('G2P manifests are unavailable in this build.');
      }
      const languages = options.languages ?? options.language;
      const keys = JSON.parse(module.g2pDependencies(languages)) as string[];
      const base = (options.assetBaseUrl ?? DEFAULT_G2P_ASSET_BASE).replace(/\/+$/, '');
      const downloader =
        options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
      assets = new Map();
      for (const key of keys) {
        assets.set(key, await downloader.fetchFile(`${base}/${key.replace(/^\/+/, '')}`));
      }
    }

    const raw = wrapErrors(
      () =>
        new module.GraphemeToPhonemizer!(
          options.language,
          [...assets.keys()],
          [...assets.values()],
        ),
    );
    return new GraphemeToPhonemizer(raw);
  }

  /** Converts `text` into IPA phonemes. */
  textToPhonemes(text: string): string {
    return wrapErrors(() => this.raw.textToPhonemes(text));
  }

  close(): void {
    wrapErrors(() => this.raw.close());
  }

  [Symbol.dispose](): void {
    this.close();
  }
}
