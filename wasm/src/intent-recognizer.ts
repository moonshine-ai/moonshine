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
import {
  loadMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawIntentRecognizer,
} from './module.js';
import { normalizeIntentMatches, type IntentMatch } from './types.js';

let stagedModelCounter = 0;

export interface IntentFromCatalog {
  /** Embedding model id (e.g. `"embeddinggemma-300m"`). Empty = default. */
  modelName?: string;
  modelArch?: EmbeddingModelArch;
  /** One of "q4", "q8", "fp16", "fp32", "q4f16". Empty = model default. */
  variant?: string;
  downloader?: AssetDownloader;
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}

export type IntentRecognizerOptions = IntentFromCatalog & {
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
};

/** A phrase to register, with an optional priority for tie-breaking. */
export interface IntentPhrase {
  phrase: string;
  priority?: number;
}

export class IntentRecognizer {
  private readonly raw: RawIntentRecognizer;

  private constructor(raw: RawIntentRecognizer) {
    this.raw = raw;
  }

  static async load(
    options: IntentRecognizerOptions = {},
  ): Promise<IntentRecognizer> {
    const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
    if (!module.FS) {
      throw new Error(
        'Intent recognition needs the Emscripten filesystem; rebuild with -sFORCE_FILESYSTEM=1.',
      );
    }

    const arch = options.modelArch ?? EmbeddingModelArch.Gemma300M;
    const downloader =
      options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
    const manifest = module.intentDependencies(
      options.modelName ?? '',
      options.variant ?? '',
    );
    const files = await downloader.downloadManifest(manifest);

    const dir = `/moonshine-intent-${stagedModelCounter++}`;
    stageFiles(module, dir, files);

    const raw = wrapErrors(
      () => new module.IntentRecognizer(dir, arch, options.variant ?? ''),
    );
    return new IntentRecognizer(raw);
  }

  /** Registers a phrase (optionally many). */
  register(phrases: string | IntentPhrase | Array<string | IntentPhrase>): void {
    const list = Array.isArray(phrases) ? phrases : [phrases];
    for (const item of list) {
      const phrase = typeof item === 'string' ? item : item.phrase;
      const priority = typeof item === 'string' ? 0 : (item.priority ?? 0);
      wrapErrors(() => this.raw.registerIntent(phrase, priority));
    }
  }

  unregister(phrase: string): void {
    wrapErrors(() => this.raw.unregisterIntent(phrase));
  }

  clear(): void {
    wrapErrors(() => this.raw.clearIntents());
  }

  /** Returns registered phrases ranked by similarity to `utterance`. */
  closestIntents(utterance: string, threshold = 0): IntentMatch[] {
    return wrapErrors(() =>
      normalizeIntentMatches(this.raw.closestIntents(utterance, threshold)),
    );
  }

  /** Convenience: the single best match above `threshold`, or null. */
  bestIntent(utterance: string, threshold = 0): IntentMatch | null {
    const matches = this.closestIntents(utterance, threshold);
    return matches.length > 0 ? matches[0] : null;
  }

  close(): void {
    wrapErrors(() => this.raw.close());
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

/** Writes files into a fresh MEMFS directory (basename keys only). */
function stageFiles(
  module: MoonshineModule,
  dir: string,
  files: Map<string, Uint8Array>,
): void {
  const fs = module.FS!;
  if (!fs.analyzePath(dir).exists) fs.mkdir(dir);
  for (const [name, bytes] of files) {
    fs.writeFile(`${dir}/${name}`, bytes);
  }
}
