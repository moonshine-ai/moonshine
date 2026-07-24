/**
 * High-level speech-to-text entry point, mirroring the Python/Swift/Android
 * `Transcriber`. Load it with the async {@link Transcriber.load} factory (which
 * fetches the model from the CDN), then either transcribe a whole buffer or
 * drive a streaming {@link Stream}.
 */

import { AssetDownloader } from './asset-downloader.js';
import { ModelArch, TranscribeFlags, modelArchToString } from './enums.js';
import { wrapErrors } from './errors.js';
import {
  loadMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawTranscriber,
} from './module.js';
import { Stream } from './stream.js';
import type { TranscriptEventListener } from './events.js';
import { normalizeTranscript, type Transcript } from './types.js';

/**
 * Load a transcriber from raw in-memory model bytes for a non-streaming model.
 * For streaming models (or word-timestamp/spelling extras) use
 * {@link TranscriberFromFiles} and key each buffer by its canonical filename.
 */
export interface TranscriberFromBytes {
  encoder: Uint8Array;
  decoder: Uint8Array;
  tokenizer: Uint8Array;
  /** Optional spelling-CNN for alphanumeric fusion (SpellingMode). */
  spelling?: Uint8Array;
  modelArch?: ModelArch;
}

/**
 * Load a transcriber from in-memory model buffers keyed by their canonical
 * manifest filename (e.g. `encoder_model.ort`, `decoder_model_merged.ort`,
 * `tokenizer.bin` for non-streaming; `frontend.ort`, `encoder.ort`,
 * `adapter.ort`, `cross_kv.ort`, `decoder_kv.ort`, `streaming_config.json`,
 * `tokenizer.bin` for streaming; plus optional `decoder_with_attention.ort` /
 * `decoder_kv_with_attention.ort` and `spelling_cnn.ort`). This is the general
 * in-memory loader and works for every architecture.
 */
export interface TranscriberFromFiles {
  files: Record<string, Uint8Array> | Map<string, Uint8Array>;
  modelArch?: ModelArch;
}

/** Load a transcriber by fetching a model from the CDN by language. */
export interface TranscriberFromCatalog {
  /** Language code (e.g. `"en"`) or English name (e.g. `"English"`). */
  language: string;
  modelArch?: ModelArch;
  /** Also fetch + load the spelling model if one is published. */
  includeSpelling?: boolean;
  downloader?: AssetDownloader;
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
}

export type TranscriberLoadOptions = (
  | TranscriberFromBytes
  | TranscriberFromFiles
  | TranscriberFromCatalog
) & {
  /** Options forwarded to the WASM module loader. */
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
};

/** Options for {@link Transcriber.loadFromUrls}. */
export interface TranscriberFromUrlsOptions {
  modelArch?: ModelArch;
  downloader?: AssetDownloader;
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
}

const ENCODER_FILE = 'encoder_model.ort';
const DECODER_FILE = 'decoder_model_merged.ort';
const TOKENIZER_FILE = 'tokenizer.bin';
const SPELLING_FILE = 'spelling_cnn.ort';

function isFromBytes(o: TranscriberLoadOptions): o is TranscriberFromBytes & object {
  return 'encoder' in o && 'decoder' in o && 'tokenizer' in o;
}

function isFromFiles(o: TranscriberLoadOptions): o is TranscriberFromFiles & object {
  return 'files' in o;
}

function toFileMap(
  files: Record<string, Uint8Array> | Map<string, Uint8Array>,
): Map<string, Uint8Array> {
  return files instanceof Map ? files : new Map(Object.entries(files));
}

export class Transcriber {
  private readonly raw: RawTranscriber;
  private readonly module: MoonshineModule;
  private defaultStream: Stream | undefined;
  private closed = false;

  private constructor(raw: RawTranscriber, module: MoonshineModule) {
    this.raw = raw;
    this.module = module;
  }

  /**
   * Loads a transcriber. Pass raw non-streaming bytes ({@link
   * TranscriberFromBytes}), a keyed map of model files ({@link
   * TranscriberFromFiles}, which also supports streaming), or a `language` to
   * fetch the model from the Moonshine CDN ({@link TranscriberFromCatalog},
   * cached for next time). All paths load the model purely in memory — the
   * browser has no natural filesystem.
   */
  static async load(options: TranscriberLoadOptions): Promise<Transcriber> {
    const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));

    if (isFromBytes(options)) {
      const files = new Map<string, Uint8Array>([
        [ENCODER_FILE, options.encoder],
        [DECODER_FILE, options.decoder],
        [TOKENIZER_FILE, options.tokenizer],
      ]);
      if (options.spelling) files.set(SPELLING_FILE, options.spelling);
      return Transcriber.construct(module, files, options.modelArch ?? ModelArch.Base);
    }

    if (isFromFiles(options)) {
      return Transcriber.construct(
        module,
        toFileMap(options.files),
        options.modelArch ?? ModelArch.Base,
      );
    }

    // Catalog path: resolve the manifest via the C ABI, then download every
    // file it lists and hand the whole set to the in-memory loader. Passing all
    // files (rather than a hardcoded encoder/decoder/tokenizer trio) is what
    // lets streaming architectures — whose manifests list different filenames —
    // load correctly.
    const arch = options.modelArch ?? ModelArch.Base;
    const downloader =
      options.downloader ??
      new AssetDownloader({ onProgress: options.onProgress });
    const manifest = module.sttDependencies(
      options.language,
      String(arch),
      options.includeSpelling ?? false,
    );
    const files = await downloader.downloadManifest(manifest);
    return Transcriber.construct(module, files, arch);
  }

  /**
   * Loads a transcriber from a map of canonical filename -> URL. Downloads each
   * remote file into a buffer (with caching, via {@link AssetDownloader}) and
   * feeds the buffers through the in-memory loader. Convenient when you host the
   * model files yourself instead of using the Moonshine CDN catalog.
   *
   * @example
   * const t = await Transcriber.loadFromUrls({
   *   'encoder_model.ort': '/models/encoder_model.ort',
   *   'decoder_model_merged.ort': '/models/decoder_model_merged.ort',
   *   'tokenizer.bin': '/models/tokenizer.bin',
   * }, { modelArch: ModelArch.Base });
   */
  static async loadFromUrls(
    files: Record<string, string> | Map<string, string>,
    options: TranscriberFromUrlsOptions = {},
  ): Promise<Transcriber> {
    const module =
      options.module ?? (await loadMoonshineModule(options.moduleOptions));
    const downloader =
      options.downloader ??
      new AssetDownloader({ onProgress: options.onProgress });
    const downloaded = await downloader.downloadNamedFiles(files);
    return Transcriber.construct(
      module,
      downloaded,
      options.modelArch ?? ModelArch.Base,
    );
  }

  /** Builds the raw WASM transcriber from a keyed, in-memory file map. */
  private static construct(
    module: MoonshineModule,
    files: Map<string, Uint8Array>,
    arch: ModelArch,
  ): Transcriber {
    const keys = [...files.keys()];
    const buffers = keys.map((k) => files.get(k)!);
    const raw = wrapErrors(() => new module.Transcriber(keys, buffers, arch));
    return new Transcriber(raw, module);
  }

  /** Transcribes a complete buffer of PCM audio (non-streaming). */
  transcribe(
    audio: Float32Array,
    options: { sampleRate?: number; flags?: TranscribeFlags } = {},
  ): Transcript {
    const sampleRate = options.sampleRate ?? 16000;
    const flags = options.flags ?? TranscribeFlags.None;
    return wrapErrors(() =>
      normalizeTranscript(this.raw.transcribe(audio, sampleRate, flags)),
    );
  }

  /** Creates a new streaming session. */
  createStream(options: { flags?: TranscribeFlags } = {}): Stream {
    const flags = options.flags ?? TranscribeFlags.None;
    const rawStream = wrapErrors(() => new this.module.Stream(this.raw, flags));
    return new Stream(rawStream);
  }

  // --- Convenience: a built-in default stream, matching Python's Transcriber. ---

  private ensureDefaultStream(): Stream {
    if (!this.defaultStream) this.defaultStream = this.createStream();
    return this.defaultStream;
  }

  addListener(listener: TranscriptEventListener): void {
    this.ensureDefaultStream().addListener(listener);
  }

  removeAllListeners(): void {
    this.defaultStream?.removeAllListeners();
  }

  start(): void {
    this.ensureDefaultStream().start();
  }

  addAudio(
    audio: Float32Array,
    sampleRate: number,
    flags: TranscribeFlags = TranscribeFlags.None,
  ): void {
    const stream = this.ensureDefaultStream();
    stream.addAudio(audio, sampleRate, flags);
    stream.transcribe(flags);
  }

  stop(): void {
    this.defaultStream?.stop();
  }

  /** Architecture-name helper for logging/UX. */
  archName(arch: ModelArch): string {
    return modelArchToString(arch);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.defaultStream?.close();
    wrapErrors(() => this.raw.close());
  }

  [Symbol.dispose](): void {
    this.close();
  }
}
