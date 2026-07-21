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

/** Load a transcriber from raw in-memory model bytes. */
export interface TranscriberFromBytes {
  encoder: Uint8Array;
  decoder: Uint8Array;
  tokenizer: Uint8Array;
  /** Optional spelling-CNN for alphanumeric fusion (SpellingMode). */
  spelling?: Uint8Array;
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
  | TranscriberFromCatalog
) & {
  /** Options forwarded to the WASM module loader. */
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
};

const ENCODER_FILE = 'encoder_model.ort';
const DECODER_FILE = 'decoder_model_merged.ort';
const TOKENIZER_FILE = 'tokenizer.bin';
const SPELLING_FILE = 'spelling_cnn.ort';

function isFromBytes(o: TranscriberLoadOptions): o is TranscriberFromBytes & object {
  return 'encoder' in o && 'decoder' in o && 'tokenizer' in o;
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
   * Loads a transcriber. Either pass raw model bytes, or a `language` to fetch
   * the model from the Moonshine CDN (cached for next time).
   */
  static async load(options: TranscriberLoadOptions): Promise<Transcriber> {
    const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));

    if (isFromBytes(options)) {
      const arch = options.modelArch ?? ModelArch.Base;
      const raw = wrapErrors(
        () =>
          new module.Transcriber(
            options.encoder,
            options.decoder,
            options.tokenizer,
            options.spelling,
            arch,
          ),
      );
      return new Transcriber(raw, module);
    }

    // Catalog path: resolve the manifest via the C ABI, then download.
    const arch = options.modelArch ?? ModelArch.Base;
    const downloader =
      options.downloader ??
      new AssetDownloader({ onProgress: options.onProgress });
    const manifest = module.sttDependencies(options.language, String(arch));
    const files = await downloader.downloadManifest(manifest);

    const encoder = requireFile(files, ENCODER_FILE, options.language);
    const decoder = requireFile(files, DECODER_FILE, options.language);
    const tokenizer = requireFile(files, TOKENIZER_FILE, options.language);
    const spelling = options.includeSpelling
      ? files.get(SPELLING_FILE)
      : undefined;

    const raw = wrapErrors(
      () => new module.Transcriber(encoder, decoder, tokenizer, spelling, arch),
    );
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

function requireFile(
  files: Map<string, Uint8Array>,
  name: string,
  language: string,
): Uint8Array {
  const bytes = files.get(name);
  if (!bytes) {
    throw new Error(
      `Model manifest for "${language}" did not include ${name}.`,
    );
  }
  return bytes;
}
