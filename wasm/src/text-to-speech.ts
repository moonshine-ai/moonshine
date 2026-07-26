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
import {
  loadMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawTextToSpeech,
} from './module.js';
import type { TtsSynthesisResult } from './types.js';

const DEFAULT_TTS_ASSET_BASE = 'https://download.moonshine.ai/tts';

/** Canonical asset key under which a ZipVoice clone reference clip is supplied. */
const ZIPVOICE_CLONE_AUDIO_KEY = 'zipvoice/clone_audio';

export interface TtsFromAssets {
  language: string;
  /** Map of canonical key (e.g. `kokoro/model.onnx`) -> bytes. */
  assets: Map<string, Uint8Array>;
  /** Selected voice (prefixed, e.g. `kokoro_af_heart` / `zipvoice_american_female`). */
  voice?: string;
  /** Extra `moonshine_option_t` entries forwarded to the native synthesizer. */
  options?: Record<string, string>;
}

export interface TtsFromCatalog {
  language: string;
  /** Comma-separated languages passed to the manifest helper (default: language). */
  languages?: string;
  voice?: string;
  /** Base URL prepended to each canonical asset key. */
  assetBaseUrl?: string;
  downloader?: AssetDownloader;
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
  /** Extra `moonshine_option_t` entries forwarded to the native synthesizer. */
  options?: Record<string, string>;
}

/**
 * ZipVoice zero-shot voice cloning from a caller-supplied reference clip. The
 * ZipVoice model assets (+ G2P) are resolved from the catalog like
 * {@link TtsFromCatalog}, and `clone` provides the reference audio. When no
 * `transcript` is given the engine clones without one (slightly lower quality)
 * — no STT model is downloaded or run, matching a `skip_transcription` capture
 * pipeline.
 */
export interface TtsClone {
  language: string;
  languages?: string;
  clone: {
    /** Mono float PCM reference clip in [-1, 1]. */
    audio: Float32Array;
    /** Sample rate of `audio` in Hz. */
    sampleRate: number;
    /** Optional transcript of the clip; improves cloning quality when present. */
    transcript?: string;
  };
  assetBaseUrl?: string;
  downloader?: AssetDownloader;
  onProgress?: (loaded: number, total: number | undefined, file: string) => void;
  options?: Record<string, string>;
}

export type TextToSpeechOptions = (
  | TtsFromAssets
  | TtsFromCatalog
  | TtsClone
) & {
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
};

/** A single voice row returned by {@link TextToSpeech.voices}. */
export interface TtsVoiceEntry {
  id: string;
  state: 'found' | 'missing';
}

export interface TtsVoicesOptions {
  /** Comma-separated languages (default: `language`). */
  languages?: string;
  language?: string;
  /**
   * A representative voice id whose prefix selects the engine to list:
   * `kokoro_*` (default), `piper_*`, or `zipvoice_*`.
   */
  voice?: string;
  moduleOptions?: LoadModuleOptions;
  module?: MoonshineModule;
}

function isFromAssets(o: TextToSpeechOptions): o is TtsFromAssets & object {
  return 'assets' in o;
}

function isClone(o: TextToSpeechOptions): o is TtsClone & object {
  return 'clone' in o;
}

/** Views a mono Float32 PCM buffer as little-endian raw bytes (no copy). */
function floatPcmToBytes(pcm: Float32Array): Uint8Array {
  return new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
}

function optionsToArrays(
  options: Record<string, string> | undefined,
): [string[], string[]] {
  const names = options ? Object.keys(options) : [];
  return [names, names.map((k) => options![k])];
}

/**
 * Resolves the TTS dependency manifest for `voice` and downloads every listed
 * asset into an in-memory key -> bytes map (prefixing each canonical key with
 * the asset base URL). Shared by the catalog and clone load paths.
 */
async function downloadTtsAssets(
  module: MoonshineModule,
  options: TtsFromCatalog | TtsClone,
  voice: string,
): Promise<Map<string, Uint8Array>> {
  if (!module.ttsDependencies) {
    throw new Error('TTS manifests are unavailable in this build.');
  }
  const languages = options.languages ?? options.language;
  const keysJson = module.ttsDependencies(languages, voice);
  const keys = JSON.parse(keysJson) as string[];
  const base = (options.assetBaseUrl ?? DEFAULT_TTS_ASSET_BASE).replace(/\/+$/, '');
  const downloader =
    options.downloader ?? new AssetDownloader({ onProgress: options.onProgress });
  const assets = new Map<string, Uint8Array>();
  for (const key of keys) {
    const url = `${base}/${key.replace(/^\/+/, '')}`;
    assets.set(key, await downloader.fetchFile(url));
  }
  return assets;
}

export class TextToSpeech {
  private readonly raw: RawTextToSpeech;

  private constructor(raw: RawTextToSpeech) {
    this.raw = raw;
  }

  static async load(options: TextToSpeechOptions): Promise<TextToSpeech> {
    const module = options.module ?? (await loadMoonshineModule(options.moduleOptions));
    if (!module.TextToSpeech) {
      throw new Error(
        'This Moonshine WASM build was compiled without TTS support.',
      );
    }

    let assets: Map<string, Uint8Array>;
    let nativeOptions: Record<string, string> = { ...(options.options ?? {}) };

    if (isFromAssets(options)) {
      assets = options.assets;
      if (options.voice) nativeOptions.voice = options.voice;
    } else if (isClone(options)) {
      // ZipVoice cloning: fetch the ZipVoice engine + G2P assets, then add the
      // caller's reference clip as an in-memory `zipvoice/clone_audio` buffer.
      assets = await downloadTtsAssets(module, options, 'zipvoice');
      assets.set(
        ZIPVOICE_CLONE_AUDIO_KEY,
        floatPcmToBytes(options.clone.audio),
      );
      nativeOptions.voice = 'zipvoice';
      nativeOptions.zipvoice_clone_sample_rate = String(
        Math.round(options.clone.sampleRate),
      );
      if (options.clone.transcript) {
        nativeOptions.zipvoice_clone_transcript = options.clone.transcript;
      }
    } else {
      assets = await downloadTtsAssets(module, options, options.voice ?? '');
      if (options.voice) nativeOptions.voice = options.voice;
    }

    const keys = [...assets.keys()];
    const buffers = [...assets.values()];
    const [optionNames, optionValues] = optionsToArrays(nativeOptions);
    const raw = wrapErrors(
      () =>
        new module.TextToSpeech!(
          options.language,
          keys,
          buffers,
          optionNames,
          optionValues,
        ),
    );
    return new TextToSpeech(raw);
  }

  /**
   * Lists the TTS voices known for a language, with availability state. Pass a
   * `voice` whose prefix selects the engine to enumerate: `kokoro_*` (the
   * default), `piper_*`, or `zipvoice_*`.
   */
  static async voices(options: TtsVoicesOptions): Promise<TtsVoiceEntry[]> {
    const module =
      options.module ?? (await loadMoonshineModule(options.moduleOptions));
    if (!module.ttsVoices) {
      throw new Error('TTS voice listing is unavailable in this build.');
    }
    const languages = options.languages ?? options.language ?? '';
    const [names, values] = optionsToArrays(
      options.voice ? { voice: options.voice } : undefined,
    );
    const json = module.ttsVoices(languages, names, values);
    const parsed = JSON.parse(json) as Record<string, TtsVoiceEntry[]>;
    // The native API returns a map of language -> voices; flatten to a single
    // list (deduping by id) since callers query one language/engine at a time.
    const seen = new Map<string, TtsVoiceEntry>();
    for (const entries of Object.values(parsed)) {
      for (const entry of entries) {
        if (!seen.has(entry.id)) seen.set(entry.id, entry);
      }
    }
    return [...seen.values()];
  }

  /** Synthesizes `text` to mono PCM. */
  say(text: string): TtsSynthesisResult {
    const result = wrapErrors(() => this.raw.say(text));
    return { audio: result.audio, sampleRate: result.sampleRate };
  }

  /**
   * Synthesizes and plays `text` through WebAudio, resolving when playback
   * finishes. Pass an existing AudioContext to reuse one.
   */
  async speak(text: string, audioContext?: AudioContext): Promise<void> {
    const { audio, sampleRate } = this.say(text);
    const ctx = audioContext ?? new AudioContext();
    try {
      const buffer = ctx.createBuffer(1, audio.length, sampleRate);
      buffer.copyToChannel(audio, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });
    } finally {
      if (!audioContext) await ctx.close();
    }
  }

  close(): void {
    wrapErrors(() => this.raw.close());
  }

  [Symbol.dispose](): void {
    this.close();
  }
}
