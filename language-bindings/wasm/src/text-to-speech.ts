/**
 * Text to speech, including zero-shot voice cloning.
 *
 * ```ts
 * const tts = new TextToSpeech().language('en_us').voice('kokoro_af_heart');
 * await tts.load();
 * await tts.say('Hello world!');
 * ```
 *
 * Cloning is a create-time mode, like choosing a catalog voice. Call
 * {@link cloning} before {@link load} so every ZipVoice + clone-ASR asset is
 * fetched up front; then {@link cloneFrom} only swaps the reference clip:
 *
 * ```ts
 * const tts = new TextToSpeech().language('en_us').cloning();
 * await tts.load();
 * await tts.cloneFrom(recording);
 * await tts.say('Hello in your voice!');
 * ```
 */

import { AssetDownloader } from './asset-downloader.js';
import { ModelArch } from './enums.js';
import { wrapErrors } from './errors.js';
import { wrapProgress, type ProgressCallback } from './mic-transcriber.js';
import {
  loadMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawTextToSpeech,
} from './module.js';
import { Transcriber } from './transcriber.js';
import {
  TtsWorkerHost,
  ttsWorkerSupported,
} from './tts-worker-host.js';
import type { TtsSynthesisResult } from './types.js';
import {
  extractSpeechClip,
  VoiceClone,
  type VoiceCloneOptions,
} from './voice-clone.js';

const DEFAULT_TTS_ASSET_BASE = 'https://download.moonshine.ai/tts';
const DEFAULT_LANGUAGE = 'en';

/** Canonical asset key under which a ZipVoice clone reference clip is supplied. */
const ZIPVOICE_CLONE_AUDIO_KEY = 'zipvoice/clone_audio';
/** Engine name used when creating ZipVoice from a captured clone clip. */
const CLONE_ENGINE = 'zipvoice';
/** Built-in ZipVoice voice used by {@link TextToSpeech.cloning} before a clip exists. */
const CLONE_PRESET_VOICE = 'zipvoice_american_female';

/** Anything {@link TextToSpeech.cloneFrom} can take a reference voice from. */
export type CloneSource =
  | string
  | URL
  | Blob
  | ArrayBuffer
  | AudioBuffer
  | Float32Array
  | VoiceClone
  | { audio: Float32Array; sampleRate: number };

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

export class TextToSpeech {
  private raw?: RawTextToSpeech;
  private mod?: MoonshineModule;
  private moduleOpts?: LoadModuleOptions;
  private downloader?: AssetDownloader;

  private languageCode = DEFAULT_LANGUAGE;
  private voiceId?: string;
  private assetBase?: string;
  private suppliedAssets?: Map<string, Uint8Array>;
  /** Assets fetched by {@link load}; reused by {@link cloneFrom} with no re-download. */
  private loadedAssets?: Map<string, Uint8Array>;
  private extraOptions: Record<string, string> = {};
  private progressCallback?: ProgressCallback;
  private context?: AudioContext;
  private ownsContext = false;
  private cloningWanted = false;

  /** The clip the current voice was cloned from, if any. */
  private cloneAudio?: Float32Array;
  private cloneTranscript?: string;

  /** Last {@link say} output, keyed by the spoken text, for instant replay. */
  private sayCache?: { text: string; chunks: TtsSynthesisResult[] };

  /**
   * Browser worker that owns a synthesizer for {@link say}. Absent in Node
   * tests, which keep synthesis on the main thread.
   */
  private workerHost?: TtsWorkerHost;

  /** Snapshot used to recreate a main-thread synthesizer for {@link synthesize}. */
  private mainEngine?: {
    language: string;
    keys: string[];
    buffers: Uint8Array[];
    optionNames: string[];
    optionValues: string[];
  };

  /** Synthesis language, e.g. `"en"` or `"en_us"`. Defaults to `"en"`. */
  language(code: string): this {
    this.languageCode = code;
    return this;
  }

  /**
   * Catalog voice id, e.g. `"kokoro_af_heart"`. Clears {@link cloning} — a
   * synthesizer is either a catalog voice or a cloning engine, not both.
   */
  voice(id: string): this {
    this.voiceId = id;
    this.cloningWanted = false;
    return this;
  }

  /**
   * Fetches the voice and G2P assets from a base URL you host instead of the
   * Moonshine CDN. Canonical names (e.g. `kokoro/model.ort`) are appended.
   */
  modelsFrom(baseUrl: string): this {
    this.assetBase = baseUrl;
    return this;
  }

  /**
   * Supplies voice assets directly, keyed by canonical name (e.g.
   * `kokoro/model.ort`). Nothing is downloaded when this is set.
   */
  assets(assets: Map<string, Uint8Array>): this {
    this.suppliedAssets = assets;
    return this;
  }

  /**
   * Create this synthesizer as a ZipVoice cloning engine. Call before
   * {@link load} so ZipVoice and clone-ASR assets are fetched up front.
   * Clears {@link voice}. Only then may {@link cloneFrom} / {@link startCloning}
   * be used.
   */
  cloning(enabled = true): this {
    this.cloningWanted = enabled;
    if (enabled) this.voiceId = undefined;
    return this;
  }

  /** Model download progress, as a `0..1` fraction. */
  onProgress(callback: ProgressCallback): this {
    this.progressCallback = callback;
    return this;
  }

  /** Reuses an AudioContext for playback rather than creating one per call. */
  audioContext(context: AudioContext): this {
    this.context = context;
    this.ownsContext = false;
    return this;
  }

  /** Shares an already-initialised WASM module. */
  useModule(module: MoonshineModule): this {
    this.mod = module;
    return this;
  }

  /** Shares a downloader, so several engines report progress together. */
  useDownloader(downloader: AssetDownloader): this {
    this.downloader = downloader;
    return this;
  }

  /** Escape hatch for `moonshine_option_t` entries the builder doesn't cover. */
  nativeOptions(options: Record<string, string>): this {
    this.extraOptions = { ...this.extraOptions, ...options };
    return this;
  }

  /**
   * Downloads every asset this synthesizer needs and prepares it. With
   * {@link cloning}, that includes ZipVoice and clone ASR — afterwards
   * {@link cloneFrom} does not go back to the network.
   */
  async load(): Promise<this> {
    this.mod ??= await loadMoonshineModule(this.moduleOpts);
    if (!this.mod.TextToSpeech) {
      throw new Error('This Moonshine WASM build was compiled without TTS support.');
    }
    const voice = this.cloningWanted
      ? CLONE_PRESET_VOICE
      : (this.voiceId ?? '');
    await this.build(voice, { allowDownload: true });
    return this;
  }

  /**
   * Clones the voice in `source` and uses it for subsequent synthesis. Accepts
   * a URL or path, a `File` / `Blob`, an `AudioBuffer`, raw 16 kHz mono PCM, or
   * a {@link VoiceClone} captured with {@link startCloning}.
   *
   * Requires {@link cloning} before {@link load}. The library trims the
   * recording and transcribes it for the vocoder (using assets already fetched
   * by `load`) unless you pass `transcript`.
   */
  async cloneFrom(
    source: CloneSource,
    options: { transcript?: string } = {},
  ): Promise<this> {
    this.requireCloningMode('cloneFrom()');
    if (!this.isEngineReady()) {
      throw new Error('Call load() before cloneFrom().');
    }
    const { audio, sampleRate, transcript } = await this.resolveCloneSource(
      source,
    );
    // Prefer a main-thread handle for extract; create lazily if say() has been
    // using the worker-only engine so far.
    this.ensureMainThreadEngine();
    const ttsHandle = this.raw!.handle();
    const clip =
      sampleRate === 16000 && audio.length <= 16000 * 10
        ? audio
        : await extractSpeechClip(audio, sampleRate, {
            module: this.mod,
            ttsHandle,
          });

    this.cloneAudio = clip;
    this.cloneTranscript =
      options.transcript ?? transcript ?? undefined;
    // Let the UI paint before sync rebuild work on the main thread.
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
    await this.build(CLONE_ENGINE, { allowDownload: false });
    return this;
  }

  /**
   * Starts capturing a reference voice incrementally, for cloning from a live
   * microphone. Requires {@link cloning} before {@link load}.
   */
  startCloning(options: VoiceCloneOptions = {}): VoiceClone {
    this.requireCloningMode('startCloning()');
    if (!this.isEngineReady()) {
      throw new Error('Call load() before startCloning().');
    }
    this.ensureMainThreadEngine();
    return new VoiceClone(this.mod!, this.raw!.handle(), options);
  }

  /** True once a voice has been cloned into this instance. */
  get isCloned(): boolean {
    return this.cloneAudio !== undefined;
  }

  /** Synthesizes `text` to mono PCM without playing it. */
  synthesize(text: string): TtsSynthesisResult {
    this.ensureMainThreadEngine();
    const result = wrapErrors(() => this.raw!.say(text));
    return { audio: result.audio, sampleRate: result.sampleRate };
  }

  /**
   * Concatenated PCM from the last {@link say} call, if any. Handy for a
   * download link or for tests; replay uses the per-sentence cache instead.
   */
  get lastSaid(): TtsSynthesisResult | undefined {
    const chunks = this.sayCache?.chunks;
    if (!chunks?.length) return undefined;
    if (chunks.length === 1) return chunks[0];
    const sampleRate = chunks[0]!.sampleRate;
    let total = 0;
    for (const chunk of chunks) total += chunk.audio.length;
    const audio = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      audio.set(chunk.audio, offset);
      offset += chunk.audio.length;
    }
    return { audio, sampleRate };
  }

  /**
   * Speaks `text` out loud, resolving when playback finishes.
   *
   * Long strings are split on an approximate sentence boundary (`.`, `!`,
   * `?`, or `:` followed by whitespace). The first sentence starts playing as soon
   * as it is ready; later sentences synthesize on a Web Worker while the
   * previous one plays (main-thread fallback where Workers are unavailable).
   * Calling again with the same text replays the cached audio instantly.
   */
  async say(text: string): Promise<void> {
    if (!this.isEngineReady()) {
      throw new Error('Call load() before say().');
    }
    const sentences = splitSayUtterances(text);
    if (!sentences.length) return;

    if (this.sayCache?.text === text && this.sayCache.chunks.length > 0) {
      await this.playChunks(this.sayCache.chunks);
      return;
    }

    // Let the caller paint (disabled button, status text) before work starts.
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

    const synthesizeOne = (sentence: string): Promise<TtsSynthesisResult> =>
      this.workerHost
        ? this.workerHost.synthesize(sentence)
        : Promise.resolve(this.synthesize(sentence));

    const chunks: TtsSynthesisResult[] = [];
    let pending: Promise<TtsSynthesisResult> | undefined;
    for (let i = 0; i < sentences.length; i++) {
      const result = await (pending ?? synthesizeOne(sentences[i]!));
      pending = undefined;
      chunks.push(result);
      const playing = this.playOne(result);
      if (i + 1 < sentences.length) {
        // Kick off the next sentence on the worker while this one plays.
        pending = synthesizeOne(sentences[i + 1]!);
      }
      await playing;
    }
    this.sayCache = { text, chunks };
  }

  private async playChunks(chunks: TtsSynthesisResult[]): Promise<void> {
    for (const chunk of chunks) {
      await this.playOne(chunk);
    }
  }

  private playOne(result: TtsSynthesisResult): Promise<void> {
    if (result.audio.length === 0) return Promise.resolve();
    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(1, result.audio.length, result.sampleRate);
    buffer.copyToChannel(result.audio, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    return new Promise((resolve) => {
      source.onended = () => resolve();
      source.start();
    });
  }

  /**
   * Lists the voices known for a language, with availability state. Pass a
   * `voice` whose prefix selects the engine to enumerate: `kokoro_*` (the
   * default), `piper_*`, or `zipvoice_*`.
   */
  static async voices(options: TtsVoicesOptions = {}): Promise<TtsVoiceEntry[]> {
    const module =
      options.module ?? (await loadMoonshineModule(options.moduleOptions));
    if (!module.ttsVoices) {
      throw new Error('TTS voice listing is unavailable in this build.');
    }
    const languages = options.languages ?? options.language ?? '';
    const names = options.voice ? ['voice'] : [];
    const values = options.voice ? [options.voice] : [];
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

  close(): void {
    this.workerHost?.close();
    this.workerHost = undefined;
    if (this.raw) wrapErrors(() => this.raw!.close());
    this.raw = undefined;
    this.mainEngine = undefined;
    this.loadedAssets = undefined;
    this.sayCache = undefined;
    if (this.ownsContext) void this.context?.close();
    this.context = undefined;
  }

  [Symbol.dispose](): void {
    this.close();
  }

  // --- Internals ---

  private requireCloningMode(what: string): void {
    if (!this.cloningWanted) {
      throw new Error(
        `Call cloning() before load() to use ${what}. ` +
          'Catalog voices and cloning are separate synthesizer modes.',
      );
    }
  }

  private isEngineReady(): boolean {
    return this.workerHost !== undefined || this.raw !== undefined || this.mainEngine !== undefined;
  }

  /**
   * Creates (or recreates) the main-thread synthesizer used by
   * {@link synthesize} and {@link startCloning}. No-op when one already exists
   * matching {@link mainEngine}.
   */
  private ensureMainThreadEngine(): void {
    if (this.raw) return;
    if (!this.mod?.TextToSpeech || !this.mainEngine) {
      throw new Error('Call load() before synthesizing.');
    }
    const { language, keys, buffers, optionNames, optionValues } = this.mainEngine;
    this.raw = wrapErrors(
      () =>
        new this.mod!.TextToSpeech!(
          language,
          keys,
          buffers,
          optionNames,
          optionValues,
        ),
    );
  }

  /**
   * (Re)creates the native synthesizer. `allowDownload` is true for
   * {@link load}; {@link cloneFrom} reuses {@link loadedAssets}.
   *
   * When Workers are available the engine used by {@link say} is built on a
   * worker (main thread stays free). A main-thread copy is created lazily for
   * {@link synthesize} / {@link startCloning}.
   */
  private async build(
    voice: string,
    options: { allowDownload: boolean },
  ): Promise<void> {
    const assets = await this.resolveAssets(voice, options.allowDownload);

    if (this.cloneAudio && !this.cloneTranscript) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      this.cloneTranscript = await this.autotranscribeCloneClip(
        this.cloneAudio,
        assets,
      );
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    const createAssets = new Map(assets);
    const nativeOptions: Record<string, string> = { ...this.extraOptions };

    if (this.cloneAudio) {
      createAssets.set(
        ZIPVOICE_CLONE_AUDIO_KEY,
        floatPcmToBytes(this.cloneAudio),
      );
      nativeOptions.voice = CLONE_ENGINE;
      nativeOptions.zipvoice_clone_sample_rate = '16000';
      if (this.cloneTranscript) {
        nativeOptions.zipvoice_clone_transcript = this.cloneTranscript;
      }
    } else if (voice) {
      nativeOptions.voice = voice;
    }

    // Native create does not need clone_asr once the transcript is known (or
    // when there is no clip yet). Keep them in loadedAssets for later.
    for (const key of [...createAssets.keys()]) {
      if (key.startsWith('clone_asr/')) createAssets.delete(key);
    }

    const keys = [...createAssets.keys()];
    const buffers = [...createAssets.values()].map((b) => new Uint8Array(b));
    const optionNames = Object.keys(nativeOptions);
    const optionValues = optionNames.map((name) => nativeOptions[name]!);
    this.mainEngine = {
      language: this.languageCode,
      keys,
      buffers,
      optionNames,
      optionValues,
    };

    // Drop any previous main-thread engine; recreate on demand.
    this.raw?.close();
    this.raw = undefined;
    this.sayCache = undefined;

    if (ttsWorkerSupported()) {
      this.workerHost ??= new TtsWorkerHost();
      await this.workerHost.setEngine(this.mainEngine);
    } else {
      this.ensureMainThreadEngine();
    }
  }

  private async resolveAssets(
    voice: string,
    allowDownload: boolean,
  ): Promise<Map<string, Uint8Array>> {
    if (this.suppliedAssets) {
      this.loadedAssets = new Map(this.suppliedAssets);
      return new Map(this.suppliedAssets);
    }
    if (this.loadedAssets && this.loadedAssets.size > 0) {
      return new Map(this.loadedAssets);
    }
    if (!allowDownload) {
      throw new Error(
        'Clone assets were not loaded. Call cloning() before load() so ' +
          'ZipVoice and clone ASR are fetched up front.',
      );
    }
    const downloaded = await this.downloadAssets(voice);
    this.loadedAssets = downloaded;
    return new Map(downloaded);
  }

  /** One-shot STT of a clone clip using the advertised clone_asr assets. */
  private async autotranscribeCloneClip(
    clip: Float32Array,
    assets: Map<string, Uint8Array>,
  ): Promise<string | undefined> {
    const asrFiles = new Map<string, Uint8Array>();
    for (const [key, bytes] of assets) {
      if (key.startsWith('clone_asr/')) {
        asrFiles.set(key.slice('clone_asr/'.length), bytes);
      }
    }
    if (asrFiles.size === 0) return undefined;

    const arch = asrFiles.has('frontend.ort')
      ? ModelArch.MediumStreaming
      : ModelArch.Base;
    const transcriber = await Transcriber.load({
      files: asrFiles,
      modelArch: arch,
      module: this.mod,
      options: { word_timestamps: 'true' },
    });
    try {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      const result = transcriber.transcribe(clip, { sampleRate: 16000 });
      const text = result.lines
        .map((line) => line.text.trim())
        .filter(Boolean)
        .join(' ')
        .trim();
      return text || undefined;
    } finally {
      transcriber.close();
    }
  }

  private async downloadAssets(voice: string): Promise<Map<string, Uint8Array>> {
    const module = this.mod!;
    if (!module.ttsDependencies) {
      throw new Error('TTS manifests are unavailable in this build.');
    }
    const depsJson = module.ttsDependencies(this.languageCode, voice);
    const parsed = JSON.parse(depsJson) as {
      groups?: Array<{
        base_url?: string;
        role?: string;
        files?: Array<{ name?: string; url?: string }>;
      }>;
    };
    const urls = new Map<string, string>();
    const ttsBase = (this.assetBase ?? DEFAULT_TTS_ASSET_BASE).replace(
      /\/+$/,
      '',
    );
    for (const group of parsed.groups ?? []) {
      const isCloneAsr = group.role === 'clone_asr';
      const base = (group.base_url ?? DEFAULT_TTS_ASSET_BASE).replace(
        /\/+$/,
        '',
      );
      for (const file of group.files ?? []) {
        const name = file.name?.trim();
        if (!name || !name.includes('/')) continue;
        let url = file.url?.trim() || `${base}/${name.replace(/^\/+/, '')}`;
        if (!isCloneAsr && this.assetBase) {
          url = `${ttsBase}/${name.replace(/^\/+/, '')}`;
        }
        urls.set(name, url);
      }
    }
    if (urls.size === 0) {
      throw new Error('TTS dependency manifest listed no downloadable files.');
    }
    const downloader =
      this.downloader ??
      new AssetDownloader({ onProgress: wrapProgress(this.progressCallback) });
    return downloader.downloadNamedFiles(urls);
  }

  private async resolveCloneSource(
    source: CloneSource,
  ): Promise<{ audio: Float32Array; sampleRate: number; transcript?: string }> {
    if (source instanceof VoiceClone) {
      const audio = source.audio;
      if (!audio) {
        throw new Error(
          'That VoiceClone has not captured enough speech yet — wait for onReady.',
        );
      }
      return {
        audio,
        sampleRate: source.sampleRate,
        transcript: source.transcript,
      };
    }
    if (source instanceof Float32Array) {
      return { audio: source, sampleRate: 16000 };
    }
    if (typeof source === 'object' && 'audio' in source && 'sampleRate' in source) {
      return { audio: source.audio, sampleRate: source.sampleRate };
    }
    if (isAudioBuffer(source)) {
      return {
        audio: source.getChannelData(0),
        sampleRate: source.sampleRate,
      };
    }

    let bytes: ArrayBuffer;
    if (typeof source === 'string' || source instanceof URL) {
      const response = await fetch(String(source));
      if (!response.ok) {
        throw new Error(`Failed to fetch clone audio: ${response.status}`);
      }
      bytes = await response.arrayBuffer();
    } else if (source instanceof Blob) {
      bytes = await source.arrayBuffer();
    } else {
      bytes = source;
    }
    const decoded = await this.ensureContext().decodeAudioData(bytes.slice(0));
    return {
      audio: decoded.getChannelData(0),
      sampleRate: decoded.sampleRate,
    };
  }

  private ensureContext(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext();
      this.ownsContext = true;
    }
    return this.context;
  }
}

/** `AudioBuffer` is a browser global, so guard the check for other runtimes. */
function isAudioBuffer(value: unknown): value is AudioBuffer {
  return typeof AudioBuffer !== 'undefined' && value instanceof AudioBuffer;
}

/**
 * Approximate sentence split for {@link TextToSpeech.say}: break on `.` / `!`
 * / `?` / `:` followed by whitespace so the first clause can start sooner.
 */
export function splitSayUtterances(text: string): string[] {
  const stripped = text.trim();
  if (!stripped) return [];
  const parts: string[] = [];
  let start = 0;
  let i = 0;
  while (i < stripped.length) {
    const ch = stripped[i];
    if (
      (ch === '.' || ch === '!' || ch === '?' || ch === ':') &&
      i + 1 < stripped.length &&
      /\s/.test(stripped[i + 1]!)
    ) {
      const end = i + 1;
      let j = i + 1;
      while (j < stripped.length && /\s/.test(stripped[j]!)) j += 1;
      const piece = stripped.slice(start, end).trim();
      if (piece) parts.push(piece);
      start = j;
      i = j;
      continue;
    }
    i += 1;
  }
  const tail = stripped.slice(start).trim();
  if (tail) parts.push(tail);
  return parts;
}

/** Views a mono Float32 PCM buffer as little-endian raw bytes (no copy). */
function floatPcmToBytes(pcm: Float32Array): Uint8Array {
  return new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
}
