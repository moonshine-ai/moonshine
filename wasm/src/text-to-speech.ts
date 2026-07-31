/**
 * Text to speech, including zero-shot voice cloning.
 *
 * ```ts
 * const tts = new TextToSpeech();
 * await tts.load();
 * await tts.say('Hello world!');
 * ```
 *
 * Cloning a voice is two more lines, and the awkward parts — finding the
 * speech in the reference recording, and transcribing it so the vocoder knows
 * what was said — happen inside the library:
 *
 * ```ts
 * await tts.cloneFrom('some-speech.wav');
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
/** The only engine that supports cloning an arbitrary reference voice. */
const CLONE_VOICE = 'zipvoice';

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
  private extraOptions: Record<string, string> = {};
  private progressCallback?: ProgressCallback;
  private context?: AudioContext;
  private ownsContext = false;
  private cloningWanted = false;
  /** Assets fetched by {@link load} for an engine that isn't built yet. */
  private prefetched?: { voice: string; assets: Map<string, Uint8Array> };

  /** The clip the current voice was cloned from, if any. */
  private cloneAudio?: Float32Array;
  private cloneTranscript?: string;
  /** Loaded lazily, only when a clip needs transcribing. */
  private clipTranscriber?: Transcriber;

  /** Synthesis language, e.g. `"en"` or `"en_us"`. Defaults to `"en"`. */
  language(code: string): this {
    this.languageCode = code;
    return this;
  }

  /** Voice id, e.g. `"kokoro_af_heart"`. Defaults to the engine's own default. */
  voice(id: string): this {
    this.voiceId = id;
    return this;
  }

  /**
   * Fetches the voice and G2P assets from a base URL you host instead of the
   * Moonshine CDN. Canonical names (e.g. `kokoro/model.onnx`) are appended.
   */
  modelsFrom(baseUrl: string): this {
    this.assetBase = baseUrl;
    return this;
  }

  /**
   * Supplies voice assets directly, keyed by canonical name (e.g.
   * `kokoro/model.onnx`). Nothing is downloaded when this is set.
   */
  assets(assets: Map<string, Uint8Array>): this {
    this.suppliedAssets = assets;
    return this;
  }

  /**
   * Fetches the cloning engine during {@link load} rather than on the first
   * {@link cloneFrom}, so the first clone is quick.
   */
  cloning(enabled = true): this {
    this.cloningWanted = enabled;
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

  /** Downloads the voice assets and prepares the synthesizer. */
  async load(): Promise<this> {
    this.mod ??= await loadMoonshineModule(this.moduleOpts);
    if (!this.mod.TextToSpeech) {
      throw new Error('This Moonshine WASM build was compiled without TTS support.');
    }
    if (this.cloningWanted && !this.cloneAudio) {
      // The cloning engine can't exist until there's a voice to clone, so all
      // load() can usefully do is fetch its assets. cloneFrom() picks them up
      // from here and builds the engine without going back to the network.
      this.prefetched = {
        voice: CLONE_VOICE,
        assets: await this.downloadAssets(CLONE_VOICE),
      };
      return this;
    }
    await this.build(this.voiceId ?? '');
    return this;
  }

  /**
   * Clones the voice in `source` and uses it for subsequent synthesis. Accepts
   * a URL or path, a `File` / `Blob`, an `AudioBuffer`, raw 16 kHz mono PCM, or
   * a {@link VoiceClone} captured with {@link startCloning}.
   *
   * The library trims the recording down to a few seconds of actual speech and
   * transcribes that clip for the vocoder, downloading a small speech-to-text
   * model the first time it needs to. Callers who already know what was said
   * can skip that by passing `transcript`.
   */
  async cloneFrom(
    source: CloneSource,
    options: { transcript?: string } = {},
  ): Promise<this> {
    this.mod ??= await loadMoonshineModule(this.moduleOpts);
    const { audio, sampleRate, transcript } = await this.resolveCloneSource(
      source,
    );
    const clip =
      sampleRate === 16000 && audio.length <= 16000 * 10
        ? audio
        : await extractSpeechClip(audio, sampleRate, { module: this.mod });

    this.cloneAudio = clip;
    this.cloneTranscript =
      options.transcript ?? transcript ?? (await this.transcribeClip(clip));
    await this.build(CLONE_VOICE);
    return this;
  }

  /**
   * Starts capturing a reference voice incrementally, for cloning from a live
   * microphone. The returned object reports when it has heard enough.
   */
  startCloning(options: VoiceCloneOptions = {}): VoiceClone {
    if (!this.mod) {
      throw new Error('Call load() before startCloning().');
    }
    return new VoiceClone(this.mod, options);
  }

  /** True once a voice has been cloned into this instance. */
  get isCloned(): boolean {
    return this.cloneAudio !== undefined;
  }

  /** Synthesizes `text` to mono PCM without playing it. */
  synthesize(text: string): TtsSynthesisResult {
    if (!this.raw) {
      throw new Error(
        this.cloningWanted
          ? 'Call cloneFrom() before synthesizing with a cloned voice.'
          : 'Call load() before synthesizing.',
      );
    }
    const result = wrapErrors(() => this.raw!.say(text));
    return { audio: result.audio, sampleRate: result.sampleRate };
  }

  /** Speaks `text` out loud, resolving when playback finishes. */
  async say(text: string): Promise<void> {
    if (!text) return;
    const { audio, sampleRate } = this.synthesize(text);
    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(1, audio.length, sampleRate);
    buffer.copyToChannel(audio, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    await new Promise<void>((resolve) => {
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
    if (this.raw) wrapErrors(() => this.raw!.close());
    this.raw = undefined;
    this.prefetched = undefined;
    this.clipTranscriber?.close();
    this.clipTranscriber = undefined;
    if (this.ownsContext) void this.context?.close();
    this.context = undefined;
  }

  [Symbol.dispose](): void {
    this.close();
  }

  // --- Internals ---

  /** (Re)creates the native synthesizer for `voice`, downloading its assets. */
  private async build(voice: string): Promise<void> {
    const module = this.mod!;
    const assets = await this.downloadAssets(voice);
    const nativeOptions: Record<string, string> = { ...this.extraOptions };

    if (this.cloneAudio) {
      assets.set(ZIPVOICE_CLONE_AUDIO_KEY, floatPcmToBytes(this.cloneAudio));
      nativeOptions.voice = CLONE_VOICE;
      nativeOptions.zipvoice_clone_sample_rate = '16000';
      if (this.cloneTranscript) {
        nativeOptions.zipvoice_clone_transcript = this.cloneTranscript;
      }
    } else if (voice) {
      nativeOptions.voice = voice;
    }

    const names = Object.keys(nativeOptions);
    const values = names.map((name) => nativeOptions[name]);
    const next = wrapErrors(
      () =>
        new module.TextToSpeech!(
          this.languageCode,
          [...assets.keys()],
          [...assets.values()],
          names,
          values,
        ),
    );
    // Only tear the old engine down once the new one exists, so a failed clone
    // leaves the caller with a working synthesizer.
    this.raw?.close();
    this.raw = next;
  }

  private async downloadAssets(voice: string): Promise<Map<string, Uint8Array>> {
    if (this.suppliedAssets) return new Map(this.suppliedAssets);
    if (this.prefetched?.voice === voice) {
      const { assets } = this.prefetched;
      this.prefetched = undefined;
      return assets;
    }
    const module = this.mod!;
    if (!module.ttsDependencies) {
      throw new Error('TTS manifests are unavailable in this build.');
    }
    const keysJson = module.ttsDependencies(this.languageCode, voice);
    const keys = JSON.parse(keysJson) as string[];
    const base = (this.assetBase ?? DEFAULT_TTS_ASSET_BASE).replace(/\/+$/, '');
    const downloader =
      this.downloader ??
      new AssetDownloader({ onProgress: wrapProgress(this.progressCallback) });
    // Downloading these as one named set (rather than a file at a time) lets
    // the downloader report progress across the whole voice.
    const urls = new Map<string, string>(
      keys.map((key) => [key, `${base}/${key.replace(/^\/+/, '')}`]),
    );
    return downloader.downloadNamedFiles(urls);
  }

  /**
   * Transcribes a clone clip so the vocoder knows what the reference voice
   * said. The speech-to-text model this needs is an implementation detail, so
   * it is loaded here rather than being the caller's problem.
   */
  private async transcribeClip(clip: Float32Array): Promise<string | undefined> {
    try {
      this.clipTranscriber ??= await Transcriber.load({
        language: sttLanguageFor(this.languageCode),
        modelArch: ModelArch.Base,
        onProgress: wrapProgress(this.progressCallback),
        module: this.mod,
        downloader: this.downloader,
      });
      const transcript = this.clipTranscriber.transcribe(clip, {
        sampleRate: 16000,
      });
      const text = transcript.lines
        .map((line) => line.text)
        .join(' ')
        .trim();
      return text || undefined;
    } catch {
      // Cloning still works without a transcript, just less faithfully, so a
      // failure here should not sink the whole operation.
      return undefined;
    }
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
      return { audio, sampleRate: source.sampleRate };
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

/** Views a mono Float32 PCM buffer as little-endian raw bytes (no copy). */
function floatPcmToBytes(pcm: Float32Array): Uint8Array {
  return new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
}

/** TTS languages are regional (`en_us`); speech-to-text ones are not (`en`). */
function sttLanguageFor(ttsLanguage: string): string {
  return ttsLanguage.split(/[_-]/)[0] || DEFAULT_LANGUAGE;
}
