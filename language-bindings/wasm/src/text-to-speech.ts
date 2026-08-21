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

/** No complete utterance is buffered yet; push more text or flush. */
const TTS_NEED_TEXT = 1;
/** Input ended and everything queued has been synthesized. */
const TTS_END_OF_STREAM = 2;
/**
 * A cancel discarded the reply being generated. Reported once, and only when
 * there was something to discard, so a consumer can tell an interruption from
 * having run out of text.
 */
const TTS_CANCELLED = 3;

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

/** Anything {@link TextToSpeech.stream} can take its text from. */
export type TtsTextSource = string | Iterable<string> | AsyncIterable<string>;

/** One piece of streamed audio from {@link TextToSpeech.stream}. */
export interface TtsChunk {
  audio: Float32Array;
  sampleRate: number;
  /** The text this chunk covers, or `''` when the engine cannot attribute it. */
  text: string;
  /** Which queued utterance this chunk belongs to, counting from one. */
  utteranceId: number;
  /** True for the last chunk of an utterance. */
  isFinal: boolean;
}

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

  /** End of already-scheduled playback, on the AudioContext clock. */
  private playbackTail = 0;
  /** Scheduled-but-unfinished sources, so {@link stop} can silence them. */
  private readonly scheduledSources = new Set<AudioBufferSourceNode>();

  /** The clip the current voice was cloned from, if any. */
  private cloneAudio?: Float32Array;
  private cloneTranscript?: string;

  /** Last {@link say} output, keyed by the spoken text, for instant replay. */
  private sayCache?: { text: string; chunks: TtsSynthesisResult[] };

  /** True once the streaming reply in flight has drained or been abandoned. */
  private streamEnded = false;
  /** Resolved when text arrives or the reply ends, so a waiting pull retries. */
  private streamWaiter?: () => void;

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
   * Moonshine CDN. Canonical names (e.g. `kokoro/prosody.model.ort`) are
   * appended.
   */
  modelsFrom(baseUrl: string): this {
    this.assetBase = baseUrl;
    return this;
  }

  /**
   * Supplies voice assets directly, keyed by canonical name (e.g.
   * `kokoro/prosody.model.ort`). Nothing is downloaded when this is set.
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
    const sentences = this.splitUtterances(text);
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

  /**
   * Queues `result` immediately after whatever is already scheduled and
   * resolves when it has finished playing. Scheduling against the running
   * clock rather than waiting for `onended` means consecutive chunks join
   * without an audible gap.
   */
  private playOne(result: TtsSynthesisResult): Promise<void> {
    if (result.audio.length === 0) return Promise.resolve();
    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(1, result.audio.length, result.sampleRate);
    buffer.copyToChannel(result.audio, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime, this.playbackTail);
    this.playbackTail = startAt + buffer.duration;
    this.scheduledSources.add(source);
    return new Promise((resolve) => {
      source.onended = () => {
        this.scheduledSources.delete(source);
        resolve();
      };
      source.start(startAt);
    });
  }

  /**
   * Queues already-synthesized audio — a {@link TtsChunk} from
   * {@link stream}, say — after whatever is already scheduled, so successive
   * chunks join without a gap. Resolves once this piece has played.
   */
  playChunk(chunk: { audio: Float32Array; sampleRate: number }): Promise<void> {
    return this.playOne({ audio: chunk.audio, sampleRate: chunk.sampleRate });
  }

  /** Resolves when everything scheduled has finished playing. */
  waitForPlayback(): Promise<void> {
    if (!this.context) return Promise.resolve();
    const remaining = this.playbackTail - this.context.currentTime;
    if (remaining <= 0) return Promise.resolve();
    return new Promise((resolve) => {
      setTimeout(resolve, remaining * 1000);
    });
  }

  /** Drops anything queued and silences playback immediately. */
  stop(): void {
    for (const source of this.scheduledSources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Already stopped, or never started; nothing to undo.
      }
      source.disconnect();
    }
    this.scheduledSources.clear();
    this.playbackTail = 0;
  }

  /** True while scheduled audio is still playing. */
  get isTalking(): boolean {
    if (!this.context) return false;
    return this.context.currentTime < this.playbackTail;
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

  /**
   * Splits `text` into the utterances {@link say} would speak one at a time,
   * using the shared native splitter.
   */
  splitUtterances(text: string): string[] {
    if (!this.mod) {
      throw new Error('Call load() before splitting text.');
    }
    return splitSayUtterancesWith(this.mod, text, this.languageCode);
  }

  // --- Streaming ---
  //
  // Text goes in as it is written and audio comes out in pieces, so the first
  // clause of a reply can play while the rest is still being generated. For
  // text you already have in full, `say` and `synthesize` are simpler.
  //
  // A synthesizer speaks one thing at a time: `pushText` starts a reply,
  // `endInput` finishes it, `cancelStream` abandons it, and `synthesize`
  // throws a `MoonshineBusyError` while one is in flight. There is no session
  // object to create or close.

  /**
   * Appends text to the reply being spoken, starting one if needed.
   *
   * Pieces are concatenated verbatim, so a model's output can go in token by
   * token. Text is held back until it forms a complete sentence, because
   * synthesizing half a clause gets the prosody wrong.
   */
  pushText(text: string): void {
    if (!text) return;
    this.ensureMainThreadEngine();
    wrapErrors(() => this.raw!.pushText(text));
    this.streamEnded = false;
    this.wakeStream();
  }

  /** Queues the buffered fragment even though it has no terminator. */
  flush(): void {
    this.ensureMainThreadEngine();
    wrapErrors(() => this.raw!.flush());
    this.wakeStream();
  }

  /** Declares that no more text is coming, letting the reply finish. */
  endInput(): void {
    this.ensureMainThreadEngine();
    wrapErrors(() => this.raw!.endInput());
    this.wakeStream();
  }

  /**
   * Drops queued text and abandons the reply in progress. This is the barge-in
   * path: when someone interrupts the assistant, stop the reply.
   */
  cancelStream(): void {
    this.ensureMainThreadEngine();
    wrapErrors(() => this.raw!.cancel());
    // An iteration in flight learns the reply was abandoned from its next
    // pull, not from here, so only wake one that is waiting for text. Ending
    // it without that pull would leave the cancellation to be reported
    // against whatever reply came next.
    this.wakeStream();
  }

  /** True while a reply is part-spoken. */
  get isStreaming(): boolean {
    if (!this.raw) return false;
    return wrapErrors(() => this.raw!.isStreaming());
  }

  /**
   * Synthesizes and returns the next chunk, blocking while it computes.
   *
   * `undefined` means there is nothing to hand back: either no complete
   * sentence is buffered yet, or the reply is over — drained, or discarded by
   * a {@link cancelStream}. {@link isStreaming} tells those apart. Prefer
   * {@link stream}, which yields between chunks so playback and the page get a
   * turn.
   */
  nextChunk(): TtsChunk | undefined {
    this.ensureMainThreadEngine();
    const chunk = wrapErrors(() => this.raw!.nextChunk());
    if (chunk.status === TTS_END_OF_STREAM || chunk.status === TTS_CANCELLED) {
      this.streamEnded = true;
      return undefined;
    }
    if (chunk.status === TTS_NEED_TEXT || !chunk.audio) return undefined;
    return {
      audio: chunk.audio,
      sampleRate: chunk.sampleRate,
      text: chunk.text,
      utteranceId: chunk.utteranceId,
      isFinal: chunk.isFinal,
    };
  }

  /**
   * The chunks of a reply, in order:
   *
   * ```ts
   * for await (const chunk of tts.stream(llm.tokens(question))) {
   *   await tts.playChunk(chunk);
   * }
   * ```
   *
   * With `text`, that whole reply is pushed and ended for you: pass a string,
   * or an iterable of pieces to forward as they arrive. Without it, call
   * {@link pushText} from elsewhere and iterate here. Iteration ends once
   * {@link endInput} lets the queue drain, or as soon as a
   * {@link cancelStream} elsewhere abandons the reply; leaving the loop early
   * abandons the rest of it.
   */
  async *stream(text?: TtsTextSource): AsyncGenerator<TtsChunk> {
    this.ensureMainThreadEngine();
    this.streamEnded = false;
    // The source runs alongside the pull loop rather than between chunks, so a
    // model that pauses mid-reply cannot hold back audio that is already due.
    let stopped = false;
    const feeding =
      text === undefined ? undefined : this.feedStream(text, () => stopped);
    // Nothing awaits `feeding` until the loop ends, so let a failing source
    // unwind the loop now; it is rethrown below.
    feeding?.catch(() => this.endStreamIteration());

    try {
      while (!this.streamEnded) {
        const chunk = this.nextChunk();
        if (chunk) {
          yield chunk;
          // Give playback, the page, and the text producer a turn.
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
          continue;
        }
        if (this.streamEnded) break;
        await this.waitForStream();
      }
    } finally {
      stopped = true;
      // A consumer that walked away, or a source that failed, leaves a
      // half-spoken reply the synthesizer would otherwise still be busy with.
      if (this.isStreaming) this.cancelStream();
      // A cancellation is held for the next pull, and nothing will pull on
      // this iteration's behalf again, so take it here rather than let it
      // land at the head of the reply after this one.
      if (!this.streamEnded) this.nextChunk();
    }
    await feeding;
  }

  close(): void {
    this.stop();
    this.endStreamIteration();
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

  /** Pushes every piece of `text`, giving up if the consumer walked away. */
  private async feedStream(
    text: TtsTextSource,
    stopped: () => boolean,
  ): Promise<void> {
    if (typeof text === 'string') {
      this.pushText(text);
    } else {
      for await (const piece of text) {
        if (stopped()) return;
        this.pushText(piece);
      }
    }
    if (!stopped()) this.endInput();
  }

  private waitForStream(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.streamWaiter = resolve;
    });
  }

  private wakeStream(): void {
    const waiter = this.streamWaiter;
    this.streamWaiter = undefined;
    waiter?.();
  }

  /** Ends any iteration in flight, for when the engine is about to go away. */
  private endStreamIteration(): void {
    this.streamEnded = true;
    this.wakeStream();
  }

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
    this.streamEnded = false;

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
 * Splits `text` the way {@link TextToSpeech.say} does, into the utterances a
 * synthesizer speaks one at a time. Uses the shared native splitter, so it
 * knows about abbreviations like `Dr.`, initials, quotes, and non-Latin
 * terminators such as `。` and `؟`.
 */
export async function splitSayUtterances(
  text: string,
  options: { language?: string; module?: MoonshineModule } = {},
): Promise<string[]> {
  const module = options.module ?? (await loadMoonshineModule());
  return splitSayUtterancesWith(module, text, options.language);
}

/** Synchronous split for callers that already hold a module. */
function splitSayUtterancesWith(
  module: MoonshineModule,
  text: string,
  language?: string,
): string[] {
  if (!module.ttsSplitUtterances) {
    throw new Error('TTS sentence splitting is unavailable in this build.');
  }
  const json = wrapErrors(() =>
    module.ttsSplitUtterances!(language ?? '', text),
  );
  return JSON.parse(json) as string[];
}

/** Views a mono Float32 PCM buffer as little-endian raw bytes (no copy). */
function floatPcmToBytes(pcm: Float32Array): Uint8Array {
  return new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
}
