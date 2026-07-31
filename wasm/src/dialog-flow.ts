/**
 * Voice dialogs: the one-call way to build a speech interface.
 *
 * ```ts
 * const dialog = new DialogFlow();
 *
 * dialog.listenFor('set up wifi', async (d) => {
 *   const ssid = await d.ask("What's the name of your wifi network?");
 *   if (await d.confirm(`I heard ${ssid}. Is that right?`)) {
 *     await d.say(`Done. Connecting to ${ssid}.`);
 *   }
 * });
 *
 * await dialog.load();
 * await dialog.startListening();
 * ```
 *
 * `load()` downloads and wires everything a voice interface needs: a streaming
 * speech-to-text model, an embedding model for matching trigger phrases, a
 * text-to-speech voice, and a microphone. A flow is an ordinary async function,
 * so it reads top to bottom and `try` / `finally` work the way you expect.
 *
 * Scope note vs. the Python runner: this port implements free-form asks plus
 * confirm/choose matching. The alphanumeric dictation subsystem and the
 * success/error beep diagnostics are not here yet.
 */

import { AssetDownloader } from './asset-downloader.js';
import { ModelArch } from './enums.js';
import type { LineCompleted, TranscriptEventListener } from './events.js';
import {
  EmbeddingModel,
  PhraseMatcher,
  type PhraseGroup,
} from './embedding-model.js';
import {
  MicTranscriber,
  wrapProgress,
  type ProgressCallback,
} from './mic-transcriber.js';
import { loadMoonshineModule, type MoonshineModule } from './module.js';
import { TextToSpeech } from './text-to-speech.js';

const DEFAULT_YES = [
  'yes', 'yeah', 'yep', 'correct', "that's right", 'sure', 'affirmative',
  'okay', 'please do', 'do it',
];
const DEFAULT_NO = [
  'no', 'nope', 'incorrect', "that's wrong", 'negative', 'cancel',
  "don't do it", 'stop',
];

const DEFAULT_TRIGGER_THRESHOLD = 0.7;
/**
 * Prompt answers ("yes", "the blue one") are short and varied, so they match on
 * a looser threshold than trigger phrases.
 */
const PROMPT_THRESHOLD = 0.55;

/** Thrown into a flow when the user (or a global handler) cancels it. */
export class DialogCancelled extends Error {
  constructor() {
    super('DialogCancelled');
    this.name = 'DialogCancelled';
  }
}

/** Thrown into a flow when it should start again from the top. */
export class DialogRestart extends Error {
  constructor() {
    super('DialogRestart');
    this.name = 'DialogRestart';
  }
}

/** Thrown out of `ask` / `confirm` / `choose` after the retries run out. */
export class DialogNoMatch extends Error {
  constructor(message = 'No matching answer') {
    super(message);
    this.name = 'DialogNoMatch';
  }
}

export interface AskOptions {
  /** Give up waiting after this long and re-prompt. */
  timeoutMs?: number;
  /** Spoken when the answer wasn't understood. `{prompt}` is substituted. */
  reprompt?: string;
  /** How many times to re-prompt before giving up. Defaults to 2. */
  maxRetries?: number;
}

export interface ConfirmOptions extends AskOptions {
  yesPhrases?: string[];
  noPhrases?: string[];
}

/**
 * The conversation, handed to a flow as its only argument. Every method speaks
 * and then waits, so a flow is just straight-line code.
 */
export class Dialog {
  /** The phrase that started this flow. */
  readonly triggerPhrase: string;
  /** Scratch space for the flow's own use; the runner never touches it. */
  readonly state: Record<string, unknown> = {};

  private readonly runner: DialogFlow;

  constructor(runner: DialogFlow, triggerPhrase = '') {
    this.runner = runner;
    this.triggerPhrase = triggerPhrase;
  }

  /** Speaks `text` and waits for playback to finish. */
  async say(text: string): Promise<void> {
    await this.runner.speakInFlow(text);
  }

  /** Asks an open question and returns what the user said. */
  async ask(prompt: string, options: AskOptions = {}): Promise<string> {
    return this.runner.promptForAnswer<string>(prompt, options, (text) =>
      text ? { ok: true, value: text } : { ok: false },
    );
  }

  /** Asks a yes/no question. */
  async confirm(prompt: string, options: ConfirmOptions = {}): Promise<boolean> {
    const yes = options.yesPhrases ?? DEFAULT_YES;
    const no = options.noPhrases ?? DEFAULT_NO;
    return this.runner.promptForAnswer<boolean>(
      prompt,
      {
        maxRetries: 1,
        reprompt: "Sorry, I didn't catch that. Was that a yes or a no? {prompt}",
        ...options,
      },
      (text) => {
        const key = this.runner.matchKey(text, [
          { key: 'yes', phrases: yes },
          { key: 'no', phrases: no },
        ]);
        if (key === 'yes') return { ok: true, value: true };
        if (key === 'no') return { ok: true, value: false };
        return { ok: false };
      },
    );
  }

  /**
   * Offers a set of choices and returns the key of the one picked. Each key
   * maps to the phrases that select it; the key itself always counts.
   */
  async choose(
    prompt: string,
    options: Record<string, string[]>,
    settings: AskOptions = {},
  ): Promise<string> {
    return this.runner.promptForAnswer<string>(prompt, settings, (text) => {
      // The key itself always counts as one of its phrases.
      const groups = Object.entries(options).map(([key, phrases]) => ({
        key,
        phrases: [key, ...phrases],
      }));
      const match = this.runner.matchKey(text, groups);
      return match ? { ok: true, value: match } : { ok: false };
    });
  }

  /** Abandons the flow. */
  cancel(): never {
    throw new DialogCancelled();
  }

  /** Runs the flow again from the beginning. */
  restart(): never {
    throw new DialogRestart();
  }
}

export type FlowFn = (dialog: Dialog) => void | Promise<void>;
export type GlobalHandler = (dialog: Dialog) => void | Promise<void>;

interface PendingAnswer {
  resolve(text: string): void;
  reject(error: Error): void;
  timer?: ReturnType<typeof setTimeout>;
}

type Interpretation<T> = { ok: true; value: T } | { ok: false };

export class DialogFlow {
  private readonly flows = new Map<string, FlowFn>();
  private readonly globals = new Map<string, GlobalHandler>();

  private languageCode = 'en';
  private arch: ModelArch = ModelArch.MediumStreaming;
  private voiceId?: string;
  private wantsMicrophone = true;
  private threshold = DEFAULT_TRIGGER_THRESHOLD;
  private assetBase?: string;
  private context?: AudioContext;
  private progressCallback?: ProgressCallback;
  private speakOverride?: (text: string) => void | Promise<void>;
  private heardCallbacks: Array<(text: string) => void> = [];
  private saidCallbacks: Array<(text: string) => void> = [];
  private errorCallbacks: Array<(error: Error) => void> = [];

  private mod?: MoonshineModule;
  private sharedDownloader?: AssetDownloader;
  private tts?: TextToSpeech;
  private embedding?: EmbeddingModel;
  private matcher = new PhraseMatcher();
  private mic?: MicTranscriber;
  private micConstraints: MediaTrackConstraints | boolean = true;
  private ownsTts = true;
  private ownsMic = true;

  private activeDialog?: Dialog;
  private activeTriggerPhrase?: string;
  private pending?: PendingAnswer;
  private speaking = false;
  /** Serializes utterance handling so one flow advances at a time. */
  private queue: Promise<void> = Promise.resolve();
  /**
   * Woken when the runner comes to rest, meaning the flow either finished or
   * is parked waiting for the next thing the user says. Handing an utterance
   * in resolves at that point rather than when the whole flow completes, which
   * would deadlock: the flow is waiting for the utterance after this one.
   */
  private settleWaiters: Array<() => void> = [];

  constructor() {
    // "cancel" and "start over" are what people actually say to a voice
    // interface, so they work without every application registering them.
    this.globals.set('cancel', (d) => d.cancel());
    this.globals.set('start over', (d) => d.restart());
  }

  // --- Configuration ---

  /** Speech-to-text and synthesis language. Defaults to `"en"`. */
  language(code: string): this {
    this.languageCode = code;
    return this;
  }

  /** Overrides the streaming speech-to-text model. */
  modelArch(arch: ModelArch): this {
    this.arch = arch;
    return this;
  }

  /** Voice used for spoken prompts, e.g. `"kokoro_af_heart"`. */
  voice(id: string): this {
    this.voiceId = id;
    return this;
  }

  /** Fetches all model assets from a base URL you host instead of the CDN. */
  modelsFrom(baseUrl: string): this {
    this.assetBase = baseUrl;
    return this;
  }

  /** Set to false to drive the dialog from text instead of a microphone. */
  microphone(enabled: boolean): this {
    this.wantsMicrophone = enabled;
    return this;
  }

  /**
   * Constraints for the microphone this opens, e.g. to name a capture device
   * rather than accept the browser's default. Ignored when a transcriber is
   * supplied through {@link useMicTranscriber}, which brings its own.
   */
  audioConstraints(constraints: MediaTrackConstraints | boolean): this {
    this.micConstraints = constraints;
    return this;
  }

  /** Similarity a trigger phrase needs to match, 0 to 1. Defaults to 0.7. */
  triggerThreshold(threshold: number): this {
    this.threshold = threshold;
    return this;
  }

  audioContext(context: AudioContext): this {
    this.context = context;
    return this;
  }

  /** Combined download progress for every model, as a `0..1` fraction. */
  onProgress(callback: ProgressCallback): this {
    this.progressCallback = callback;
    return this;
  }

  /** Called with each thing the user says. */
  onHeard(callback: (text: string) => void): this {
    this.heardCallbacks.push(callback);
    return this;
  }

  /** Called with each thing the assistant says. */
  onSaid(callback: (text: string) => void): this {
    this.saidCallbacks.push(callback);
    return this;
  }

  /** Called when a flow throws something the runner doesn't handle itself. */
  onError(callback: (error: Error) => void): this {
    this.errorCallbacks.push(callback);
    return this;
  }

  /** Replaces the built-in synthesizer, e.g. to route prompts somewhere else. */
  speakWith(speak: (text: string) => void | Promise<void>): this {
    this.speakOverride = speak;
    return this;
  }

  /** Registers a flow to run when the user says something like `phrase`. */
  listenFor(phrase: string, flow: FlowFn): this {
    this.flows.set(phrase, flow);
    return this;
  }

  /**
   * Registers a handler that runs whenever `phrase` is heard, even in the
   * middle of a flow. This is how `cancel` and `start over` are implemented.
   */
  always(phrase: string, handler: GlobalHandler): this {
    this.globals.set(phrase, handler);
    return this;
  }

  useModule(module: MoonshineModule): this {
    this.mod = module;
    return this;
  }

  useDownloader(downloader: AssetDownloader): this {
    this.sharedDownloader = downloader;
    return this;
  }

  useTextToSpeech(tts: TextToSpeech): this {
    this.tts = tts;
    this.ownsTts = false;
    return this;
  }

  useMicTranscriber(mic: MicTranscriber): this {
    this.mic = mic;
    this.ownsMic = false;
    return this;
  }

  // --- Lifecycle ---

  /** Downloads and wires every model the dialog needs. */
  async load(): Promise<this> {
    this.mod ??= await loadMoonshineModule();
    const progress = this.progressCallback;

    if (this.assetBase && !this.sharedDownloader) {
      // The embedding model is fetched through a manifest, so redirecting it to a
      // self-hosted base URL happens at the downloader.
      this.sharedDownloader = new AssetDownloader({
        baseUrl: this.assetBase,
        onProgress: wrapProgress(progress),
      });
    }

    if (!this.tts) {
      const tts = new TextToSpeech().language(this.languageCode).useModule(this.mod);
      if (this.voiceId) tts.voice(this.voiceId);
      if (this.assetBase) tts.modelsFrom(this.assetBase);
      if (this.context) tts.audioContext(this.context);
      if (progress) tts.onProgress(progress);
      if (this.sharedDownloader) tts.useDownloader(this.sharedDownloader);
      this.tts = await tts.load();
    }

    if (!this.embedding) {
      this.embedding = await EmbeddingModel.load({
        module: this.mod,
        downloader: this.sharedDownloader,
        onProgress: wrapProgress(progress),
      });
      this.matcher = new PhraseMatcher(this.embedding);
    }

    if (this.wantsMicrophone && !this.mic) {
      const mic = new MicTranscriber()
        .language(this.languageCode)
        .modelArch(this.arch)
        .audioConstraints(this.micConstraints);
      if (this.assetBase) mic.modelsFrom(this.assetBase);
      if (progress) mic.onProgress(progress);
      this.mic = await mic.load();
    }
    this.mic?.addListener(this.transcriptListener());
    return this;
  }

  /** Opens the microphone and starts responding to trigger phrases. */
  async startListening(): Promise<void> {
    if (!this.mic) {
      throw new Error(
        'No microphone. Call load() first, or use handleUtterance() for text input.',
      );
    }
    await this.mic.start();
  }

  async stopListening(): Promise<void> {
    await this.mic?.stop();
  }

  /** Says something outside any flow, e.g. a welcome message. */
  async say(text: string): Promise<void> {
    if (text) await this.speak(text);
  }

  /**
   * Feeds in an utterance the dialog didn't hear itself. Useful for text input
   * and for tests. Resolves once the flow has advanced as far as it can.
   */
  handleUtterance(text: string): Promise<void> {
    const utterance = text.trim();
    if (!utterance) return Promise.resolve();
    for (const cb of this.heardCallbacks) cb(utterance);
    this.queue = this.queue
      .then(() => this.dispatch(utterance))
      .catch(() => {});
    return this.queue;
  }

  /** True while a flow is running. */
  get isActive(): boolean {
    return this.activeDialog !== undefined;
  }

  /** The trigger phrase of the running flow, if any. */
  get activeTrigger(): string | undefined {
    return this.activeTriggerPhrase;
  }

  /** Abandons the running flow. Returns false if there wasn't one. */
  cancel(): boolean {
    if (!this.activeDialog) return false;
    this.rejectPending(new DialogCancelled());
    this.activeDialog = undefined;
    this.activeTriggerPhrase = undefined;
    return true;
  }

  close(): void {
    if (this.ownsMic) this.mic?.close();
    if (this.ownsTts) this.tts?.close();
    this.embedding?.close();
    this.mic = undefined;
    this.tts = undefined;
    this.embedding = undefined;
    this.matcher = new PhraseMatcher();
  }

  // --- Internals used by Dialog ---

  /** @internal */
  async speakInFlow(text: string): Promise<void> {
    await this.speak(text);
  }

  /**
   * Speaks a prompt, waits for an answer, and re-prompts until `interpret`
   * accepts one or the retries run out.
   * @internal
   */
  async promptForAnswer<T>(
    prompt: string,
    options: AskOptions,
    interpret: (text: string) => Interpretation<T>,
  ): Promise<T> {
    const maxRetries = options.maxRetries ?? 2;
    const reprompt = options.reprompt ?? "Sorry, I didn't catch that. {prompt}";
    for (let attempt = 0; ; attempt++) {
      const line = attempt === 0 ? prompt : reprompt.replace('{prompt}', prompt);
      await this.speak(line);

      let answer: string;
      try {
        answer = await this.waitForAnswer(options.timeoutMs);
      } catch (err) {
        if (err instanceof DialogNoMatch && attempt < maxRetries) continue;
        throw err;
      }

      const result = interpret(answer.trim());
      if (result.ok) return result.value;
      if (attempt >= maxRetries) {
        throw new DialogNoMatch(`Gave up understanding: "${answer}"`);
      }
    }
  }

  // --- Internals ---

  private transcriptListener(): TranscriptEventListener {
    return {
      onLineCompleted: (event: LineCompleted) => {
        if (this.speaking) return; // don't transcribe our own voice
        void this.handleUtterance(event.line.text);
      },
    };
  }

  private waitForAnswer(timeoutMs?: number): Promise<string> {
    const answer = new Promise<string>((resolve, reject) => {
      const entry: PendingAnswer = { resolve, reject };
      if (timeoutMs !== undefined) {
        entry.timer = setTimeout(() => {
          if (this.pending === entry) this.pending = undefined;
          reject(new DialogNoMatch('Timed out waiting for an answer'));
        }, timeoutMs);
      }
      this.pending = entry;
    });
    this.notifySettled();
    return answer;
  }

  /** Resolves the next time the runner finishes a flow or parks on a prompt. */
  private settledSignal(): Promise<void> {
    return new Promise<void>((resolve) => this.settleWaiters.push(resolve));
  }

  private notifySettled(): void {
    const waiters = this.settleWaiters;
    this.settleWaiters = [];
    for (const resolve of waiters) resolve();
  }

  private resolvePending(text: string): boolean {
    const entry = this.pending;
    if (!entry) return false;
    this.pending = undefined;
    if (entry.timer) clearTimeout(entry.timer);
    entry.resolve(text);
    return true;
  }

  private rejectPending(error: Error): void {
    const entry = this.pending;
    if (!entry) return;
    this.pending = undefined;
    if (entry.timer) clearTimeout(entry.timer);
    entry.reject(error);
  }

  private async dispatch(utterance: string): Promise<void> {
    // Globals win over everything, so "cancel" works mid-question.
    const trigger = this.matchTrigger(utterance);
    if (trigger && this.globals.has(trigger)) {
      const settled = this.settledSignal();
      await this.invokeGlobal(trigger);
      // A global that cancelled or restarted the flow left it unwinding, so
      // wait for it to come to rest. One that just spoke did not.
      if (this.activeDialog && !this.pending) await settled;
      return;
    }
    if (this.pending) {
      const settled = this.settledSignal();
      this.resolvePending(utterance);
      await settled;
      return;
    }
    if (this.activeDialog) return; // busy between prompts; drop the line
    if (trigger && this.flows.has(trigger)) {
      const settled = this.settledSignal();
      void this.runFlow(trigger);
      await settled;
    }
  }

  private matchTrigger(utterance: string): string | undefined {
    const phrases = [...this.globals.keys(), ...this.flows.keys()];
    if (phrases.length === 0) return undefined;

    return this.matcher.matchPhrases(utterance, phrases, this.threshold);
  }

  /**
   * The key of the group whose phrases best match `utterance`, used by
   * `Dialog.confirm` and `Dialog.choose`.
   *
   * @internal
   */
  matchKey(utterance: string, groups: PhraseGroup[]): string | undefined {
    return this.matcher.match(utterance, groups, PROMPT_THRESHOLD);
  }

  private async runFlow(triggerPhrase: string): Promise<void> {
    const flow = this.flows.get(triggerPhrase);
    if (!flow) return;
    try {
      for (;;) {
        const dialog = new Dialog(this, triggerPhrase);
        this.activeDialog = dialog;
        this.activeTriggerPhrase = triggerPhrase;
        try {
          await flow(dialog);
          return;
        } catch (err) {
          if (err instanceof DialogRestart) continue; // round again
          if (err instanceof DialogCancelled) return;
          if (err instanceof DialogNoMatch) {
            await this.speak("Sorry, I didn't get that. Let's start over.");
            return;
          }
          for (const cb of this.errorCallbacks) {
            cb(err instanceof Error ? err : new Error(String(err)));
          }
          return;
        }
      }
    } finally {
      this.activeDialog = undefined;
      this.activeTriggerPhrase = undefined;
      this.notifySettled();
    }
  }

  private async invokeGlobal(triggerPhrase: string): Promise<void> {
    const handler = this.globals.get(triggerPhrase);
    if (!handler) return;
    const dialog = this.activeDialog ?? new Dialog(this, triggerPhrase);
    try {
      await handler(dialog);
    } catch (err) {
      if (err instanceof DialogCancelled || err instanceof DialogRestart) {
        // Hand the interruption to the flow, which is parked in an `await`.
        if (this.pending) {
          this.rejectPending(err);
        } else if (err instanceof DialogCancelled) {
          this.activeDialog = undefined;
          this.activeTriggerPhrase = undefined;
        }
        return;
      }
      throw err;
    }
  }

  private async speak(text: string): Promise<void> {
    if (!text) return;
    for (const cb of this.saidCallbacks) cb(text);
    this.speaking = true;
    this.mic?.mute(true);
    try {
      if (this.speakOverride) {
        await this.speakOverride(text);
      } else if (this.tts) {
        await this.tts.say(text);
      } else {
        // eslint-disable-next-line no-console
        console.log(`[DialogFlow] ${text}`);
      }
    } finally {
      this.mic?.mute(false);
      this.speaking = false;
    }
  }
}

/** Renders a string as a space-separated spoken form for reading back. */
export function spellOut(value: string): string {
  return value.split('').join(' ');
}
