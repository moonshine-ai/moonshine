/**
 * Generator-based dialog-flow runner, a TypeScript port of the Python
 * `dialog_flow.py`. A *flow* is a generator function that yields prompts and
 * resumes with the user's answer:
 *
 * ```ts
 * function* setupWifi(d: Dialog) {
 *   const ssid = yield d.ask("What's the name of your wifi network?");
 *   if (!(yield d.confirm(`I heard ${ssid}. Is that right?`))) {
 *     yield d.say("No problem, let's start over.");
 *     return;
 *   }
 *   yield d.say(`Great, connecting to ${ssid}.`);
 * }
 * ```
 *
 * The runner is a {@link TranscriptEventListener}, so it composes with a
 * {@link Stream} / {@link MicrophoneTranscriber} exactly like the Python
 * version composes with `MicTranscriber`. Because the browser TTS path is
 * asynchronous, the runner is async internally (it awaits playback between
 * generator sends) and serializes incoming utterances.
 *
 * Scope note vs. the Python runner: this initial web port implements FREE-form
 * asks plus confirm/choose matching. The SPELLED/DIGITS alphanumeric dictation
 * subsystem and the success/error beep diagnostics are intentionally omitted
 * for now (they depend on helpers outside the WASM binding).
 */

import type { IntentRecognizer } from './intent-recognizer.js';
import type { LineCompleted, TranscriptEventListener } from './events.js';
import type { TextToSpeech } from './text-to-speech.js';

export const InputMode = {
  Free: 'free',
  Phrase: 'phrase',
} as const;
export type InputMode = (typeof InputMode)[keyof typeof InputMode];

// --- Prompt objects a flow yields to the runner ----------------------------

export interface Say {
  readonly kind: 'say';
  readonly text: string;
}

export interface Ask {
  readonly kind: 'ask';
  readonly prompt: string;
  readonly mode: InputMode;
  readonly timeoutMs?: number;
  readonly noInputReprompt?: string;
  readonly maxRetries: number;
}

export interface Confirm {
  readonly kind: 'confirm';
  readonly prompt: string;
  readonly maxRetries: number;
  readonly yesPhrases: readonly string[];
  readonly noPhrases: readonly string[];
  readonly noInputReprompt?: string;
}

export interface Choose {
  readonly kind: 'choose';
  readonly prompt: string;
  readonly options: Readonly<Record<string, readonly string[]>>;
  readonly maxRetries: number;
  readonly noInputReprompt?: string;
}

export type Prompt = Say | Ask | Confirm | Choose;

const DEFAULT_YES = [
  'yes', 'yeah', 'yep', 'correct', "that's right", 'sure', 'affirmative',
  'okay', 'please do', 'do it',
];
const DEFAULT_NO = [
  'no', 'nope', 'incorrect', "that's wrong", 'negative', 'cancel',
  "don't do it", 'stop',
];

// --- Flow-control exceptions thrown into / out of the generator ------------

export class DialogCancelled extends Error {
  constructor() {
    super('DialogCancelled');
    this.name = 'DialogCancelled';
  }
}
export class DialogRestart extends Error {
  constructor() {
    super('DialogRestart');
    this.name = 'DialogRestart';
  }
}

/** Context object handed to a flow as its first argument. Performs no I/O. */
export class Dialog {
  readonly triggerPhrase: string;
  readonly state: Record<string, unknown> = {};
  private lastSpokenPrompt?: string;

  constructor(triggerPhrase = '') {
    this.triggerPhrase = triggerPhrase;
  }

  say(text: string): Say {
    this.lastSpokenPrompt = text;
    return { kind: 'say', text };
  }

  ask(
    prompt: string,
    options: {
      mode?: InputMode;
      timeoutMs?: number;
      noInputReprompt?: string;
      maxRetries?: number;
    } = {},
  ): Ask {
    this.lastSpokenPrompt = prompt;
    return {
      kind: 'ask',
      prompt,
      mode: options.mode ?? InputMode.Free,
      timeoutMs: options.timeoutMs,
      noInputReprompt:
        options.noInputReprompt ?? "Sorry, I didn't catch that. {prompt}",
      maxRetries: options.maxRetries ?? 2,
    };
  }

  confirm(
    prompt: string,
    options: {
      maxRetries?: number;
      yesPhrases?: string[];
      noPhrases?: string[];
      noInputReprompt?: string;
    } = {},
  ): Confirm {
    this.lastSpokenPrompt = prompt;
    return {
      kind: 'confirm',
      prompt,
      maxRetries: options.maxRetries ?? 1,
      yesPhrases: options.yesPhrases ?? DEFAULT_YES,
      noPhrases: options.noPhrases ?? DEFAULT_NO,
      noInputReprompt:
        options.noInputReprompt ??
        "Sorry, I didn't catch that. Was that a yes or a no? {prompt}",
    };
  }

  choose(
    prompt: string,
    options: Record<string, string[]>,
    settings: { maxRetries?: number; noInputReprompt?: string } = {},
  ): Choose {
    this.lastSpokenPrompt = prompt;
    return {
      kind: 'choose',
      prompt,
      options,
      maxRetries: settings.maxRetries ?? 2,
      noInputReprompt:
        settings.noInputReprompt ?? "Sorry, I didn't catch that. {prompt}",
    };
  }

  cancel(): never {
    throw new DialogCancelled();
  }

  restart(): never {
    throw new DialogRestart();
  }

  replayLastPrompt(): Say | undefined {
    return this.lastSpokenPrompt !== undefined
      ? { kind: 'say', text: this.lastSpokenPrompt }
      : undefined;
  }
}

export type FlowFn = (d: Dialog) => Generator<Prompt, void, any>;
export type GlobalHandler = (d: Dialog) => Prompt | void;

export interface DialogFlowOptions {
  /** TTS used to speak prompts. `speakFn` overrides it. */
  tts?: TextToSpeech;
  /** Custom speak function: `(text) => Promise<void>` resolving after playback. */
  speakFn?: (text: string) => void | Promise<void>;
  /** Intent recognizer used for embedding-based trigger matching. */
  intentRecognizer?: IntentRecognizer;
  /** Similarity threshold for trigger matching (0–1). */
  triggerThreshold?: number;
  /** Invoked with `true` before speaking and `false` after (mic muting). */
  muteFn?: (mute: boolean) => void | Promise<void>;
  /** Drop utterances that arrive while the assistant is speaking. */
  ignoreSttDuringTts?: boolean;
  /** Optional shared AudioContext for TTS playback. */
  audioContext?: AudioContext;
}

interface ActiveFlow {
  flowFn: FlowFn;
  triggerPhrase: string;
  dialog: Dialog;
  generator: Generator<Prompt, void, any>;
  currentPrompt?: Prompt;
  retryCount: number;
}

/**
 * Runs generator-based conversational flows, driven by completed transcript
 * lines. Register flows against trigger phrases; when no flow is active, a
 * completed line is matched against triggers, otherwise it answers the pending
 * prompt.
 */
export class DialogFlow implements TranscriptEventListener {
  private readonly options: DialogFlowOptions;
  private readonly flows = new Map<string, FlowFn>();
  private readonly globals = new Map<string, GlobalHandler>();
  private active?: ActiveFlow;
  private speaking = false;
  /** Serializes async utterance processing so flows advance one at a time. */
  private queue: Promise<void> = Promise.resolve();
  private triggersRegistered = false;

  constructor(options: DialogFlowOptions = {}) {
    this.options = { ignoreSttDuringTts: true, triggerThreshold: 0.7, ...options };
  }

  registerFlow(triggerPhrase: string, flow: FlowFn): void {
    this.flows.set(triggerPhrase, flow);
    this.triggersRegistered = false;
  }

  registerGlobal(triggerPhrase: string, handler: GlobalHandler): void {
    this.globals.set(triggerPhrase, handler);
    this.triggersRegistered = false;
  }

  get isActive(): boolean {
    return this.active !== undefined;
  }

  get activeTrigger(): string | undefined {
    return this.active?.triggerPhrase;
  }

  // --- TranscriptEventListener ---

  onLineCompleted(event: LineCompleted): void {
    const utterance = event.line.text.trim();
    if (!utterance) return;
    if (this.options.ignoreSttDuringTts && this.speaking) return;
    // Chain onto the queue so overlapping completed lines are serialized.
    this.queue = this.queue.then(() => this.processUtterance(utterance)).catch(() => {});
  }

  /** Speaks `text` outside any flow (welcome messages, announcements). */
  async say(text: string): Promise<void> {
    if (text) await this.speak(text);
  }

  // --- Core dispatch ---

  private async processUtterance(utterance: string): Promise<void> {
    const trigger = this.matchTrigger(utterance);

    if (trigger?.kind === 'global') {
      await this.invokeGlobal(trigger.phrase);
      return;
    }
    if (this.active) {
      await this.deliverToActive(this.active, utterance);
      return;
    }
    if (trigger?.kind === 'flow') {
      await this.startFlow(trigger.phrase);
    }
  }

  private matchTrigger(
    utterance: string,
  ): { kind: 'global' | 'flow'; phrase: string } | undefined {
    const phrase = this.bestTriggerPhrase(utterance);
    if (phrase === undefined) return undefined;
    if (this.globals.has(phrase)) return { kind: 'global', phrase };
    if (this.flows.has(phrase)) return { kind: 'flow', phrase };
    return undefined;
  }

  private bestTriggerPhrase(utterance: string): string | undefined {
    const phrases = [...this.globals.keys(), ...this.flows.keys()];
    if (phrases.length === 0) return undefined;

    const recognizer = this.options.intentRecognizer;
    if (recognizer) {
      if (!this.triggersRegistered) {
        recognizer.clear();
        recognizer.register(phrases);
        this.triggersRegistered = true;
      }
      const best = recognizer.bestIntent(utterance, this.options.triggerThreshold ?? 0);
      return best?.canonicalPhrase;
    }
    // Fallback: case-insensitive substring match.
    const lower = utterance.toLowerCase();
    return phrases.find((p) => lower.includes(p.toLowerCase()));
  }

  private async startFlow(triggerPhrase: string): Promise<void> {
    const flowFn = this.flows.get(triggerPhrase);
    if (!flowFn) return;
    const dialog = new Dialog(triggerPhrase);
    this.active = {
      flowFn,
      triggerPhrase,
      dialog,
      generator: flowFn(dialog),
      retryCount: 0,
    };
    await this.advance(this.active, undefined);
  }

  private async deliverToActive(active: ActiveFlow, utterance: string): Promise<void> {
    const prompt = active.currentPrompt;
    if (!prompt) return;
    const result = this.interpret(prompt, utterance);

    if (result.status === 'reprompt') {
      active.retryCount++;
      if (active.retryCount > getMaxRetries(prompt)) {
        await this.throwInto(active, new NoMatchError());
        return;
      }
      await this.speak(result.text);
      return;
    }
    active.retryCount = 0;
    await this.advance(active, result.value);
  }

  private async advance(active: ActiveFlow, value: unknown): Promise<void> {
    for (;;) {
      let res: IteratorResult<Prompt, void>;
      try {
        res = active.generator.next(value);
      } catch (err) {
        this.finishFlow(active);
        if (err instanceof DialogRestart) {
          await this.restartFlow(active);
        }
        return;
      }
      if (res.done) {
        this.finishFlow(active);
        return;
      }
      const prompt = res.value;
      if (prompt.kind === 'say') {
        await this.speak(prompt.text);
        value = undefined;
        continue;
      }
      // Ask / Confirm / Choose: speak the prompt and wait for the next line.
      active.currentPrompt = prompt;
      active.retryCount = 0;
      if (prompt.prompt) await this.speak(prompt.prompt);
      return;
    }
  }

  private async throwInto(active: ActiveFlow, err: Error): Promise<void> {
    let res: IteratorResult<Prompt, void>;
    try {
      res = active.generator.throw(err);
    } catch (thrown) {
      this.finishFlow(active);
      if (thrown instanceof DialogRestart) await this.restartFlow(active);
      return;
    }
    if (res.done) {
      this.finishFlow(active);
      return;
    }
    const prompt = res.value;
    if (prompt.kind === 'say') {
      await this.speak(prompt.text);
      await this.advance(active, undefined);
    } else {
      active.currentPrompt = prompt;
      active.retryCount = 0;
      if (prompt.prompt) await this.speak(prompt.prompt);
    }
  }

  private async restartFlow(previous: ActiveFlow): Promise<void> {
    const dialog = new Dialog(previous.triggerPhrase);
    this.active = {
      flowFn: previous.flowFn,
      triggerPhrase: previous.triggerPhrase,
      dialog,
      generator: previous.flowFn(dialog),
      retryCount: 0,
    };
    await this.advance(this.active, undefined);
  }

  private finishFlow(active: ActiveFlow): void {
    if (this.active === active) this.active = undefined;
  }

  cancelActive(): boolean {
    if (!this.active) return false;
    try {
      this.active.generator.return?.();
    } catch {
      /* ignore */
    }
    this.finishFlow(this.active);
    return true;
  }

  private async invokeGlobal(triggerPhrase: string): Promise<void> {
    const handler = this.globals.get(triggerPhrase);
    if (!handler) return;
    const dialog = this.active?.dialog ?? new Dialog(triggerPhrase);
    let prompt: Prompt | void;
    try {
      prompt = handler(dialog);
    } catch (err) {
      if (err instanceof DialogCancelled) {
        if (this.active) this.finishFlow(this.active);
      } else if (err instanceof DialogRestart) {
        if (this.active) await this.restartFlow(this.active);
      }
      return;
    }
    if (prompt?.kind === 'say') await this.speak(prompt.text);
  }

  // --- Answer interpretation ---

  private interpret(
    prompt: Prompt,
    utterance: string,
  ):
    | { status: 'ok'; value: unknown }
    | { status: 'reprompt'; text: string } {
    const text = utterance.trim();
    switch (prompt.kind) {
      case 'ask':
        if (!text) return this.reprompt(prompt);
        return { status: 'ok', value: text };
      case 'confirm': {
        if (matchesAny(text, prompt.yesPhrases)) return { status: 'ok', value: true };
        if (matchesAny(text, prompt.noPhrases)) return { status: 'ok', value: false };
        return this.reprompt(prompt);
      }
      case 'choose': {
        for (const [key, phrases] of Object.entries(prompt.options)) {
          if (matchesAny(text, [key, ...phrases])) return { status: 'ok', value: key };
        }
        return this.reprompt(prompt);
      }
      default:
        return { status: 'ok', value: text };
    }
  }

  private reprompt(prompt: Prompt): { status: 'reprompt'; text: string } {
    const template =
      ('noInputReprompt' in prompt && prompt.noInputReprompt) || '{prompt}';
    const promptText = 'prompt' in prompt ? prompt.prompt : '';
    return { status: 'reprompt', text: template.replace('{prompt}', promptText) };
  }

  // --- TTS ---

  private async speak(text: string): Promise<void> {
    if (!text) return;
    this.speaking = true;
    try {
      await this.options.muteFn?.(true);
      if (this.options.speakFn) {
        await this.options.speakFn(text);
      } else if (this.options.tts) {
        await this.options.tts.speak(text, this.options.audioContext);
      } else {
        // eslint-disable-next-line no-console
        console.log(`[DialogFlow say] ${text}`);
      }
    } finally {
      await this.options.muteFn?.(false);
      this.speaking = false;
    }
  }
}

class NoMatchError extends Error {
  constructor() {
    super('NoMatchError');
    this.name = 'NoMatchError';
  }
}

function getMaxRetries(prompt: Prompt): number {
  return 'maxRetries' in prompt ? prompt.maxRetries : 0;
}

function matchesAny(utterance: string, phrases: readonly string[]): boolean {
  const lower = utterance.toLowerCase();
  return phrases.some((p) => {
    const needle = p.toLowerCase();
    return lower === needle || lower.includes(needle);
  });
}

/** Renders a string as a space-separated spoken form for reading back. */
export function spellOut(s: string): string {
  return s.split('').join(' ');
}
