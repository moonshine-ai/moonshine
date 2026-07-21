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
export declare const InputMode: {
    readonly Free: "free";
    readonly Phrase: "phrase";
};
export type InputMode = (typeof InputMode)[keyof typeof InputMode];
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
export declare class DialogCancelled extends Error {
    constructor();
}
export declare class DialogRestart extends Error {
    constructor();
}
/** Context object handed to a flow as its first argument. Performs no I/O. */
export declare class Dialog {
    readonly triggerPhrase: string;
    readonly state: Record<string, unknown>;
    private lastSpokenPrompt?;
    constructor(triggerPhrase?: string);
    say(text: string): Say;
    ask(prompt: string, options?: {
        mode?: InputMode;
        timeoutMs?: number;
        noInputReprompt?: string;
        maxRetries?: number;
    }): Ask;
    confirm(prompt: string, options?: {
        maxRetries?: number;
        yesPhrases?: string[];
        noPhrases?: string[];
        noInputReprompt?: string;
    }): Confirm;
    choose(prompt: string, options: Record<string, string[]>, settings?: {
        maxRetries?: number;
        noInputReprompt?: string;
    }): Choose;
    cancel(): never;
    restart(): never;
    replayLastPrompt(): Say | undefined;
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
/**
 * Runs generator-based conversational flows, driven by completed transcript
 * lines. Register flows against trigger phrases; when no flow is active, a
 * completed line is matched against triggers, otherwise it answers the pending
 * prompt.
 */
export declare class DialogFlow implements TranscriptEventListener {
    private readonly options;
    private readonly flows;
    private readonly globals;
    private active?;
    private speaking;
    /** Serializes async utterance processing so flows advance one at a time. */
    private queue;
    private triggersRegistered;
    constructor(options?: DialogFlowOptions);
    registerFlow(triggerPhrase: string, flow: FlowFn): void;
    registerGlobal(triggerPhrase: string, handler: GlobalHandler): void;
    get isActive(): boolean;
    get activeTrigger(): string | undefined;
    onLineCompleted(event: LineCompleted): void;
    /** Speaks `text` outside any flow (welcome messages, announcements). */
    say(text: string): Promise<void>;
    private processUtterance;
    private matchTrigger;
    private bestTriggerPhrase;
    private startFlow;
    private deliverToActive;
    private advance;
    private throwInto;
    private restartFlow;
    private finishFlow;
    cancelActive(): boolean;
    private invokeGlobal;
    private interpret;
    private reprompt;
    private speak;
}
/** Renders a string as a space-separated spoken form for reading back. */
export declare function spellOut(s: string): string;
//# sourceMappingURL=dialog-flow.d.ts.map