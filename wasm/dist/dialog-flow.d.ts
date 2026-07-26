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
import type { AssetDownloader } from './asset-downloader.js';
import { ModelArch } from './enums.js';
import type { LineCompleted, TranscriptEventListener } from './events.js';
import { IntentRecognizer, type IntentRecognizerOptions } from './intent-recognizer.js';
import { MicrophoneTranscriber } from './microphone-transcriber.js';
import { type MoonshineModule } from './module.js';
import { TextToSpeech, type TextToSpeechOptions } from './text-to-speech.js';
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
 * Options for {@link DialogFlow.load}, which downloads and wires the TTS, intent, and (optionally)
 * microphone engines a voice dialog needs, then returns them all ready to use.
 *
 * Pass pre-loaded `tts` / `intentRecognizer` to reuse existing instances (or to load them from
 * self-hosted assets via their own `loadFromUrls` factories); otherwise they are fetched from the
 * Moonshine CDN. A single `module`, `onProgress`, and `downloader` are shared across every load.
 */
export interface DialogFlowLoadOptions {
    /** STT language for the microphone transcriber (e.g. `"en"`). Default `"en"`. */
    language?: string;
    /** Streaming architecture for the microphone transcriber. Default {@link ModelArch.MediumStreaming}. */
    modelArch?: ModelArch;
    /** Build a {@link MicrophoneTranscriber} wired to the runner. Default `true`. */
    microphone?: boolean;
    /** Extra listeners added to the mic alongside the runner (e.g. to log user lines). */
    micListeners?: TranscriptEventListener[];
    /** Constraints forwarded to the microphone transcriber. */
    audioConstraints?: MediaTrackConstraints | boolean;
    /** A pre-loaded TTS engine. When omitted, one is loaded (from `ttsOptions` or the CDN). */
    tts?: TextToSpeech;
    /** Options for loading TTS when `tts` is not supplied. Defaults to a CDN load for `language`. */
    ttsOptions?: TextToSpeechOptions;
    /** A pre-loaded intent recognizer. When omitted, one is loaded (from `intentOptions` or the CDN). */
    intentRecognizer?: IntentRecognizer;
    /** Options for loading the intent recognizer when one is not supplied. */
    intentOptions?: IntentRecognizerOptions;
    /** Flows to register on the runner, keyed by trigger phrase. */
    flows?: Record<string, FlowFn>;
    /** Global handlers to register on the runner, keyed by trigger phrase. */
    globals?: Record<string, GlobalHandler>;
    /** Forwarded to the {@link DialogFlow} constructor. */
    triggerThreshold?: number;
    ignoreSttDuringTts?: boolean;
    muteFn?: (mute: boolean) => void | Promise<void>;
    speakFn?: (text: string) => void | Promise<void>;
    audioContext?: AudioContext;
    /** Shared WASM module, progress callback, and downloader across all three loads. */
    module?: MoonshineModule;
    onProgress?: (loaded: number, total: number | undefined, file: string) => void;
    downloader?: AssetDownloader;
}
/** The wired-up engines returned by {@link DialogFlow.load}. */
export interface DialogFlowBundle {
    /** The runner, with `flows` / `globals` registered and driven by `mic`. */
    dialog: DialogFlow;
    /** The microphone transcriber, unless `microphone: false` was passed. */
    mic?: MicrophoneTranscriber;
    tts: TextToSpeech;
    intent: IntentRecognizer;
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
    /**
     * Downloads (or reuses) the TTS, intent, and microphone engines a voice dialog needs, wires them
     * together, registers the given flows/globals, and returns the ready {@link DialogFlow} plus the
     * engines. This is the one-call equivalent of the manual "load TTS + intent + mic, then `new
     * DialogFlow(...)`, then `registerFlow`, then `mic.addListener(runner)`" dance.
     *
     * Progress from all downloads is reported through a single `onProgress`. Call `await
     * bundle.mic?.start()` to begin listening.
     */
    static load(options?: DialogFlowLoadOptions): Promise<DialogFlowBundle>;
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