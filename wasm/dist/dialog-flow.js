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
export const InputMode = {
    Free: 'free',
    Phrase: 'phrase',
};
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
    triggerPhrase;
    state = {};
    lastSpokenPrompt;
    constructor(triggerPhrase = '') {
        this.triggerPhrase = triggerPhrase;
    }
    say(text) {
        this.lastSpokenPrompt = text;
        return { kind: 'say', text };
    }
    ask(prompt, options = {}) {
        this.lastSpokenPrompt = prompt;
        return {
            kind: 'ask',
            prompt,
            mode: options.mode ?? InputMode.Free,
            timeoutMs: options.timeoutMs,
            noInputReprompt: options.noInputReprompt ?? "Sorry, I didn't catch that. {prompt}",
            maxRetries: options.maxRetries ?? 2,
        };
    }
    confirm(prompt, options = {}) {
        this.lastSpokenPrompt = prompt;
        return {
            kind: 'confirm',
            prompt,
            maxRetries: options.maxRetries ?? 1,
            yesPhrases: options.yesPhrases ?? DEFAULT_YES,
            noPhrases: options.noPhrases ?? DEFAULT_NO,
            noInputReprompt: options.noInputReprompt ??
                "Sorry, I didn't catch that. Was that a yes or a no? {prompt}",
        };
    }
    choose(prompt, options, settings = {}) {
        this.lastSpokenPrompt = prompt;
        return {
            kind: 'choose',
            prompt,
            options,
            maxRetries: settings.maxRetries ?? 2,
            noInputReprompt: settings.noInputReprompt ?? "Sorry, I didn't catch that. {prompt}",
        };
    }
    cancel() {
        throw new DialogCancelled();
    }
    restart() {
        throw new DialogRestart();
    }
    replayLastPrompt() {
        return this.lastSpokenPrompt !== undefined
            ? { kind: 'say', text: this.lastSpokenPrompt }
            : undefined;
    }
}
/**
 * Runs generator-based conversational flows, driven by completed transcript
 * lines. Register flows against trigger phrases; when no flow is active, a
 * completed line is matched against triggers, otherwise it answers the pending
 * prompt.
 */
export class DialogFlow {
    options;
    flows = new Map();
    globals = new Map();
    active;
    speaking = false;
    /** Serializes async utterance processing so flows advance one at a time. */
    queue = Promise.resolve();
    triggersRegistered = false;
    constructor(options = {}) {
        this.options = { ignoreSttDuringTts: true, triggerThreshold: 0.7, ...options };
    }
    registerFlow(triggerPhrase, flow) {
        this.flows.set(triggerPhrase, flow);
        this.triggersRegistered = false;
    }
    registerGlobal(triggerPhrase, handler) {
        this.globals.set(triggerPhrase, handler);
        this.triggersRegistered = false;
    }
    get isActive() {
        return this.active !== undefined;
    }
    get activeTrigger() {
        return this.active?.triggerPhrase;
    }
    // --- TranscriptEventListener ---
    onLineCompleted(event) {
        const utterance = event.line.text.trim();
        if (!utterance)
            return;
        if (this.options.ignoreSttDuringTts && this.speaking)
            return;
        // Chain onto the queue so overlapping completed lines are serialized.
        this.queue = this.queue.then(() => this.processUtterance(utterance)).catch(() => { });
    }
    /** Speaks `text` outside any flow (welcome messages, announcements). */
    async say(text) {
        if (text)
            await this.speak(text);
    }
    // --- Core dispatch ---
    async processUtterance(utterance) {
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
    matchTrigger(utterance) {
        const phrase = this.bestTriggerPhrase(utterance);
        if (phrase === undefined)
            return undefined;
        if (this.globals.has(phrase))
            return { kind: 'global', phrase };
        if (this.flows.has(phrase))
            return { kind: 'flow', phrase };
        return undefined;
    }
    bestTriggerPhrase(utterance) {
        const phrases = [...this.globals.keys(), ...this.flows.keys()];
        if (phrases.length === 0)
            return undefined;
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
    async startFlow(triggerPhrase) {
        const flowFn = this.flows.get(triggerPhrase);
        if (!flowFn)
            return;
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
    async deliverToActive(active, utterance) {
        const prompt = active.currentPrompt;
        if (!prompt)
            return;
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
    async advance(active, value) {
        for (;;) {
            let res;
            try {
                res = active.generator.next(value);
            }
            catch (err) {
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
            if (prompt.prompt)
                await this.speak(prompt.prompt);
            return;
        }
    }
    async throwInto(active, err) {
        let res;
        try {
            res = active.generator.throw(err);
        }
        catch (thrown) {
            this.finishFlow(active);
            if (thrown instanceof DialogRestart)
                await this.restartFlow(active);
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
        }
        else {
            active.currentPrompt = prompt;
            active.retryCount = 0;
            if (prompt.prompt)
                await this.speak(prompt.prompt);
        }
    }
    async restartFlow(previous) {
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
    finishFlow(active) {
        if (this.active === active)
            this.active = undefined;
    }
    cancelActive() {
        if (!this.active)
            return false;
        try {
            this.active.generator.return?.();
        }
        catch {
            /* ignore */
        }
        this.finishFlow(this.active);
        return true;
    }
    async invokeGlobal(triggerPhrase) {
        const handler = this.globals.get(triggerPhrase);
        if (!handler)
            return;
        const dialog = this.active?.dialog ?? new Dialog(triggerPhrase);
        let prompt;
        try {
            prompt = handler(dialog);
        }
        catch (err) {
            if (err instanceof DialogCancelled) {
                if (this.active)
                    this.finishFlow(this.active);
            }
            else if (err instanceof DialogRestart) {
                if (this.active)
                    await this.restartFlow(this.active);
            }
            return;
        }
        if (prompt?.kind === 'say')
            await this.speak(prompt.text);
    }
    // --- Answer interpretation ---
    interpret(prompt, utterance) {
        const text = utterance.trim();
        switch (prompt.kind) {
            case 'ask':
                if (!text)
                    return this.reprompt(prompt);
                return { status: 'ok', value: text };
            case 'confirm': {
                if (matchesAny(text, prompt.yesPhrases))
                    return { status: 'ok', value: true };
                if (matchesAny(text, prompt.noPhrases))
                    return { status: 'ok', value: false };
                return this.reprompt(prompt);
            }
            case 'choose': {
                for (const [key, phrases] of Object.entries(prompt.options)) {
                    if (matchesAny(text, [key, ...phrases]))
                        return { status: 'ok', value: key };
                }
                return this.reprompt(prompt);
            }
            default:
                return { status: 'ok', value: text };
        }
    }
    reprompt(prompt) {
        const template = ('noInputReprompt' in prompt && prompt.noInputReprompt) || '{prompt}';
        const promptText = 'prompt' in prompt ? prompt.prompt : '';
        return { status: 'reprompt', text: template.replace('{prompt}', promptText) };
    }
    // --- TTS ---
    async speak(text) {
        if (!text)
            return;
        this.speaking = true;
        try {
            await this.options.muteFn?.(true);
            if (this.options.speakFn) {
                await this.options.speakFn(text);
            }
            else if (this.options.tts) {
                await this.options.tts.speak(text, this.options.audioContext);
            }
            else {
                // eslint-disable-next-line no-console
                console.log(`[DialogFlow say] ${text}`);
            }
        }
        finally {
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
function getMaxRetries(prompt) {
    return 'maxRetries' in prompt ? prompt.maxRetries : 0;
}
function matchesAny(utterance, phrases) {
    const lower = utterance.toLowerCase();
    return phrases.some((p) => {
        const needle = p.toLowerCase();
        return lower === needle || lower.includes(needle);
    });
}
/** Renders a string as a space-separated spoken form for reading back. */
export function spellOut(s) {
    return s.split('').join(' ');
}
//# sourceMappingURL=dialog-flow.js.map