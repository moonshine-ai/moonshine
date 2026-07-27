package ai.moonshine.voice;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.Nullable;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Voice dialogs: the one-call way to build a speech interface.
 *
 * <pre>{@code
 * DialogFlow dialog = new DialogFlow(this);
 *
 * dialog.listenFor("set up wifi", d -> {
 *     String ssid = d.ask("What's the name of your wifi network?");
 *     if (d.confirm("I heard " + ssid + ". Is that right?")) {
 *         d.say("Done. Connecting to " + ssid + ".");
 *     }
 * });
 *
 * executor.execute(() -> {
 *     dialog.load();
 *     dialog.startListening();
 * });
 * }</pre>
 *
 * <p>{@link #load()} downloads and wires everything a voice interface needs: a
 * streaming speech-to-text model, an intent model for matching trigger phrases,
 * a text-to-speech voice, and a microphone. A flow is ordinary blocking code
 * running on its own thread, so it reads top to bottom and {@code try} /
 * {@code finally} work the way you expect.
 */
public final class DialogFlow {

    private static final float DEFAULT_TRIGGER_THRESHOLD = 0.7f;
    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    /** A conversation, written as straight-line code. */
    public interface Flow {
        void run(Dialog dialog) throws Exception;
    }

    /** Thrown into a flow when the user (or a global handler) cancels it. */
    public static final class DialogCancelled extends RuntimeException {
        public DialogCancelled() {
            super("Dialog cancelled");
        }
    }

    /** Thrown into a flow when it should start again from the top. */
    public static final class DialogRestart extends RuntimeException {
        public DialogRestart() {
            super("Dialog restarted");
        }
    }

    /** Thrown out of {@code ask} / {@code confirm} / {@code choose} after the retries run out. */
    public static final class DialogNoMatch extends RuntimeException {
        public DialogNoMatch(String message) {
            super(message);
        }
    }

    /** How long to wait for an answer, and what to say when one doesn't arrive. */
    public static final class AskOptions {
        /** Give up waiting after this long and re-prompt. Zero waits forever. */
        public long timeoutMillis = 0;
        /** Spoken when the answer wasn't understood. {@code {prompt}} is substituted. */
        @Nullable public String reprompt;
        /** How many times to re-prompt before giving up. */
        public int maxRetries = 2;

        public AskOptions timeout(long millis) {
            this.timeoutMillis = millis;
            return this;
        }

        public AskOptions reprompt(String text) {
            this.reprompt = text;
            return this;
        }

        public AskOptions maxRetries(int retries) {
            this.maxRetries = retries;
            return this;
        }
    }

    /** What people say when they mean yes, matched by {@link Dialog#confirm}. */
    public static final List<String> DEFAULT_YES_PHRASES = Arrays.asList(
            "yes", "yeah", "yep", "correct", "that's right", "sure", "affirmative", "okay",
            "please do", "do it");
    /** What people say when they mean no. */
    public static final List<String> DEFAULT_NO_PHRASES = Arrays.asList(
            "no", "nope", "incorrect", "that's wrong", "negative", "cancel", "don't do it",
            "stop");

    /**
     * The conversation, handed to a flow as its only argument. Every method
     * speaks and then waits, so a flow is just straight-line code.
     */
    public final class Dialog {
        /** The phrase that started this flow. */
        public final String triggerPhrase;
        /** Scratch space for the flow's own use; the runner never touches it. */
        public final Map<String, Object> state = new LinkedHashMap<>();

        Dialog(String triggerPhrase) {
            this.triggerPhrase = triggerPhrase;
        }

        /** Speaks {@code text} and waits for playback to finish. */
        public void say(String text) {
            speak(text);
        }

        /** Asks an open question and returns what the user said. */
        public String ask(String prompt) {
            return ask(prompt, new AskOptions());
        }

        /** As {@link #ask(String)}, with timeout and retry behaviour tuned. */
        public String ask(String prompt, AskOptions options) {
            return promptForAnswer(prompt, options, text -> text.isEmpty() ? null : text);
        }

        /** Asks a yes/no question. */
        public boolean confirm(String prompt) {
            return confirm(prompt, DEFAULT_YES_PHRASES, DEFAULT_NO_PHRASES, new AskOptions());
        }

        /** As {@link #confirm(String)}, with the accepted phrasings supplied. */
        public boolean confirm(String prompt, List<String> yesPhrases, List<String> noPhrases,
                AskOptions options) {
            AskOptions settings = new AskOptions();
            settings.timeoutMillis = options.timeoutMillis;
            settings.maxRetries = options.maxRetries == 2 ? 1 : options.maxRetries;
            settings.reprompt = options.reprompt != null ? options.reprompt
                    : "Sorry, I didn't catch that. Was that a yes or a no? {prompt}";
            Boolean answer = promptForAnswer(prompt, settings, text -> {
                if (matchesAny(text, yesPhrases)) return Boolean.TRUE;
                if (matchesAny(text, noPhrases)) return Boolean.FALSE;
                return null;
            });
            return Boolean.TRUE.equals(answer);
        }

        /**
         * Offers a set of choices and returns the key of the one picked. Each key
         * maps to the phrases that select it; the key itself always counts.
         */
        public String choose(String prompt, Map<String, List<String>> choices) {
            return choose(prompt, choices, new AskOptions());
        }

        /** As {@link #choose(String, Map)}, with timeout and retry behaviour tuned. */
        public String choose(String prompt, Map<String, List<String>> choices,
                AskOptions options) {
            return promptForAnswer(prompt, options, text -> {
                for (Map.Entry<String, List<String>> choice : choices.entrySet()) {
                    List<String> phrases = new ArrayList<>(choice.getValue());
                    phrases.add(choice.getKey());
                    if (matchesAny(text, phrases)) {
                        return choice.getKey();
                    }
                }
                return null;
            });
        }

        /** Abandons the flow. */
        public void cancel() {
            throw new DialogCancelled();
        }

        /** Runs the flow again from the beginning. */
        public void restart() {
            throw new DialogRestart();
        }
    }

    /** Interprets an utterance, returning null when it doesn't understand. */
    private interface Interpreter<T> {
        @Nullable T interpret(String utterance);
    }

    private final Context appContext;

    private final List<String> flowOrder = new ArrayList<>();
    private final Map<String, Flow> flows = new LinkedHashMap<>();
    private final List<String> globalOrder = new ArrayList<>();
    private final Map<String, Flow> globals = new LinkedHashMap<>();

    private String languageCode = "en";
    private int arch = JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
    @Nullable private String voiceId;
    private boolean wantsMicrophone = true;
    private float threshold = DEFAULT_TRIGGER_THRESHOLD;
    @Nullable private File modelDirectory;
    @Nullable private ProgressCallback progressCallback;
    @Nullable private Consumer<String> speakOverride;
    private final CopyOnWriteArrayList<Consumer<String>> heardHandlers = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<Consumer<String>> saidHandlers = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<Consumer<Throwable>> errorHandlers =
            new CopyOnWriteArrayList<>();

    @Nullable private TextToSpeech tts;
    @Nullable private IntentRecognizer intent;
    @Nullable private MicTranscriber mic;
    private boolean ownsTts = true;
    private boolean ownsMic = true;
    private boolean triggersRegistered = false;

    private final ExecutorService flowExecutor = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "moonshine-dialog-flow");
        thread.setDaemon(true);
        return thread;
    });

    private final Object lock = new Object();
    /** Hands the next utterance to a flow parked in {@code ask}. */
    private final SynchronousQueue<String> answers = new SynchronousQueue<>();
    private volatile boolean awaitingAnswer = false;
    private volatile boolean active = false;
    @Nullable private volatile String activeTrigger;
    private volatile boolean speaking = false;
    /** Set by a global handler so the parked flow unwinds when it wakes. */
    @Nullable private volatile RuntimeException pendingInterrupt;

    public DialogFlow(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        this.appContext = context.getApplicationContext();
        // "cancel" and "start over" are what people actually say to a voice
        // interface, so they work without every application registering them.
        always("cancel", Dialog::cancel);
        always("start over", Dialog::restart);
    }

    // -- Configuration -------------------------------------------------------

    /** Speech-to-text and synthesis language. Defaults to English. */
    public DialogFlow language(String code) {
        this.languageCode = code;
        return this;
    }

    /** Overrides the streaming speech-to-text model. */
    public DialogFlow modelArch(int modelArch) {
        this.arch = modelArch;
        return this;
    }

    /** Voice used for spoken prompts, e.g. {@code "kokoro_af_heart"}. */
    public DialogFlow voice(String id) {
        this.voiceId = id;
        return this;
    }

    /** Loads every model from a directory you supply rather than the CDN. */
    public DialogFlow modelsFrom(File directory) {
        this.modelDirectory = directory;
        return this;
    }

    /** Set false to drive the dialog from text instead of a microphone. */
    public DialogFlow microphone(boolean enabled) {
        this.wantsMicrophone = enabled;
        return this;
    }

    /** Similarity a trigger phrase needs to match, 0 to 1. Defaults to 0.7. */
    public DialogFlow triggerThreshold(float value) {
        this.threshold = value;
        return this;
    }

    /** Combined download progress for every model, as a {@code 0..1} fraction. */
    public DialogFlow onProgress(ProgressCallback callback) {
        this.progressCallback = callback;
        return this;
    }

    /** Called on the main thread with each thing the user says. */
    public DialogFlow onHeard(Consumer<String> handler) {
        heardHandlers.add(handler);
        return this;
    }

    /** Called on the main thread with each thing the assistant says. */
    public DialogFlow onSaid(Consumer<String> handler) {
        saidHandlers.add(handler);
        return this;
    }

    /** Called on the main thread when a flow throws something unhandled. */
    public DialogFlow onError(Consumer<Throwable> handler) {
        errorHandlers.add(handler);
        return this;
    }

    /** Replaces the built-in synthesizer, e.g. to route prompts somewhere else. */
    public DialogFlow speakWith(Consumer<String> speak) {
        this.speakOverride = speak;
        return this;
    }

    /** Registers a flow to run when the user says something like {@code phrase}. */
    public DialogFlow listenFor(String phrase, Flow flow) {
        if (!flows.containsKey(phrase)) {
            flowOrder.add(phrase);
        }
        flows.put(phrase, flow);
        triggersRegistered = false;
        return this;
    }

    /**
     * Registers a handler that runs whenever {@code phrase} is heard, even in the
     * middle of a flow. This is how {@code cancel} and {@code start over} work.
     */
    public DialogFlow always(String phrase, Flow handler) {
        if (!globals.containsKey(phrase)) {
            globalOrder.add(phrase);
        }
        globals.put(phrase, handler);
        triggersRegistered = false;
        return this;
    }

    public DialogFlow useTextToSpeech(TextToSpeech engine) {
        this.tts = engine;
        this.ownsTts = false;
        return this;
    }

    public DialogFlow useMicTranscriber(MicTranscriber transcriber) {
        this.mic = transcriber;
        this.ownsMic = false;
        return this;
    }

    // -- Lifecycle -----------------------------------------------------------

    /**
     * Downloads and wires every model the dialog needs. Blocks; call from a
     * background thread.
     */
    public void load() {
        if (tts == null) {
            TextToSpeech synthesizer = new TextToSpeech(appContext).language(languageCode);
            if (voiceId != null) {
                synthesizer.voice(voiceId);
            }
            if (modelDirectory != null) {
                synthesizer.modelsFrom(modelDirectory);
            }
            if (progressCallback != null) {
                synthesizer.onProgress(progressCallback);
            }
            synthesizer.load();
            tts = synthesizer;
            ownsTts = true;
        }

        if (intent == null) {
            try {
                ModelSpec spec = ModelSpec.intent(null, null);
                File directory = Models.ensureOne(appContext, spec, modelDirectory,
                        progressCallback);
                intent = new IntentRecognizer(directory.getAbsolutePath(),
                        JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M, null);
            } catch (Exception e) {
                throw new RuntimeException("Failed to load the intent-matching model", e);
            }
        }

        if (wantsMicrophone && mic == null) {
            MicTranscriber transcriber = new MicTranscriber(appContext)
                    .language(languageCode)
                    .modelArch(arch);
            if (modelDirectory != null) {
                transcriber.modelsFrom(modelDirectory);
            }
            if (progressCallback != null) {
                transcriber.onProgress(progressCallback);
            }
            transcriber.load();
            mic = transcriber;
            ownsMic = true;
        }

        if (mic != null) {
            // Listen on the audio thread rather than through onLine, whose
            // main-thread delivery would put blocking flow code on the UI thread.
            mic.addListener(event -> {
                if (!(event instanceof TranscriptEvent.LineCompleted)) {
                    return;
                }
                if (speaking) {
                    // Don't let the assistant transcribe its own voice.
                    return;
                }
                TranscriptLine line = ((TranscriptEvent.LineCompleted) event).line;
                handleUtterance(line.text != null ? line.text : "");
            });
        }
    }

    /** Opens the microphone and starts responding to trigger phrases. */
    public void startListening() {
        MicTranscriber transcriber = mic;
        if (transcriber == null) {
            throw new IllegalStateException(
                    "No microphone. Call load() first, or use handleUtterance() for text input.");
        }
        transcriber.start();
    }

    public void stopListening() {
        if (mic != null) {
            mic.stop();
        }
    }

    /** Says something outside any flow, e.g. a welcome message. */
    public void say(String text) {
        speak(text);
    }

    /**
     * Feeds in an utterance the dialog didn't hear itself. Useful for text input
     * and for tests. Returns immediately; the flow advances on its own thread.
     */
    public void handleUtterance(String text) {
        final String utterance = text == null ? "" : text.trim();
        if (utterance.isEmpty()) {
            return;
        }
        for (Consumer<String> handler : heardHandlers) {
            MAIN.post(() -> handler.accept(utterance));
        }

        String trigger = matchTrigger(utterance);
        if (trigger != null && globals.containsKey(trigger)) {
            invokeGlobal(trigger);
            return;
        }
        if (active) {
            // A flow owns the conversation. Hand the line to it if it is parked
            // in ask(), and drop it otherwise rather than interleaving flows.
            // The short wait when it is not parked yet covers the gap between a
            // prompt finishing and the flow reaching the queue.
            offerAnswer(utterance, awaitingAnswer ? 1000 : 100);
            return;
        }
        if (trigger != null) {
            Flow flow = flows.get(trigger);
            if (flow != null) {
                runFlow(trigger, flow);
            }
        }
    }

    /** True while a flow is running. */
    public boolean isActive() {
        return active;
    }

    /** The trigger phrase of the running flow, if any. */
    @Nullable
    public String getActiveTrigger() {
        return activeTrigger;
    }

    /** True while a flow is parked waiting for the user to answer a question. */
    public boolean isAwaitingAnswer() {
        return awaitingAnswer;
    }

    /** Abandons the running flow. Returns false if there wasn't one. */
    public boolean cancel() {
        if (!active) {
            return false;
        }
        interrupt(new DialogCancelled());
        return true;
    }

    /**
     * Blocks until no flow is running, for tests and for shutdown. Returns false
     * if the timeout ran out first.
     */
    public boolean waitUntilIdle(long timeoutMillis) {
        long deadline = System.nanoTime() + timeoutMillis * 1_000_000L;
        while (active || !flowIdle) {
            if (System.nanoTime() >= deadline) {
                return false;
            }
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return true;
    }

    public void close() {
        flowExecutor.shutdownNow();
        if (ownsMic && mic != null) {
            mic.close();
        }
        if (ownsTts && tts != null) {
            tts.close();
        }
        if (intent != null) {
            intent.close();
        }
        mic = null;
        tts = null;
        intent = null;
    }

    // -- Internals used by Dialog --------------------------------------------

    /**
     * Speaks a prompt, waits for an answer, and re-prompts until
     * {@code interpret} accepts one or the retries run out.
     */
    private <T> T promptForAnswer(String prompt, AskOptions options, Interpreter<T> interpret) {
        String reprompt = options.reprompt != null ? options.reprompt
                : "Sorry, I didn't catch that. {prompt}";
        int attempt = 0;
        while (true) {
            speak(attempt == 0 ? prompt : reprompt.replace("{prompt}", prompt));

            String answer;
            try {
                answer = waitForAnswer(options.timeoutMillis);
            } catch (DialogNoMatch e) {
                if (attempt < options.maxRetries) {
                    attempt++;
                    continue;
                }
                throw e;
            }

            T value = interpret.interpret(answer.trim());
            if (value != null) {
                return value;
            }
            if (attempt >= options.maxRetries) {
                throw new DialogNoMatch("Gave up understanding: \"" + answer + "\"");
            }
            attempt++;
        }
    }

    private void speak(String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        throwPendingInterrupt();
        for (Consumer<String> handler : saidHandlers) {
            MAIN.post(() -> handler.accept(text));
        }
        speaking = true;
        if (mic != null) {
            mic.mute(true);
        }
        try {
            if (speakOverride != null) {
                speakOverride.accept(text);
            } else if (tts != null) {
                tts.say(text);
            } else {
                Log.i("MoonshineDialog", text);
            }
        } finally {
            if (mic != null) {
                mic.mute(false);
            }
            speaking = false;
        }
    }

    private String waitForAnswer(long timeoutMillis) {
        throwPendingInterrupt();
        awaitingAnswer = true;
        try {
            String answer = timeoutMillis > 0
                    ? answers.poll(timeoutMillis, TimeUnit.MILLISECONDS)
                    : answers.take();
            throwPendingInterrupt();
            if (answer == null) {
                throw new DialogNoMatch("Timed out waiting for an answer");
            }
            return answer;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DialogCancelled();
        } finally {
            awaitingAnswer = false;
        }
    }

    private void throwPendingInterrupt() {
        RuntimeException interruption = pendingInterrupt;
        if (interruption != null) {
            pendingInterrupt = null;
            throw interruption;
        }
    }

    // -- Internals -----------------------------------------------------------

    private volatile boolean flowIdle = true;

    private void runFlow(String triggerPhrase, Flow flow) {
        active = true;
        activeTrigger = triggerPhrase;
        flowIdle = false;
        flowExecutor.execute(() -> {
            try {
                while (true) {
                    Dialog dialog = new Dialog(triggerPhrase);
                    try {
                        flow.run(dialog);
                        return;
                    } catch (DialogRestart restart) {
                        continue;  // round again
                    } catch (DialogCancelled cancelled) {
                        return;
                    } catch (DialogNoMatch noMatch) {
                        speak("Sorry, I didn't get that. Let's start over.");
                        return;
                    } catch (Throwable error) {
                        reportError(error);
                        return;
                    }
                }
            } finally {
                pendingInterrupt = null;
                activeTrigger = null;
                active = false;
                flowIdle = true;
            }
        });
    }

    /**
     * Runs a global handler. Cancel and restart are interruptions rather than
     * failures, so they get handed to the parked flow instead of being reported.
     */
    private void invokeGlobal(String triggerPhrase) {
        Flow handler = globals.get(triggerPhrase);
        if (handler == null) {
            return;
        }
        try {
            handler.run(new Dialog(triggerPhrase));
        } catch (DialogCancelled | DialogRestart interruption) {
            if (active) {
                interrupt(interruption);
            }
        } catch (Throwable error) {
            reportError(error);
        }
    }

    /** Unwinds the running flow with {@code interruption}, waking it if parked. */
    private void interrupt(RuntimeException interruption) {
        pendingInterrupt = interruption;
        if (awaitingAnswer) {
            // Any value will do: the flow rethrows the interrupt on waking.
            offerAnswer("", 1000);
        }
    }

    private void offerAnswer(String utterance, long timeoutMillis) {
        try {
            answers.offer(utterance, timeoutMillis, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void reportError(Throwable error) {
        if (errorHandlers.isEmpty()) {
            Log.e("MoonshineDialog", "Dialog flow failed", error);
            return;
        }
        for (Consumer<Throwable> handler : errorHandlers) {
            MAIN.post(() -> handler.accept(error));
        }
    }

    @Nullable
    private String matchTrigger(String utterance) {
        List<String> phrases = new ArrayList<>(globalOrder);
        phrases.addAll(flowOrder);
        if (phrases.isEmpty()) {
            return null;
        }
        IntentRecognizer recognizer = intent;
        if (recognizer != null) {
            synchronized (lock) {
                if (!triggersRegistered) {
                    recognizer.clearIntents();
                    for (String phrase : phrases) {
                        recognizer.registerIntent(phrase);
                    }
                    triggersRegistered = true;
                }
            }
            List<IntentMatch> matches = recognizer.getClosestIntents(utterance, threshold);
            return matches.isEmpty() ? null : matches.get(0).canonicalPhrase;
        }
        String lower = utterance.toLowerCase();
        for (String phrase : phrases) {
            if (lower.contains(phrase.toLowerCase())) {
                return phrase;
            }
        }
        return null;
    }

    private static boolean matchesAny(String utterance, List<String> phrases) {
        String lower = utterance.toLowerCase();
        for (String phrase : phrases) {
            String needle = phrase.toLowerCase();
            if (lower.equals(needle) || lower.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    /** Renders a string as a space-separated spoken form for reading back. */
    public static String spellOut(String value) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            if (i > 0) {
                builder.append(' ');
            }
            builder.append(value.charAt(i));
        }
        return builder.toString();
    }
}
