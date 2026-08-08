package ai.moonshine.voice;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.Nullable;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Live speech-to-text from the device microphone.
 *
 * <pre>{@code
 * MicTranscriber mic = new MicTranscriber(this)
 *         .onText(text -> binding.liveTranscript.setText(text))
 *         .onLine(line -> appendFinal(line.text));
 *
 * executor.execute(() -> {
 *     mic.load();
 *     mic.start();
 * });
 * }</pre>
 *
 * <p>Construction is cheap and synchronous, chained setters configure, and
 * {@link #load()} is the one slow call: it downloads the speech-to-text model
 * if it is not cached yet, then loads it. {@link #start()} asks for the
 * microphone permission if the user has not granted it, so an app does not have
 * to declare, request, and hand back the grant itself.
 *
 * <p>{@link #load()} and {@link #start()} block, so call them off the main
 * thread. Callbacks come back <i>on</i> the main thread, which is where an app
 * wants them, so no {@code runOnUiThread} wrapper is needed.
 */
public class MicTranscriber extends Transcriber {

    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static final int SAMPLE_RATE = 16000;

    private final Context appContext;

    private String languageCode = "en";
    private int arch = JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
    private boolean includeSpelling = false;
    @Nullable private File modelDirectory;
    @Nullable private ProgressCallback progressCallback;
    private boolean deliverOnMainThread = true;

    private final List<Consumer<String>> textHandlers = new CopyOnWriteArrayList<>();
    private final List<Consumer<TranscriptLine>> lineHandlers = new CopyOnWriteArrayList<>();
    private final List<Consumer<Throwable>> errorHandlers = new CopyOnWriteArrayList<>();

    private volatile boolean running = false;
    private volatile boolean muted = false;
    @Nullable private MicCaptureProcessor micCaptureProcessor;
    @Nullable private Thread micThread;
    @Nullable private Thread processingThread;
    private final Object captureLock = new Object();

    public MicTranscriber(Context context) {
        super();
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        this.appContext = context.getApplicationContext();
        addListener(this::dispatch);
    }

    // -- Configuration -------------------------------------------------------

    /** Language to transcribe, as a code like {@code "en"}. Defaults to English. */
    public MicTranscriber language(String code) {
        this.languageCode = code;
        return this;
    }

    /**
     * Picks a different speech-to-text model, e.g.
     * {@link JNI#MOONSHINE_MODEL_ARCH_TINY_STREAMING} for a smaller download on
     * slower hardware. Defaults to the medium streaming model.
     */
    public MicTranscriber modelArch(int modelArch) {
        this.arch = modelArch;
        return this;
    }

    /**
     * Also downloads the spelling model and turns on spelling fusion, which
     * makes letter-by-letter dictation ("W I F I") come out right.
     */
    public MicTranscriber spelling(boolean enabled) {
        this.includeSpelling = enabled;
        setTranscribeFlags(enabled ? JNI.MOONSHINE_FLAG_SPELLING_MODE : 0);
        return this;
    }

    /** Loads models from {@code directory} rather than downloading them. */
    public MicTranscriber modelsFrom(File directory) {
        this.modelDirectory = directory;
        return this;
    }

    /** Called as the model downloads, with a {@code 0..1} fraction. */
    public MicTranscriber onProgress(ProgressCallback callback) {
        this.progressCallback = callback;
        return this;
    }

    /**
     * Called with the text of the line currently being spoken, which is revised
     * as more audio arrives. Use it for the live, in-progress display.
     */
    public MicTranscriber onText(Consumer<String> handler) {
        textHandlers.add(handler);
        return this;
    }

    /** Called once per finished line, when the speaker pauses. */
    public MicTranscriber onLine(Consumer<TranscriptLine> handler) {
        lineHandlers.add(handler);
        return this;
    }

    /** Called when capture or transcription fails after {@link #start()}. */
    public MicTranscriber onError(Consumer<Throwable> handler) {
        errorHandlers.add(handler);
        return this;
    }

    /**
     * Set false to receive callbacks on the audio thread instead of the main
     * thread. Only worth doing when a callback feeds something that is not UI
     * and the extra hop matters.
     */
    public MicTranscriber callbacksOnMainThread(boolean enabled) {
        this.deliverOnMainThread = enabled;
        return this;
    }

    // -- Lifecycle -----------------------------------------------------------

    /**
     * Downloads the model if needed and loads it. Blocks; call from a
     * background thread.
     */
    public void load() {
        if (isLoaded()) {
            return;
        }
        try {
            ModelSpec spec = ModelSpec.stt(languageCode, arch, includeSpelling);
            File directory = Models.ensureOne(appContext, spec, modelDirectory, progressCallback);
            loadFromFiles(directory.getAbsolutePath(), arch);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load the speech-to-text model", e);
        }
    }

    /**
     * Opens the microphone and starts transcribing, prompting for the recording
     * permission if it has not been granted. Blocks while the user answers, so
     * call from a background thread.
     */
    @Override
    public void start() {
        if (!isLoaded()) {
            throw new IllegalStateException("Call load() before start().");
        }
        MicrophonePermission.ensureGranted(appContext);
        synchronized (captureLock) {
            if (micThread == null) {
                micCaptureProcessor = new MicCaptureProcessor();
                micThread = new Thread(micCaptureProcessor, "moonshine-mic-capture");
                micThread.setDaemon(true);
                micThread.start();
            }
            if (processingThread == null) {
                processingThread = new Thread(this::audioProcessingLoop, "moonshine-mic-transcribe");
                processingThread.setDaemon(true);
                processingThread.start();
            }
        }
        running = true;
    }

    /** Stops transcribing and flushes the trailing line. */
    @Override
    public void stop() {
        running = false;
    }

    /**
     * Drops incoming audio while muted, without tearing down the microphone.
     * {@link AgentFlow} uses this so the assistant does not transcribe itself.
     */
    public void mute(boolean muted) {
        this.muted = muted;
    }

    public boolean isRunning() {
        return running;
    }

    /** Stops capture and releases the model. */
    @Override
    public void close() {
        running = false;
        Thread mic;
        Thread processing;
        synchronized (captureLock) {
            mic = micThread;
            processing = processingThread;
            micThread = null;
            processingThread = null;
            micCaptureProcessor = null;
        }
        if (mic != null) {
            mic.interrupt();
        }
        if (processing != null) {
            processing.interrupt();
            try {
                processing.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        super.close();
    }

    // -- Internals -----------------------------------------------------------

    private void dispatch(TranscriptEvent event) {
        if (event instanceof TranscriptEvent.LineTextChanged) {
            TranscriptLine line = ((TranscriptEvent.LineTextChanged) event).line;
            String text = line.text != null ? line.text : "";
            for (Consumer<String> handler : textHandlers) {
                deliver(() -> handler.accept(text));
            }
        } else if (event instanceof TranscriptEvent.LineCompleted) {
            TranscriptLine line = ((TranscriptEvent.LineCompleted) event).line;
            for (Consumer<TranscriptLine> handler : lineHandlers) {
                deliver(() -> handler.accept(line));
            }
        }
    }

    private void reportError(Throwable error) {
        if (errorHandlers.isEmpty()) {
            Log.e("MoonshineMic", "Microphone transcription failed", error);
            return;
        }
        for (Consumer<Throwable> handler : errorHandlers) {
            deliver(() -> handler.accept(error));
        }
    }

    private void deliver(Runnable action) {
        if (deliverOnMainThread && Looper.myLooper() != Looper.getMainLooper()) {
            MAIN.post(action);
        } else {
            action.run();
        }
    }

    private void audioProcessingLoop() {
        int streamHandle = createStream();
        boolean streaming = false;
        try {
            while (!Thread.currentThread().isInterrupted()) {
                MicCaptureProcessor capture;
                synchronized (captureLock) {
                    capture = micCaptureProcessor;
                }
                if (capture == null) {
                    break;
                }
                float[] audio = capture.consumeAudio();
                boolean wantStreaming = running && !muted;
                if (wantStreaming && !streaming) {
                    startStream(streamHandle);
                    streaming = true;
                }
                if (streaming && audio.length > 0) {
                    addAudioToStream(streamHandle, audio, SAMPLE_RATE);
                }
                if (!wantStreaming && streaming) {
                    stopStream(streamHandle);
                    streaming = false;
                }
                if (audio.length == 0) {
                    // Nothing captured yet; idle briefly rather than spinning.
                    Thread.sleep(10);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Throwable error) {
            reportError(error);
        } finally {
            try {
                if (streaming) {
                    stopStream(streamHandle);
                }
                freeStream(streamHandle);
            } catch (Throwable ignored) {
                // The transcriber may already have been freed by close().
            }
        }
    }
}
