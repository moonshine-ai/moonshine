package ai.moonshine.voice;

import android.content.Context;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Captures the short reference clip that zero-shot voice cloning needs.
 *
 * <pre>{@code
 * VoiceClone clone = tts.startCloning();
 * clone.onReady(() -> status.setText("Got it - you can stop talking."));
 * clone.fromMicrophone();
 * tts.cloneFrom(clone);
 * }</pre>
 *
 * <p>Finding a usable clip means locating a window of the recording that is
 * mostly speech rather than silence or breathing. That search runs in the core,
 * so the browser, iOS and Android bindings all agree on what a good clip looks
 * like. No model download is involved: the voice-activity detector is compiled
 * into the library.
 */
public final class VoiceClone {

    /** Sample rate of the clip handed back by {@link #getAudio()}. */
    public static final int CLIP_SAMPLE_RATE = SpeechClip.SAMPLE_RATE;
    /** How long {@link #fromMicrophone()} keeps listening before giving up. */
    public static final float DEFAULT_MAX_RECORD_SECONDS = 20f;
    /** How much new audio to accumulate between speech searches. */
    private static final float SEARCH_INTERVAL_SECONDS = 0.25f;

    /** Fires once, as soon as enough speech has been captured. */
    public interface ReadyCallback {
        void onReady();
    }

    /** Reports seconds recorded so far and seconds of speech in the best window. */
    public interface CloneProgressCallback {
        void onProgress(float recordedSeconds, float speechSeconds);
    }

    private final Context appContext;
    private final int ttsHandle;
    private final float clipDurationSeconds;
    private final float minimumSpeechSeconds;

    private final Object lock = new Object();
    private final List<float[]> recording = new ArrayList<>();
    private int recordedSamples = 0;
    private int recordingSampleRate = CLIP_SAMPLE_RATE;
    private int samplesSinceSearch = 0;
    @Nullable private float[] clip;
    @Nullable private String transcript;
    private float speechSeconds = 0;

    private final CopyOnWriteArrayList<ReadyCallback> readyCallbacks = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<CloneProgressCallback> progressCallbacks =
            new CopyOnWriteArrayList<>();

    private volatile boolean capturing = false;

    public VoiceClone(Context context, int ttsHandle) {
        this(context, ttsHandle, 4f, 2f);
    }

    public VoiceClone(Context context, int ttsHandle, float clipDurationSeconds,
            float minimumSpeechSeconds) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        JNI.ensureLibraryLoaded();
        this.appContext = context.getApplicationContext();
        this.ttsHandle = ttsHandle;
        this.clipDurationSeconds = clipDurationSeconds;
        this.minimumSpeechSeconds = minimumSpeechSeconds;
    }

    /** Fires once enough speech has been captured, or immediately if it already has. */
    public VoiceClone onReady(ReadyCallback callback) {
        boolean alreadyReady;
        synchronized (lock) {
            alreadyReady = clip != null;
            if (!alreadyReady) {
                readyCallbacks.add(callback);
            }
        }
        if (alreadyReady) {
            callback.onReady();
        }
        return this;
    }

    /** Reports recording and speech durations so the app can show progress. */
    public VoiceClone onProgress(CloneProgressCallback callback) {
        progressCallbacks.add(callback);
        return this;
    }

    /** True once {@link #getAudio()} holds a usable reference clip. */
    public boolean isReady() {
        synchronized (lock) {
            return clip != null;
        }
    }

    /** The captured clip (16 kHz mono), or null until {@link #isReady()}. */
    @Nullable
    public float[] getAudio() {
        synchronized (lock) {
            return clip;
        }
    }

    public int getSampleRate() {
        return CLIP_SAMPLE_RATE;
    }

    /** Speech found in the best window so far, in seconds. */
    public float getSpeechSeconds() {
        synchronized (lock) {
            return speechSeconds;
        }
    }

    /** Unused for VAD capture; cloneFrom fills the transcript via create-time ASR. */
    @Nullable
    public String getTranscript() {
        synchronized (lock) {
            return transcript;
        }
    }

    public float getRecordedSeconds() {
        synchronized (lock) {
            return recordingSampleRate > 0 ? (float) recordedSamples / recordingSampleRate : 0f;
        }
    }

    /**
     * Feeds captured audio in. Call this from your own audio pipeline; the
     * search for a usable window runs a few times a second rather than on every
     * chunk.
     */
    public void addAudio(float[] pcm, int sampleRate) {
        boolean due;
        synchronized (lock) {
            if (clip != null || pcm == null || pcm.length == 0 || sampleRate <= 0) {
                return;
            }
            if (sampleRate != recordingSampleRate) {
                // Mixed rates in one buffer would make the clip come out at the
                // wrong speed, so a change starts the recording over.
                recording.clear();
                recordedSamples = 0;
                recordingSampleRate = sampleRate;
                samplesSinceSearch = 0;
            }
            recording.add(pcm);
            recordedSamples += pcm.length;
            samplesSinceSearch += pcm.length;
            due = samplesSinceSearch >= SEARCH_INTERVAL_SECONDS * sampleRate;
            if (due) {
                samplesSinceSearch = 0;
            }
        }
        if (due) {
            search(false);
        }
    }

    /** Records until there is enough speech, or {@link #DEFAULT_MAX_RECORD_SECONDS} elapse. */
    public float[] fromMicrophone() {
        return fromMicrophone(DEFAULT_MAX_RECORD_SECONDS);
    }

    /**
     * Opens the microphone and records until there is enough speech, or until
     * {@code maxSeconds} have passed. Blocks; call from a background thread.
     *
     * @return the clip, which is also available from {@link #getAudio()}.
     */
    public float[] fromMicrophone(float maxSeconds) {
        MicrophonePermission.ensureGranted(appContext);
        MicCaptureProcessor capture = new MicCaptureProcessor();
        Thread thread = new Thread(capture, "moonshine-clone-capture");
        thread.setDaemon(true);
        capturing = true;
        thread.start();
        try {
            long deadline = System.nanoTime() + (long) (maxSeconds * 1_000_000_000L);
            while (capturing && !isReady()) {
                if (System.nanoTime() >= deadline) {
                    // Out of patience: take the best window we have, even a quiet one.
                    search(true);
                    break;
                }
                addAudio(capture.consumeAudio(), CLIP_SAMPLE_RATE);
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        } finally {
            capturing = false;
            thread.interrupt();
        }
        float[] audio = getAudio();
        if (audio == null) {
            throw new IllegalStateException("No speech detected in " + (int) maxSeconds
                    + "s of recording. Try again somewhere quieter.");
        }
        return audio;
    }

    /** Stops an in-flight {@link #fromMicrophone(float)} capture. */
    public void cancel() {
        capturing = false;
    }

    /** Throws away everything captured so far. */
    public void reset() {
        synchronized (lock) {
            recording.clear();
            recordedSamples = 0;
            samplesSinceSearch = 0;
            clip = null;
            transcript = null;
            speechSeconds = 0;
        }
    }

    // -- Internals -----------------------------------------------------------

    private void search(boolean acceptAnything) {
        float[] samples;
        int rate;
        synchronized (lock) {
            if (clip != null) {
                return;
            }
            samples = flattenLocked();
            rate = recordingSampleRate;
        }
        if (samples.length == 0) {
            return;
        }
        SpeechClip found = JNI.moonshineExtractSpeechClip(samples, rate, ttsHandle,
                clipDurationSeconds, acceptAnything ? 0f : minimumSpeechSeconds);
        if (found == null) {
            return;
        }

        float recorded = (float) samples.length / rate;
        List<ReadyCallback> ready = null;
        synchronized (lock) {
            speechSeconds = found.speechDuration;
            if (found.audio != null && found.audio.length > 0) {
                clip = found.audio;
                transcript = found.transcript;
                ready = new ArrayList<>(readyCallbacks);
                readyCallbacks.clear();
            }
        }
        for (CloneProgressCallback callback : progressCallbacks) {
            callback.onProgress(recorded, found.speechDuration);
        }
        if (ready != null) {
            for (ReadyCallback callback : ready) {
                callback.onReady();
            }
        }
    }

    private float[] flattenLocked() {
        float[] all = new float[recordedSamples];
        int offset = 0;
        for (float[] chunk : recording) {
            System.arraycopy(chunk, 0, all, offset, chunk.length);
            offset += chunk.length;
        }
        // Coalesce so the next search does not walk a long list of chunks.
        recording.clear();
        if (all.length > 0) {
            recording.add(all);
        }
        return all;
    }
}
