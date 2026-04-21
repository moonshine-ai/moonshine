package ai.moonshine.voice;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * On-device text-to-speech via the Moonshine native API (Kokoro / Piper under a {@code g2p_root} tree).
 *
 * <p>This mirrors the Python {@code moonshine_voice.TextToSpeech} surface for create / {@link #synthesize}
 * / {@link #close}, without any automatic asset download. Populate {@code g2p_root} on disk (or use
 * {@link #fromMemory}) before calling.
 *
 * <p>{@link #say} queues text for background synthesis and playback and returns immediately.
 * Synthesis of the next utterance is pipelined with playback of the current one. Use {@link #stop()}
 * to cancel pending utterances and halt playback, {@link #waitUntilDone()} to block until all
 * queued utterances finish, and {@link #isTalking()} to poll the current state.
 */
public class TextToSpeech {
    private int handle = -1;
    private final String language;

    private final Object sayLock = new Object();
    /** {@code -1} means default output route (no {@link AudioTrack#setPreferredDevice}). */
    private int sayCachedDeviceId = Integer.MIN_VALUE;
    private int sayCachedSampleRateHz;
    @Nullable
    private AudioTrack sayCachedTrack;

    // -- Queue infrastructure ------------------------------------------------

    private static final Object SHUTDOWN_SENTINEL = new Object();

    private static class SayRequest {
        final String text;
        final Context appContext;
        @Nullable final AudioDeviceInfo outputDevice;
        @Nullable final List<TranscriberOption> options;

        SayRequest(String text, Context appContext, @Nullable AudioDeviceInfo outputDevice,
                   @Nullable List<TranscriberOption> options) {
            this.text = text;
            this.appContext = appContext;
            this.outputDevice = outputDevice;
            this.options = options;
        }
    }

    private static class PlayItem {
        final float[] samples;
        final int sampleRate;
        final Context appContext;
        @Nullable final AudioDeviceInfo outputDevice;

        PlayItem(float[] samples, int sampleRate, Context appContext,
                 @Nullable AudioDeviceInfo outputDevice) {
            this.samples = samples;
            this.sampleRate = sampleRate;
            this.appContext = appContext;
            this.outputDevice = outputDevice;
        }
    }

    @SuppressWarnings("rawtypes")
    private final LinkedBlockingQueue sayQueue = new LinkedBlockingQueue();
    @SuppressWarnings("rawtypes")
    private final ArrayBlockingQueue playQueue = new ArrayBlockingQueue(1);
    private volatile boolean stopRequested = false;
    @Nullable private Thread synthThread;
    @Nullable private Thread playThread;
    private final Object workerLock = new Object();

    private final AtomicInteger pendingCount = new AtomicInteger(0);
    private final Object pendingLock = new Object();

    /**
     * Load TTS from files on disk (optional {@code filenames} keys; same semantics as the C API).
     *
     * @param language Moonshine language tag (e.g. {@code en_us}).
     * @param filenames  Canonical asset keys, or {@code null} / empty to resolve from {@code g2p_root} only.
     * @param g2pRoot    Directory containing G2P and vocoder assets; stored as option {@code g2p_root}.
     * @param options    Additional {@link TranscriberOption} entries (e.g. {@code voice}).
     */
    public TextToSpeech(String language, String[] filenames, String g2pRoot,
            List<TranscriberOption> options) {
        this.language = language;
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateTtsSynthesizerFromFiles(language, filenames,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        this.handle = h;
    }

    /**
     * Same as {@link #TextToSpeech(String, String[], String, List)} with no explicit filename keys.
     */
    public TextToSpeech(String language, String g2pRoot, List<TranscriberOption> options) {
        this(language, null, g2pRoot, options);
    }

    /**
     * Create from in-memory bytes per canonical key; {@code memory[i]} may be {@code null} or empty to load
     * that key from disk under {@code g2p_root}.
     */
    public static TextToSpeech fromMemory(String language, String[] filenames, byte[][] memory,
            String g2pRoot, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateTtsSynthesizerFromMemory(language, filenames, memory,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        return new TextToSpeech(language, h);
    }

    private TextToSpeech(String language, int handle) {
        this.language = language;
        this.handle = handle;
    }

    public String getLanguage() {
        return language;
    }

    /** Comma-separated G2P asset keys (see {@code moonshine_get_g2p_dependencies}). */
    public static String getG2pDependencies(String languages, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        String json = JNI.moonshineGetG2pDependencies(languages, toArray(options));
        if (json == null) {
            throw new RuntimeException("moonshineGetG2pDependencies failed");
        }
        return json;
    }

    /** JSON array of merged G2P + vocoder keys (see {@code moonshine_get_tts_dependencies}). */
    public static String getTtsDependencies(String languages, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        String json = JNI.moonshineGetTtsDependencies(languages, toArray(options));
        if (json == null) {
            throw new RuntimeException("moonshineGetTtsDependencies failed");
        }
        return json;
    }

    /** JSON object of voice availability (see {@code moonshine_get_tts_voices}). */
    public static String getTtsVoices(String languages, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        String json = JNI.moonshineGetTtsVoices(languages, toArray(options));
        if (json == null) {
            throw new RuntimeException("moonshineGetTtsVoices failed");
        }
        return json;
    }

    private static TranscriberOption[] toArray(List<TranscriberOption> options) {
        if (options == null || options.isEmpty()) {
            return null;
        }
        return options.toArray(new TranscriberOption[0]);
    }

    /**
     * Synthesize text to mono float PCM (approximately -1..1) and sample rate in Hz.
     * Optional per-call options are forwarded to the native layer (currently unused by the engine).
     */
    public TtsSynthesisResult synthesize(String text, List<TranscriberOption> options) {
        TtsSynthesisResult r = JNI.moonshineTextToSpeech(handle, text, toArray(options));
        if (r == null) {
            throw new RuntimeException("moonshineTextToSpeech failed");
        }
        return r;
    }

    public TtsSynthesisResult synthesize(String text) {
        return synthesize(text, null);
    }

    // -- Queued say / stop / wait / isTalking --------------------------------

    /**
     * Queue {@code text} for synthesis and playback, returning immediately.
     *
     * <p>Utterances are played in order. Synthesis of the next utterance is pipelined with playback
     * of the current one so there is minimal gap between consecutive utterances. Call {@link #stop()}
     * to cancel all pending utterances and halt the currently-playing audio.
     */
    public void say(Context context, String text) {
        say(context, text, null, null);
    }

    /**
     * Same as {@link #say(Context, String)} but routes output to {@code outputDevice} when non-null
     * (see {@link AudioTrack#setPreferredDevice}).
     */
    public void say(Context context, String text, @Nullable AudioDeviceInfo outputDevice) {
        say(context, text, outputDevice, null);
    }

    /**
     * Queue {@code text} for synthesis and playback with options, returning immediately.
     */
    @SuppressWarnings("unchecked")
    public void say(Context context, String text, @Nullable AudioDeviceInfo outputDevice,
            @Nullable List<TranscriberOption> synthesizeOptions) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        Context appContext = context.getApplicationContext();
        pendingCount.incrementAndGet();
        sayQueue.add(new SayRequest(text, appContext, outputDevice, synthesizeOptions));
        ensureWorkers();
    }

    /**
     * Queue each string for synthesis and playback, returning immediately.
     *
     * <p>Equivalent to calling {@link #say(Context, String)} once per element in order.
     */
    public void say(Context context, String[] texts) {
        say(context, texts, null, null);
    }

    /**
     * Queue each string for synthesis and playback with options, returning immediately.
     */
    public void say(Context context, String[] texts, @Nullable AudioDeviceInfo outputDevice,
            @Nullable List<TranscriberOption> synthesizeOptions) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        if (texts == null) return;
        Context appContext = context.getApplicationContext();
        for (String text : texts) {
            pendingCount.incrementAndGet();
            //noinspection unchecked
            sayQueue.add(new SayRequest(text, appContext, outputDevice, synthesizeOptions));
        }
        ensureWorkers();
    }

    /**
     * Block until all queued utterances have been synthesized and played.
     */
    public void waitUntilDone() {
        synchronized (pendingLock) {
            while (pendingCount.get() > 0) {
                try {
                    pendingLock.wait(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    /**
     * Clear the utterance queue and stop any audio currently playing.
     *
     * <p>Returns once all pending utterances are discarded and the active playback (if any)
     * has been halted. It is safe to call {@link #say} again afterwards.
     */
    public void stop() {
        stopRequested = true;

        drainQueue(sayQueue);
        drainQueue(playQueue);

        synchronized (sayLock) {
            if (sayCachedTrack != null) {
                try {
                    sayCachedTrack.stop();
                    sayCachedTrack.flush();
                } catch (Exception ignored) {
                }
            }
        }

        joinWorkers();
    }

    /**
     * Returns {@code true} if utterances are queued, being synthesized, or currently playing.
     */
    public boolean isTalking() {
        return pendingCount.get() > 0;
    }

    // -- Worker threads ------------------------------------------------------

    private void ensureWorkers() {
        synchronized (workerLock) {
            boolean alive = synthThread != null && synthThread.isAlive()
                    && playThread != null && playThread.isAlive();
            if (alive) return;

            stopRequested = false;

            synthThread = new Thread(this::synthWorker, "moonshine-tts-synth");
            synthThread.setDaemon(true);
            synthThread.start();

            playThread = new Thread(this::playWorker, "moonshine-tts-play");
            playThread.setDaemon(true);
            playThread.start();
        }
    }

    private void synthWorker() {
        while (!stopRequested) {
            Object raw;
            try {
                raw = sayQueue.poll(100, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }
            if (raw == null) continue;
            if (raw == SHUTDOWN_SENTINEL) break;
            if (stopRequested) {
                decrementPending();
                break;
            }

            SayRequest req = (SayRequest) raw;
            try {
                TtsSynthesisResult result = synthesize(req.text, req.options);
                float[] samples = result.samples != null ? result.samples : new float[0];
                int sampleRate = result.sampleRateHz;
                if (sampleRate <= 0 || samples.length == 0) {
                    decrementPending();
                    continue;
                }
                if (stopRequested) {
                    decrementPending();
                    break;
                }
                PlayItem item = new PlayItem(samples, sampleRate, req.appContext, req.outputDevice);
                //noinspection unchecked
                while (!stopRequested) {
                    if (playQueue.offer(item, 100, TimeUnit.MILLISECONDS)) break;
                }
                if (stopRequested) {
                    decrementPending();
                    break;
                }
            } catch (Exception e) {
                decrementPending();
            }
        }
    }

    private void playWorker() {
        while (!stopRequested) {
            Object raw;
            try {
                raw = playQueue.poll(100, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }
            if (raw == null) continue;
            if (raw == SHUTDOWN_SENTINEL) break;
            if (stopRequested) {
                decrementPending();
                break;
            }

            PlayItem item = (PlayItem) raw;
            try {
                playOneItem(item);
            } catch (Exception ignored) {
            } finally {
                decrementPending();
            }
        }
    }

    private void playOneItem(PlayItem item) {
        int wantDeviceId = item.outputDevice != null ? item.outputDevice.getId() : -1;
        synchronized (sayLock) {
            if (stopRequested) return;
            AudioTrack track = obtainSayTrackLocked(item.appContext, wantDeviceId,
                    item.outputDevice, item.sampleRate);
            playPcmFloat(track, item.samples);
        }
    }

    private void playPcmFloat(AudioTrack track, float[] samples) {
        if (track.getState() != AudioTrack.STATE_INITIALIZED) {
            throw new RuntimeException("AudioTrack is not initialized");
        }
        track.stop();
        track.flush();
        if (samples.length == 0) return;

        track.play();
        int offset = 0;
        while (offset < samples.length && !stopRequested) {
            int wrote = track.write(samples, offset, samples.length - offset,
                    AudioTrack.WRITE_BLOCKING);
            if (wrote <= 0) {
                track.stop();
                throw new RuntimeException("AudioTrack.write failed: " + wrote);
            }
            offset += wrote;
        }
        if (stopRequested) {
            track.stop();
            return;
        }
        final int totalFrames = samples.length;
        final long deadline = System.nanoTime() + 60_000_000_000L;
        while (System.nanoTime() < deadline && !stopRequested) {
            int head = track.getPlaybackHeadPosition();
            if (head >= totalFrames - 1) break;
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                track.stop();
                return;
            }
        }
        track.stop();
    }

    private void decrementPending() {
        if (pendingCount.decrementAndGet() <= 0) {
            pendingCount.set(0);
            synchronized (pendingLock) {
                pendingLock.notifyAll();
            }
        }
    }

    @SuppressWarnings("rawtypes")
    private static void drainQueue(java.util.concurrent.BlockingQueue queue) {
        while (queue.poll() != null) { /* discard */ }
    }

    private void joinWorkers() {
        Thread st, pt;
        synchronized (workerLock) {
            st = synthThread;
            pt = playThread;
        }
        try {
            if (st != null && st.isAlive()) st.join(2000);
        } catch (InterruptedException ignored) {
        }
        try {
            if (pt != null && pt.isAlive()) pt.join(2000);
        } catch (InterruptedException ignored) {
        }
        synchronized (workerLock) {
            synthThread = null;
            playThread = null;
        }
    }

    // -- Audio track management ----------------------------------------------

    /**
     * Lists output devices suitable for {@link #say(Context, String, AudioDeviceInfo)} (e.g. speaker,
     * wired headset, USB audio).
     */
    public static AudioDeviceInfo[] getAudioOutputDevices(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        AudioManager am = (AudioManager) context.getApplicationContext()
                .getSystemService(Context.AUDIO_SERVICE);
        if (am == null) {
            return new AudioDeviceInfo[0];
        }
        return am.getDevices(AudioManager.GET_DEVICES_OUTPUTS);
    }

    private AudioTrack obtainSayTrackLocked(Context appContext, int wantDeviceId,
            @Nullable AudioDeviceInfo outputDevice, int sampleRateHz) {
        if (sayCachedTrack != null
                && wantDeviceId == sayCachedDeviceId
                && sampleRateHz == sayCachedSampleRateHz) {
            return sayCachedTrack;
        }
        releaseSayTrackLocked();
        AudioAttributes attrs = new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build();
        AudioFormat format = new AudioFormat.Builder()
                .setSampleRate(sampleRateHz)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build();
        int minBufBytes = AudioTrack.getMinBufferSize(
                sampleRateHz,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT);
        if (minBufBytes <= 0) {
            throw new RuntimeException("AudioTrack.getMinBufferSize failed for sampleRate=" + sampleRateHz);
        }
        AudioTrack track = buildAudioTrack(appContext, attrs, format, minBufBytes);
        if (outputDevice != null) {
            track.setPreferredDevice(outputDevice);
        }
        sayCachedTrack = track;
        sayCachedDeviceId = wantDeviceId;
        sayCachedSampleRateHz = sampleRateHz;
        return track;
    }

    /**
     * Builds an {@link AudioTrack}, first trying with {@link AudioTrack.Builder#setContext(Context)}
     * and falling back to a builder without a context on failure.
     *
     * <p>Some Android 15 (API 35) configurations (notably the emulator) fail
     * {@code AudioFlinger::createTrack} with {@code NPC::validateUidPackagePair: uid not found}
     * when a Context is supplied, leaving the returned track uninitialized. Retrying without
     * {@code setContext} sidesteps that UID validation path.
     */
    private static AudioTrack buildAudioTrack(Context appContext, AudioAttributes attrs,
            AudioFormat format, int minBufBytes) {
        try {
            AudioTrack track = new AudioTrack.Builder()
                    .setContext(appContext)
                    .setAudioAttributes(attrs)
                    .setAudioFormat(format)
                    .setBufferSizeInBytes(minBufBytes)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();
            if (track.getState() == AudioTrack.STATE_INITIALIZED) {
                return track;
            }
            Log.w("MoonshineTTS",
                    "AudioTrack.Builder(setContext) produced uninitialized track"
                            + " (state=" + track.getState() + "); retrying without setContext");
            try {
                track.release();
            } catch (Exception ignored) {
            }
        } catch (Exception e) {
            Log.w("MoonshineTTS",
                    "AudioTrack.Builder(setContext) threw; retrying without setContext: "
                            + e.getMessage());
        }
        AudioTrack fallback = new AudioTrack.Builder()
                .setAudioAttributes(attrs)
                .setAudioFormat(format)
                .setBufferSizeInBytes(minBufBytes)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build();
        if (fallback.getState() != AudioTrack.STATE_INITIALIZED) {
            int state = fallback.getState();
            try {
                fallback.release();
            } catch (Exception ignored) {
            }
            throw new RuntimeException(
                    "AudioTrack failed to initialize (state=" + state + ")");
        }
        return fallback;
    }

    private void releaseSayTrackLocked() {
        if (sayCachedTrack != null) {
            try {
                sayCachedTrack.stop();
            } catch (Exception ignored) {
            }
            sayCachedTrack.release();
            sayCachedTrack = null;
        }
        sayCachedDeviceId = Integer.MIN_VALUE;
        sayCachedSampleRateHz = 0;
    }

    public void close() {
        stopRequested = true;
        drainQueue(sayQueue);
        drainQueue(playQueue);
        joinWorkers();

        synchronized (sayLock) {
            releaseSayTrackLocked();
        }
        if (handle >= 0) {
            JNI.moonshineFreeTtsSynthesizer(handle);
            handle = -1;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
}
