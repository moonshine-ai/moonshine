package ai.moonshine.voice;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * On-device text-to-speech via the Moonshine native API (Kokoro / Piper under a {@code g2p_root} tree).
 *
 * <p>This mirrors the Python {@code moonshine_voice.TextToSpeech} surface for create / {@link #synthesize}
 * / {@link #close}, without any automatic asset download. Populate {@code g2p_root} on disk (or use
 * {@link #fromMemory}) before calling.
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

    /**
     * Synthesizes {@code text} and plays mono float PCM on the default audio output, blocking until
     * playback finishes.
     *
     * <p>Uses {@link Context#getApplicationContext()} internally. Call from a background thread if
     * utterances are long to avoid janking the UI.
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
     * Synthesizes with optional native options, then plays through {@code outputDevice} or the default
     * route. Reuses a single {@link AudioTrack} when {@code outputDevice} and sample rate match the
     * previous call so routing and buffer sizing are not recomputed each time.
     */
    public void say(Context context, String text, @Nullable AudioDeviceInfo outputDevice,
            @Nullable List<TranscriberOption> synthesizeOptions) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        Context appContext = context.getApplicationContext();
        TtsSynthesisResult result = synthesize(text, synthesizeOptions);
        float[] samples = result.samples != null ? result.samples : new float[0];
        int sampleRate = result.sampleRateHz;
        if (sampleRate <= 0) {
            throw new RuntimeException("Invalid TTS sample rate: " + sampleRate);
        }
        int wantDeviceId = outputDevice != null ? outputDevice.getId() : -1;
        synchronized (sayLock) {
            AudioTrack track =
                    obtainSayTrackLocked(appContext, wantDeviceId, outputDevice, sampleRate);
            playPcmFloatBlocking(track, samples);
        }
    }

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
        AudioTrack track = new AudioTrack.Builder()
                .setContext(appContext)
                .setAudioAttributes(attrs)
                .setAudioFormat(format)
                .setBufferSizeInBytes(minBufBytes)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build();
        if (outputDevice != null) {
            track.setPreferredDevice(outputDevice);
        }
        sayCachedTrack = track;
        sayCachedDeviceId = wantDeviceId;
        sayCachedSampleRateHz = sampleRateHz;
        return track;
    }

    private static void playPcmFloatBlocking(AudioTrack track, float[] samples) {
        if (track.getState() != AudioTrack.STATE_INITIALIZED) {
            throw new RuntimeException("AudioTrack is not initialized");
        }
        track.stop();
        track.flush();
        if (samples.length == 0) {
            return;
        }
        track.play();
        int offset = 0;
        while (offset < samples.length) {
            int wrote = track.write(samples, offset, samples.length - offset, AudioTrack.WRITE_BLOCKING);
            if (wrote <= 0) {
                track.stop();
                throw new RuntimeException("AudioTrack.write failed: " + wrote);
            }
            offset += wrote;
        }
        final int totalFrames = samples.length;
        final long deadline = System.nanoTime() + 60_000_000_000L;
        while (System.nanoTime() < deadline) {
            int head = track.getPlaybackHeadPosition();
            if (head >= totalFrames - 1) {
                break;
            }
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                track.stop();
                throw new RuntimeException(e);
            }
        }
        track.stop();
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
