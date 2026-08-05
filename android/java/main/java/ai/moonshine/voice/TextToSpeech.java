package ai.moonshine.voice;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * On-device text-to-speech.
 *
 * <pre>{@code
 * TextToSpeech tts = new TextToSpeech(context);
 * tts.load();
 * tts.say("Hello world!");
 * }</pre>
 *
 * <p>Cloning a voice is one more line, and the awkward parts — finding the
 * speech in the reference recording, and transcribing it so the vocoder knows
 * what was said — happen inside the library:
 *
 * <pre>{@code
 * tts.cloneFrom("some-speech.wav");
 * tts.say("Hello in your voice!");
 * }</pre>
 *
 * <p>{@link #say} plays audio and returns when playback finishes;
 * {@link #synthesize} returns the raw PCM instead, for callers doing their own
 * mixing or encoding. Both block, as do {@link #load()} and {@link #cloneFrom},
 * so call them off the main thread.
 */
public class TextToSpeech {

    /** Canonical asset key under which a ZipVoice clone reference clip is supplied. */
    private static final String CLONE_AUDIO_KEY = "zipvoice/clone_audio";
    /** Engine name used when creating ZipVoice from a captured clone clip. */
    private static final String CLONE_VOICE = "zipvoice";
    /** Built-in ZipVoice voice used by {@link #cloning(boolean)} before a clip exists. */
    private static final String CLONE_PRESET_VOICE = "zipvoice_american_female";
    /** Reference clips are resampled to this rate before cloning. */
    private static final int CLONE_SAMPLE_RATE = 16000;

    private final Context appContext;
    private int handle = -1;

    // Deferred configuration, applied by load().
    private String languageTag = "en";
    @Nullable private String voiceId;
    @Nullable private File assetDirectory;
    private final List<TranscriberOption> extraOptions = new ArrayList<>();
    @Nullable private ProgressCallback progressCallback;
    private boolean cloningWanted = false;
    @Nullable private AudioDeviceInfo outputDevice;

    /** The clip the current voice was cloned from, if any. */
    @Nullable private float[] cloneSamples;
    @Nullable private String cloneTranscript;

    private final Object sayLock = new Object();
    /** {@code Integer.MIN_VALUE} means no track has been built yet. */
    private int sayCachedDeviceId = Integer.MIN_VALUE;
    private int sayCachedSampleRateHz;
    @Nullable private AudioTrack sayCachedTrack;

    /**
     * Creates a synthesizer that has not loaded any assets yet. Configure it
     * with the chainable setters, then call {@link #load()}.
     */
    public TextToSpeech(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("context is required");
        }
        JNI.ensureLibraryLoaded();
        this.appContext = context.getApplicationContext();
    }

    // -- Configuration -------------------------------------------------------

    /** Synthesis language, e.g. {@code "en"} or {@code "en_us"}. Defaults to English. */
    public TextToSpeech language(String code) {
        this.languageTag = code;
        return this;
    }

    /** Catalog voice id, e.g. {@code "kokoro_af_heart"}. Clears {@link #cloning(boolean)}. */
    public TextToSpeech voice(String id) {
        this.voiceId = id;
        this.cloningWanted = false;
        return this;
    }

    /** Loads voice assets from a directory you supply rather than the Moonshine CDN. */
    public TextToSpeech modelsFrom(File directory) {
        this.assetDirectory = directory;
        return this;
    }

    /**
     * Create this synthesizer as a ZipVoice cloning engine. Call before
     * {@link #load()} so ZipVoice and clone-ASR assets are fetched up front.
     * Clears {@link #voice(String)}. Only then may {@link #cloneFrom} /
     * {@link #startCloning()} be used.
     */
    public TextToSpeech cloning(boolean enabled) {
        this.cloningWanted = enabled;
        if (enabled) {
            this.voiceId = null;
        }
        return this;
    }

    /** Same as {@link #cloning(boolean) cloning(true)}. */
    public TextToSpeech cloning() {
        return cloning(true);
    }

    /** Asset download progress, as a {@code 0..1} fraction plus the file being fetched. */
    public TextToSpeech onProgress(ProgressCallback callback) {
        this.progressCallback = callback;
        return this;
    }

    /** Routes playback to a specific device (see {@link #getAudioOutputDevices}). */
    public TextToSpeech outputDevice(@Nullable AudioDeviceInfo device) {
        this.outputDevice = device;
        return this;
    }

    /** Escape hatch for options the chainable setters don't cover. */
    public TextToSpeech options(List<TranscriberOption> options) {
        if (options != null) {
            this.extraOptions.addAll(options);
        }
        return this;
    }

    /** The language tag this synthesizer will use. */
    public String getLanguage() {
        return languageTag;
    }

    // -- Loading -------------------------------------------------------------

    /**
     * Downloads the voice assets if needed and prepares the synthesizer. Blocks;
     * call from a background thread. With {@link #cloning()}, ZipVoice and clone
     * ASR are both fetched here.
     */
    public void load() {
        if (cloningWanted && cloneSamples == null) {
            build(CLONE_PRESET_VOICE);
            return;
        }
        build(voiceId);
    }

    public boolean isLoaded() {
        return handle >= 0;
    }

    /** True once a voice has been cloned into this synthesizer. */
    public boolean isCloned() {
        return cloneSamples != null;
    }

    /**
     * Clones the voice in a recording and uses it for subsequent synthesis.
     *
     * <p>{@code source} may be a path to a local WAV file or an {@code http(s)}
     * URL. The library trims the recording down to a few seconds of actual
     * speech and transcribes that clip for the vocoder, downloading a small
     * speech-to-text model the first time it needs to.
     */
    public void cloneFrom(String source) {
        cloneFrom(source, null);
    }

    /**
     * As {@link #cloneFrom(String)}, but with the words of the clip supplied, for
     * callers who already know what the reference recording says.
     */
    public void cloneFrom(String source, @Nullable String transcript) {
        WavReader.Audio audio;
        try {
            if (source.startsWith("http://") || source.startsWith("https://")) {
                try (InputStream stream = new URL(source).openStream()) {
                    audio = WavReader.read(stream);
                }
            } else {
                audio = WavReader.read(new File(source));
            }
        } catch (IOException e) {
            throw new RuntimeException("Couldn't read the recording to clone from: " + source, e);
        }
        cloneFrom(audio.samples, audio.sampleRate, transcript);
    }

    /** Clones the voice in a local WAV file. */
    public void cloneFrom(File file) {
        cloneFrom(file.getAbsolutePath(), null);
    }

    /** Clones the voice in {@code samples} (mono float PCM in -1..1). */
    public void cloneFrom(float[] samples, int sampleRate) {
        cloneFrom(samples, sampleRate, null);
    }

    /** Clones the voice in {@code samples}, with the words of the clip supplied. */
    public void cloneFrom(float[] samples, int sampleRate, @Nullable String transcript) {
        requireCloningMode("cloneFrom()");
        float[] clip = clipForCloning(samples, sampleRate);
        cloneSamples = clip;
        cloneTranscript = transcript;
        build(CLONE_VOICE);
    }

    /** Clones the voice captured by a {@link VoiceClone}. */
    public void cloneFrom(VoiceClone clone) {
        cloneFrom(clone, null);
    }

    /** Clones the voice captured by a {@link VoiceClone}, with a known transcript. */
    public void cloneFrom(VoiceClone clone, @Nullable String transcript) {
        float[] audio = clone.getAudio();
        if (audio == null) {
            throw new IllegalStateException(
                    "That VoiceClone has not captured enough speech yet - wait for onReady.");
        }
        if (transcript == null || transcript.isEmpty()) {
            transcript = clone.getTranscript();
        }
        cloneFrom(audio, clone.getSampleRate(), transcript);
    }

    /**
     * Starts capturing a reference voice from the microphone, for cloning. The
     * returned object reports when it has heard enough.
     */
    public VoiceClone startCloning() {
        requireCloningMode("startCloning()");
        checkLoaded();
        return new VoiceClone(appContext, handle);
    }

    /** As {@link #startCloning()}, with the clip length and speech minimum tuned. */
    public VoiceClone startCloning(float clipDurationSeconds, float minimumSpeechSeconds) {
        requireCloningMode("startCloning()");
        checkLoaded();
        return new VoiceClone(appContext, handle, clipDurationSeconds, minimumSpeechSeconds);
    }

    // -- Synthesis -----------------------------------------------------------

    /**
     * Synthesizes text to mono float PCM without playing it. Use {@link #say}
     * to hear it instead.
     */
    public TtsSynthesisResult synthesize(String text) {
        return synthesize(text, null);
    }

    /** As {@link #synthesize(String)}, with per-call options such as {@code speed}. */
    public TtsSynthesisResult synthesize(String text, @Nullable List<TranscriberOption> options) {
        checkLoaded();
        TtsSynthesisResult result = JNI.moonshineTextToSpeech(handle, text, toArray(options));
        if (result == null) {
            throw new RuntimeException("moonshineTextToSpeech failed");
        }
        return result;
    }

    /**
     * Synthesizes speech directly from IPA phonemes, skipping grapheme-to-phoneme
     * conversion.
     *
     * <p>{@code phonemes} is an International Phonetic Alphabet string, as produced by
     * {@link GraphemeToPhonemizer#toIpa}. Passing the phonemes for the same language yields audio
     * equivalent to {@link #synthesize(String)} on the original text, but lets you inspect or edit
     * the phonemes in between (e.g. to fix a name's pronunciation).
     */
    public TtsSynthesisResult synthesizeFromPhonemes(String phonemes) {
        return synthesizeFromPhonemes(phonemes, null);
    }

    /** As {@link #synthesizeFromPhonemes(String)}, with per-call options. */
    public TtsSynthesisResult synthesizeFromPhonemes(String phonemes,
            @Nullable List<TranscriberOption> options) {
        checkLoaded();
        TtsSynthesisResult result = JNI.moonshinePhonemesToSpeech(handle, phonemes,
                toArray(options));
        if (result == null) {
            throw new RuntimeException("moonshinePhonemesToSpeech failed");
        }
        return result;
    }

    // -- say / stop / wait / isTalking ---------------------------------------

    /**
     * Speaks {@code text} out loud, returning once playback finishes. Blocks;
     * call from a background thread.
     *
     * <p>Utterances play in the order they were requested, and synthesis of the
     * next one is pipelined with playback of the current one, so several
     * concurrent {@code say} calls still come out in order without gaps. Long
     * strings are split on an approximate sentence boundary ({@code .},
     * {@code !}, or {@code ?} followed by whitespace) so the first sentence can
     * start sooner. {@link #stop()} cancels everything queued and halts the
     * audio playing now, which makes the waiting calls return early.
     */
    public void say(String text) {
        say(text, null);
    }

    /** As {@link #say(String)}, with per-call synthesis options such as {@code speed}. */
    public void say(String text, @Nullable List<TranscriberOption> options) {
        if (text == null || text.isEmpty()) {
            return;
        }
        for (String sentence : splitSayUtterances(text)) {
            enqueue(sentence, options);
        }
        waitUntilDone();
    }

    /** Speaks each string in order, returning once the last one finishes. */
    public void say(String[] texts) {
        if (texts == null) {
            return;
        }
        for (String text : texts) {
            if (text != null && !text.isEmpty()) {
                for (String sentence : splitSayUtterances(text)) {
                    enqueue(sentence, null);
                }
            }
        }
        waitUntilDone();
    }

    /**
     * Queues {@code text} without waiting for it, for callers that just want the
     * audio to start and have somewhere else to be.
     */
    public void sayInBackground(String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        for (String sentence : splitSayUtterances(text)) {
            enqueue(sentence, null);
        }
    }

    /** Blocks until all queued utterances have been synthesized and played. */
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
     * Clears the utterance queue and stops any audio currently playing.
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
        pendingCount.set(0);
        synchronized (pendingLock) {
            pendingLock.notifyAll();
        }
    }

    /** True if utterances are queued, being synthesized, or currently playing. */
    public boolean isTalking() {
        return pendingCount.get() > 0;
    }

    // -- Load internals ------------------------------------------------------

    /**
     * (Re)creates the native synthesizer for {@code voice}, downloading its assets.
     *
     * <p>The old engine is only torn down once the new one exists, so a failed
     * clone leaves the caller with a working synthesizer.
     */
    private void build(@Nullable String voice) {
        File directory = ensureAssets(voice);
        List<TranscriberOption> options = new ArrayList<>(extraOptions);
        options.add(new TranscriberOption("g2p_root", directory.getAbsolutePath()));

        final int next;
        if (cloneSamples != null) {
            options.add(new TranscriberOption("voice", CLONE_VOICE));
            options.add(new TranscriberOption("zipvoice_clone_sample_rate",
                    Integer.toString(CLONE_SAMPLE_RATE)));
            if (cloneTranscript != null && !cloneTranscript.isEmpty()) {
                options.add(new TranscriberOption("zipvoice_clone_transcript", cloneTranscript));
            }
            next = JNI.moonshineCreateTtsSynthesizerFromMemory(languageTag,
                    new String[] {CLONE_AUDIO_KEY},
                    new byte[][] {floatPcmToLeBytes(cloneSamples)},
                    options.toArray(new TranscriberOption[0]));
        } else {
            if (voice != null) {
                options.add(new TranscriberOption("voice", voice));
            }
            next = JNI.moonshineCreateTtsSynthesizerFromFiles(languageTag, null,
                    options.toArray(new TranscriberOption[0]));
        }
        if (next < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(next));
        }
        if (handle >= 0) {
            JNI.moonshineFreeTtsSynthesizer(handle);
        }
        handle = next;
    }

    private File ensureAssets(@Nullable String voice) {
        if (assetDirectory != null) {
            return assetDirectory;
        }
        try {
            return Models.ensureOne(appContext, ModelSpec.tts(languageTag, voice), null,
                    progressCallback);
        } catch (IOException e) {
            throw new RuntimeException("Failed to download the text-to-speech assets", e);
        }
    }

    /**
     * Trims a reference recording to the few seconds of speech ZipVoice wants,
     * resampling to 16 kHz on the way.
     */
    private float[] clipForCloning(float[] samples, int sampleRate) {
        checkLoaded();
        if (sampleRate == CLONE_SAMPLE_RATE && samples.length <= CLONE_SAMPLE_RATE * 10) {
            return samples;
        }
        SpeechClip clip = JNI.moonshineExtractSpeechClip(samples, sampleRate, handle, 4f, 2f);
        if (clip != null && clip.audio != null) {
            return clip.audio;
        }
        // Nothing clearly speech-like. Rather than refuse outright, take the best
        // window the detector found - a poor clone beats no clone for a caller
        // who explicitly handed us this recording.
        clip = JNI.moonshineExtractSpeechClip(samples, sampleRate, handle, 4f, 0f);
        if (clip != null && clip.audio != null) {
            return clip.audio;
        }
        throw new IllegalArgumentException(
                "Couldn't find enough speech in that recording to clone from.");
    }

    private static byte[] floatPcmToLeBytes(float[] pcm) {
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(pcm.length * 4)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float v : pcm) {
            bb.putFloat(v);
        }
        return bb.array();
    }

    private void checkLoaded() {
        if (handle < 0) {
            throw new IllegalStateException("Call load() before synthesizing.");
        }
    }

    /**
     * Approximate sentence split for {@link #say}: break on {@code .} / {@code !}
     * / {@code ?} / {@code :} followed by whitespace so the first clause can
     * start sooner.
     */
    static List<String> splitSayUtterances(String text) {
        String stripped = text == null ? "" : text.trim();
        if (stripped.isEmpty()) {
            return java.util.Collections.emptyList();
        }
        List<String> parts = new ArrayList<>();
        int start = 0;
        int i = 0;
        final int n = stripped.length();
        while (i < n) {
            char ch = stripped.charAt(i);
            if ((ch == '.' || ch == '!' || ch == '?' || ch == ':') && i + 1 < n
                    && Character.isWhitespace(stripped.charAt(i + 1))) {
                int end = i + 1;
                int j = i + 1;
                while (j < n && Character.isWhitespace(stripped.charAt(j))) {
                    j++;
                }
                String piece = stripped.substring(start, end).trim();
                if (!piece.isEmpty()) {
                    parts.add(piece);
                }
                start = j;
                i = j;
                continue;
            }
            i++;
        }
        String tail = stripped.substring(start).trim();
        if (!tail.isEmpty()) {
            parts.add(tail);
        }
        return parts;
    }

    private void requireCloningMode(String what) {
        if (!cloningWanted) {
            throw new IllegalStateException(
                    "Call cloning() before load() to use " + what
                            + ". Catalog voices and cloning are separate synthesizer modes.");
        }
    }

    /** Package-visible for instrumented tests that need an extractSpeechClip handle. */
    int getHandleForTests() {
        checkLoaded();
        return handle;
    }

    private static TranscriberOption[] toArray(@Nullable List<TranscriberOption> options) {
        if (options == null || options.isEmpty()) {
            return null;
        }
        return options.toArray(new TranscriberOption[0]);
    }

    // -- Dependency queries --------------------------------------------------

    /** Comma-separated G2P asset keys (see {@code moonshine_get_g2p_dependencies}). */
    public static String getG2pDependencies(String languages, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        String json = JNI.moonshineGetG2pDependencies(languages, toArray(options));
        if (json == null) {
            throw new RuntimeException("moonshineGetG2pDependencies failed");
        }
        return json;
    }

    /** JSON groups manifest of merged G2P + vocoder (+ ZipVoice clone ASR) assets. */
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

    // -- Queue infrastructure ------------------------------------------------

    private static class SayRequest {
        final String text;
        @Nullable final List<TranscriberOption> options;

        SayRequest(String text, @Nullable List<TranscriberOption> options) {
            this.text = text;
            this.options = options;
        }
    }

    private static class PlayItem {
        final float[] samples;
        final int sampleRate;

        PlayItem(float[] samples, int sampleRate) {
            this.samples = samples;
            this.sampleRate = sampleRate;
        }
    }

    private final LinkedBlockingQueue<SayRequest> sayQueue = new LinkedBlockingQueue<>();
    private final ArrayBlockingQueue<PlayItem> playQueue = new ArrayBlockingQueue<>(1);
    private volatile boolean stopRequested = false;
    @Nullable private Thread synthThread;
    @Nullable private Thread playThread;
    private final Object workerLock = new Object();

    private final AtomicInteger pendingCount = new AtomicInteger(0);
    private final Object pendingLock = new Object();

    private void enqueue(String text, @Nullable List<TranscriberOption> options) {
        checkLoaded();
        pendingCount.incrementAndGet();
        sayQueue.add(new SayRequest(text, options));
        ensureWorkers();
    }

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
            SayRequest request;
            try {
                request = sayQueue.poll(100, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }
            if (request == null) continue;
            if (stopRequested) {
                decrementPending();
                break;
            }

            try {
                TtsSynthesisResult result = synthesize(request.text, request.options);
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
                PlayItem item = new PlayItem(samples, sampleRate);
                while (!stopRequested) {
                    if (playQueue.offer(item, 100, TimeUnit.MILLISECONDS)) break;
                }
                if (stopRequested) {
                    decrementPending();
                    break;
                }
            } catch (Exception e) {
                Log.w("MoonshineTTS", "Synthesis failed", e);
                decrementPending();
            }
        }
    }

    private void playWorker() {
        while (!stopRequested) {
            PlayItem item;
            try {
                item = playQueue.poll(100, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }
            if (item == null) continue;
            if (stopRequested) {
                decrementPending();
                break;
            }

            try {
                playOneItem(item);
            } catch (Exception e) {
                Log.w("MoonshineTTS", "Playback failed", e);
            } finally {
                decrementPending();
            }
        }
    }

    private void playOneItem(PlayItem item) {
        AudioDeviceInfo device = outputDevice;
        int wantDeviceId = device != null ? device.getId() : -1;
        synchronized (sayLock) {
            if (stopRequested) return;
            AudioTrack track = obtainSayTrackLocked(wantDeviceId, device, item.sampleRate);
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

    private static void drainQueue(java.util.concurrent.BlockingQueue<?> queue) {
        while (queue.poll() != null) { /* discard */ }
    }

    private void joinWorkers() {
        Thread st;
        Thread pt;
        synchronized (workerLock) {
            st = synthThread;
            pt = playThread;
        }
        try {
            if (st != null && st.isAlive() && st != Thread.currentThread()) st.join(2000);
        } catch (InterruptedException ignored) {
        }
        try {
            if (pt != null && pt.isAlive() && pt != Thread.currentThread()) pt.join(2000);
        } catch (InterruptedException ignored) {
        }
        synchronized (workerLock) {
            synthThread = null;
            playThread = null;
        }
    }

    // -- Audio track management ----------------------------------------------

    /**
     * Lists output devices suitable for {@link #outputDevice(AudioDeviceInfo)} (e.g. speaker,
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

    private AudioTrack obtainSayTrackLocked(int wantDeviceId, @Nullable AudioDeviceInfo device,
            int sampleRateHz) {
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
            throw new RuntimeException("AudioTrack.getMinBufferSize failed for sampleRate="
                    + sampleRateHz);
        }
        AudioTrack track = buildAudioTrack(appContext, attrs, format, minBufBytes);
        if (device != null) {
            track.setPreferredDevice(device);
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

    /** Releases the synthesizer, its playback resources, and any clone-clip model. */
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
