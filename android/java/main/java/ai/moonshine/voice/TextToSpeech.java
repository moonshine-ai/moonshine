package ai.moonshine.voice;

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

    public void close() {
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
