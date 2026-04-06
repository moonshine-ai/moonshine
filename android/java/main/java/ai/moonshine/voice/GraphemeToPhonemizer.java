package ai.moonshine.voice;

import java.util.ArrayList;
import java.util.List;

/**
 * Grapheme-to-phoneme (IPA) via the Moonshine native API.
 *
 * <p>Aligns with Python {@code moonshine_voice.GraphemeToPhonemizer}: construct with a language tag and
 * {@code g2p_root} on disk, then {@link #toIpa}. No download or caching is performed here.
 */
public class GraphemeToPhonemizer {
    private int handle = -1;
    private final String language;

    /**
     * @param language  Moonshine language tag (e.g. {@code en_us}).
     * @param filenames Optional canonical lexicon / model keys; {@code null} or empty lets the engine resolve
     *                  paths under {@code g2p_root}.
     * @param g2pRoot   Asset root directory; stored as option {@code g2p_root}.
     * @param options   Extra native options (e.g. {@code spanish_narrow_obstruents}).
     */
    public GraphemeToPhonemizer(String language, String[] filenames, String g2pRoot,
            List<TranscriberOption> options) {
        this.language = language;
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateGraphemeToPhonemizerFromFiles(language, filenames,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        this.handle = h;
    }

    public GraphemeToPhonemizer(String language, String g2pRoot, List<TranscriberOption> options) {
        this(language, null, g2pRoot, options);
    }

    public static GraphemeToPhonemizer fromMemory(String language, String[] filenames, byte[][] memory,
            String g2pRoot, List<TranscriberOption> options) {
        JNI.ensureLibraryLoaded();
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("g2p_root", g2pRoot));
        if (options != null) {
            opts.addAll(options);
        }
        int h = JNI.moonshineCreateGraphemeToPhonemizerFromMemory(language, filenames, memory,
                opts.toArray(new TranscriberOption[0]));
        if (h < 0) {
            throw new RuntimeException(JNI.moonshineErrorToString(h));
        }
        return new GraphemeToPhonemizer(language, h);
    }

    private GraphemeToPhonemizer(String language, int handle) {
        this.language = language;
        this.handle = handle;
    }

    /** Same as {@link TextToSpeech#getG2pDependencies(String, List)}. */
    public static String getG2pDependencies(String languages, List<TranscriberOption> options) {
        return TextToSpeech.getG2pDependencies(languages, options);
    }

    private static TranscriberOption[] toArray(List<TranscriberOption> options) {
        if (options == null || options.isEmpty()) {
            return null;
        }
        return options.toArray(new TranscriberOption[0]);
    }

    public String getLanguage() {
        return language;
    }

    /** Convert text to a single IPA string (native {@code moonshine_text_to_phonemes}). */
    public String toIpa(String text, List<TranscriberOption> options) {
        String ipa = JNI.moonshineTextToPhonemes(handle, text, toArray(options));
        if (ipa == null) {
            throw new RuntimeException("moonshineTextToPhonemes failed");
        }
        return ipa;
    }

    public String toIpa(String text) {
        return toIpa(text, null);
    }

    public void close() {
        if (handle >= 0) {
            JNI.moonshineFreeGraphemeToPhonemizer(handle);
            handle = -1;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
}
