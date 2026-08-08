package ai.moonshine.voice;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * Describes which model's files {@link AssetDownloader} (or {@link MoonshineDownloadWorker}) should
 * resolve and download. Each spec maps to one of the native dependency APIs so the exact file list
 * always comes from the library rather than being hardcoded by the app.
 *
 * <p>Build one with the static factories, e.g. {@code ModelSpec.stt("en")},
 * {@code ModelSpec.embedding(null, "q4")}, or {@code ModelSpec.tts("en_us", "kokoro_af_heart")}.
 */
public final class ModelSpec {

    public enum Type { STT, TTS, EMBEDDING, G2P, DIARIZATION }

    public final Type type;
    /**
     * Language code / English name (STT, TTS, G2P) or embedding model id (EMBEDDING). May be null
     * for EMBEDDING.
     */
    @Nullable public final String primary;
    /** STT only: a {@code MOONSHINE_MODEL_ARCH_*} value, or null for the language default. */
    @Nullable public final Integer modelArch;
    /** STT only: also fetch the alphanumeric spelling model when published for the language. */
    public final boolean includeSpelling;
    /**
     * STT only: also fetch the optional attention decoder used by the {@code word_timestamps}
     * transcriber option (roughly doubling the download). Leave false unless you need word-level
     * timestamps.
     */
    public final boolean includeWordTimestamps;
    /** TTS only: prefixed voice id (e.g. {@code kokoro_af_heart}), or null for the default. */
    @Nullable public final String voice;
    /** EMBEDDING only: variant (e.g. {@code q4}), or null for the default. */
    @Nullable public final String variant;

    private ModelSpec(Type type, @Nullable String primary, @Nullable Integer modelArch,
                      boolean includeSpelling, boolean includeWordTimestamps,
                      @Nullable String voice, @Nullable String variant) {
        this.type = type;
        this.primary = primary;
        this.modelArch = modelArch;
        this.includeSpelling = includeSpelling;
        this.includeWordTimestamps = includeWordTimestamps;
        this.voice = voice;
        this.variant = variant;
    }

    /** Speech-to-text model, using the default architecture for {@code language}. */
    public static ModelSpec stt(String language) {
        return stt(language, null, false);
    }

    /** Speech-to-text model with an explicit architecture and optional spelling model. */
    public static ModelSpec stt(String language, @Nullable Integer modelArch,
                                boolean includeSpelling) {
        return stt(language, modelArch, includeSpelling, false);
    }

    /**
     * Speech-to-text model with an explicit architecture, and optional spelling / word-timestamp
     * files.
     */
    public static ModelSpec stt(String language, @Nullable Integer modelArch,
                                boolean includeSpelling, boolean includeWordTimestamps) {
        return new ModelSpec(Type.STT, language, modelArch, includeSpelling,
                includeWordTimestamps, null, null);
    }

    /** Text-to-speech assets for {@code language} and optional prefixed {@code voice}. */
    public static ModelSpec tts(String language, @Nullable String voice) {
        return new ModelSpec(Type.TTS, language, null, false, false, voice, null);
    }

    /** Text embedding model. Pass {@code null} for the default model / variant. */
    public static ModelSpec embedding(@Nullable String modelName, @Nullable String variant) {
        return new ModelSpec(Type.EMBEDDING, modelName, null, false, false, null, variant);
    }

    /**
     * Speaker diarization models, needed by the {@code identify_speakers} transcriber option.
     * There is one set and it has no variants. About 8.2 MB.
     */
    public static ModelSpec diarization() {
        return new ModelSpec(Type.DIARIZATION, null, null, false, false, null, null);
    }

    /** Grapheme-to-phoneme assets for {@code language}. */
    public static ModelSpec g2p(String language) {
        return new ModelSpec(Type.G2P, language, null, false, false, null, null);
    }

    /**
     * Builds the option list passed to the native dependency call.
     *
     * @param root download root; used for the {@code g2p_root} option on TTS/G2P specs so the
     *             manifest reflects on-disk state. May be null for STT/EMBEDDING.
     */
    List<TranscriberOption> toOptions(@Nullable String root) {
        List<TranscriberOption> options = new ArrayList<>();
        switch (type) {
            case STT:
                if (modelArch != null) {
                    options.add(new TranscriberOption("model_arch", String.valueOf(modelArch)));
                }
                if (includeSpelling) {
                    options.add(new TranscriberOption("include_spelling", "true"));
                }
                if (includeWordTimestamps) {
                    options.add(new TranscriberOption("word_timestamps", "true"));
                }
                break;
            case EMBEDDING:
                if (variant != null) {
                    options.add(new TranscriberOption("variant", variant));
                }
                break;
            case TTS:
                if (root != null) {
                    options.add(new TranscriberOption("g2p_root", root));
                }
                if (voice != null) {
                    options.add(new TranscriberOption("voice", voice));
                }
                break;
            case G2P:
                if (root != null) {
                    options.add(new TranscriberOption("g2p_root", root));
                }
                break;
            case DIARIZATION:
                break;
        }
        return options;
    }
}
