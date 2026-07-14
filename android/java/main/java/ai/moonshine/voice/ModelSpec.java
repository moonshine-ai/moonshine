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
 * {@code ModelSpec.intent(null, "q4")}, or {@code ModelSpec.tts("en_us", "kokoro_af_heart")}.
 */
public final class ModelSpec {

    public enum Type { STT, TTS, INTENT, G2P }

    public final Type type;
    /** Language code / English name (STT, TTS, G2P) or embedding model id (INTENT). May be null for INTENT. */
    @Nullable public final String primary;
    /** STT only: a {@code MOONSHINE_MODEL_ARCH_*} value, or null for the language default. */
    @Nullable public final Integer modelArch;
    /** STT only: also fetch the alphanumeric spelling model when published for the language. */
    public final boolean includeSpelling;
    /** TTS only: prefixed voice id (e.g. {@code kokoro_af_heart}), or null for the default. */
    @Nullable public final String voice;
    /** INTENT only: embedding variant (e.g. {@code q4}), or null for the default. */
    @Nullable public final String variant;

    private ModelSpec(Type type, @Nullable String primary, @Nullable Integer modelArch,
                      boolean includeSpelling, @Nullable String voice, @Nullable String variant) {
        this.type = type;
        this.primary = primary;
        this.modelArch = modelArch;
        this.includeSpelling = includeSpelling;
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
        return new ModelSpec(Type.STT, language, modelArch, includeSpelling, null, null);
    }

    /** Text-to-speech assets for {@code language} and optional prefixed {@code voice}. */
    public static ModelSpec tts(String language, @Nullable String voice) {
        return new ModelSpec(Type.TTS, language, null, false, voice, null);
    }

    /** Intent-recognition embedding model. Pass {@code null} for the default model / variant. */
    public static ModelSpec intent(@Nullable String modelName, @Nullable String variant) {
        return new ModelSpec(Type.INTENT, modelName, null, false, null, variant);
    }

    /** Grapheme-to-phoneme assets for {@code language}. */
    public static ModelSpec g2p(String language) {
        return new ModelSpec(Type.G2P, language, null, false, null, null);
    }

    /**
     * Builds the option list passed to the native dependency call.
     *
     * @param root download root; used for the {@code g2p_root} option on TTS/G2P specs so the
     *             manifest reflects on-disk state. May be null for STT/INTENT.
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
                break;
            case INTENT:
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
        }
        return options;
    }
}
