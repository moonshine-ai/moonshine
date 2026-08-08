package ai.moonshine.voice;

import android.content.Context;

import java.io.File;
import java.util.Locale;

/**
 * Default, binding-managed on-disk location for models downloaded via {@link CatalogLoader} and the
 * per-engine {@code loadFromCatalog} helpers.
 *
 * <p>Apps that want to control where models live can pass an explicit directory to those helpers
 * (or keep using {@link AssetDownloader} directly); this class just provides a sensible default so
 * the common case needs no {@code mkdirs} boilerplate.
 *
 * <p>The default root is {@code getNoBackupFilesDir()/moonshine-models} (large, re-downloadable
 * files that should not count against Android auto-backup). Each {@link ModelSpec} maps to a stable
 * subdirectory so different models never clobber one another and re-loading an already-downloaded
 * model is instant. Mirrors the Swift {@code ModelCache}.
 */
public final class ModelCache {

    private ModelCache() {}

    /** The default root under which Moonshine stores downloaded models. */
    public static File defaultRoot(Context context) {
        File base = context.getNoBackupFilesDir();
        if (base == null) {
            base = context.getFilesDir();
        }
        return new File(base, "moonshine-models");
    }

    /** A stable, filesystem-safe subdirectory name uniquely identifying {@code spec}'s files. */
    public static String key(ModelSpec spec) {
        StringBuilder builder = new StringBuilder();
        switch (spec.type) {
            case STT:
                builder.append("stt-").append(spec.primary).append('-')
                        .append(spec.modelArch != null ? spec.modelArch.toString() : "default");
                if (spec.includeSpelling) {
                    builder.append("-spelling");
                }
                if (spec.includeWordTimestamps) {
                    builder.append("-wt");
                }
                break;
            case TTS:
                // Voices of a language share one G2P root (the manifest lays them out under
                // distinct key prefixes), so the directory is keyed by language only.
                builder.append("tts-").append(spec.primary);
                break;
            case EMBEDDING:
                // The "intent-" prefix predates the embedding naming; keeping it
                // means existing on-disk caches are still found.
                builder.append("intent-")
                        .append(spec.primary != null ? spec.primary : "default")
                        .append('-')
                        .append(spec.variant != null ? spec.variant : "default");
                break;
            case G2P:
                builder.append("g2p-").append(spec.primary);
                break;
            case DIARIZATION:
                builder.append("diarization-community1");
                break;
        }
        return sanitize(builder.toString());
    }

    /**
     * Returns (creating if needed) the directory that {@code spec}'s files should live in, under
     * {@code root} (or {@link #defaultRoot(Context)} when {@code root} is null).
     */
    public static File directoryFor(Context context, ModelSpec spec, File root) {
        File base = root != null ? root : defaultRoot(context);
        File directory = new File(base, key(spec));
        directory.mkdirs();
        return directory;
    }

    private static String sanitize(String raw) {
        StringBuilder out = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            boolean ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                    || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
            out.append(ok ? c : '-');
        }
        return out.toString().toLowerCase(Locale.ROOT);
    }
}
