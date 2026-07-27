package ai.moonshine.voice;

import android.content.Context;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves model files to directories on disk, downloading whatever is missing.
 *
 * <p>This is the blocking half of {@link CatalogLoader}: the engines' own
 * {@code load()} methods are documented as background-thread calls, so they do
 * not need the executor-and-callback dance, only the download itself.
 */
final class Models {

    private Models() {}

    /**
     * Returns the directory holding each spec's files, fetching any that are
     * not there yet, and reports combined progress across all of them.
     *
     * @param root where to cache; null uses the managed {@link ModelCache}.
     */
    static Map<ModelSpec, File> ensure(Context context, List<ModelSpec> specs,
            @Nullable File root, @Nullable ProgressCallback onProgress) throws IOException {
        Context appContext = context.getApplicationContext();
        AssetDownloader downloader = new AssetDownloader();
        Map<ModelSpec, File> directories = new LinkedHashMap<>();
        final int specCount = specs.size();
        for (int specIndex = 0; specIndex < specCount; specIndex++) {
            ModelSpec spec = specs.get(specIndex);
            File directory = ModelCache.directoryFor(appContext, spec, root);
            final int index = specIndex;
            AssetDownloader.ProgressListener listener = onProgress == null ? null
                    : (relativePath, fileIndex, totalFiles, bytesDownloaded, bytesTotal) -> {
                        float withinFile = bytesTotal > 0
                                ? (float) bytesDownloaded / (float) bytesTotal : 0f;
                        float withinSpec = totalFiles > 0
                                ? ((fileIndex - 1) + withinFile) / totalFiles : withinFile;
                        float overall = (index + withinSpec) / specCount;
                        onProgress.onProgress(Math.min(1f, Math.max(0f, overall)), relativePath);
                    };
            downloader.ensureModelPresent(directory, spec, listener);
            directories.put(spec, directory);
        }
        if (onProgress != null) {
            onProgress.onProgress(1f, "");
        }
        return directories;
    }

    /** Single-spec convenience wrapper around {@link #ensure}. */
    static File ensureOne(Context context, ModelSpec spec, @Nullable File root,
            @Nullable ProgressCallback onProgress) throws IOException {
        return ensure(context, java.util.Collections.singletonList(spec), root, onProgress)
                .get(spec);
    }
}
