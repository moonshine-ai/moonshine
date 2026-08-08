package ai.moonshine.voice;

/**
 * Reports how far a {@code load()} call has got, as a fraction of the whole.
 *
 * <p>The older {@link DownloadProgress} form (file index, byte counts) is still
 * available through {@link CatalogLoader} for apps that draw a per-file UI, but
 * a single {@code 0..1} number is what a progress bar actually wants.
 */
public interface ProgressCallback {
    /**
     * @param fraction how much of the download is done, from 0 to 1.
     * @param file     the file currently being fetched, for a status line.
     */
    void onProgress(float fraction, String file);
}
