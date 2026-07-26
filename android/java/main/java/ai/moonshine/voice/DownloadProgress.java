package ai.moonshine.voice;

/**
 * Immutable snapshot of a single file's download progress, delivered on the main thread by
 * {@link CatalogLoader} / the per-engine {@code loadFromCatalog} helpers.
 *
 * <p>Mirrors the Swift {@code DownloadProgress} value type so the two bindings report progress the
 * same way.
 */
public final class DownloadProgress {
    /** Path of the current file relative to the model directory (e.g. {@code encoder.ort}). */
    public final String relativePath;
    /** 1-based index of the file being downloaded in the current model's run. */
    public final int fileIndex;
    /** Total number of files that will be downloaded in the current model's run. */
    public final int totalFiles;
    /** Bytes written for the current file so far. */
    public final long bytesDownloaded;
    /** Total bytes for the current file, or {@code -1} if the size is unknown. */
    public final long bytesTotal;

    public DownloadProgress(String relativePath, int fileIndex, int totalFiles,
                            long bytesDownloaded, long bytesTotal) {
        this.relativePath = relativePath;
        this.fileIndex = fileIndex;
        this.totalFiles = totalFiles;
        this.bytesDownloaded = bytesDownloaded;
        this.bytesTotal = bytesTotal;
    }

    /**
     * Fraction (0..1) of the current file downloaded, or {@code -1} when the total size is unknown.
     */
    public double fraction() {
        if (bytesTotal <= 0) {
            return -1;
        }
        return Math.min(1.0, (double) bytesDownloaded / (double) bytesTotal);
    }
}
