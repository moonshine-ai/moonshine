package ai.moonshine.voice;

/**
 * Receives the outcome of a {@link CatalogLoader} / {@code loadFromCatalog} run. Every method is
 * invoked on the main (UI) thread, so implementations can touch views directly without posting.
 *
 * <p>Exactly one terminal callback fires per run: {@link #onSuccess(Object)} once the models are
 * downloaded and the engine is constructed, or {@link #onError(Throwable)} if downloading or
 * construction fails (including cancellation, which surfaces as an {@link InterruptedException} or
 * {@link java.io.IOException}).
 *
 * @param <T> the engine type produced on success.
 */
public interface LoadCallback<T> {
    /**
     * Reports download progress for the current file. Called zero or more times before the terminal
     * callback. Default is a no-op so callers that only care about the result can omit it.
     */
    default void onProgress(DownloadProgress progress) {}

    /** The models are present and {@code engine} is ready to use. */
    void onSuccess(T engine);

    /** Downloading or construction failed (or the run was cancelled). */
    void onError(Throwable error);
}
