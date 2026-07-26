package ai.moonshine.voice;

/**
 * Handle returned by {@link CatalogLoader} / {@code loadFromCatalog} that lets a caller abort an
 * in-flight download+construct run (e.g. from {@code onDestroy}). Cancelling interrupts the worker
 * thread; the terminal {@link LoadCallback} may still fire with an error describing the
 * interruption, and any partially downloaded file is left as a resumable {@code .part}.
 */
public interface Cancellable {
    /** Requests cancellation. Safe to call multiple times and from any thread. */
    void cancel();

    /** @return true once {@link #cancel()} has been called. */
    boolean isCancelled();
}
