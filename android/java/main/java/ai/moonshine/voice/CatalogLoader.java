package ai.moonshine.voice;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import androidx.annotation.Nullable;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The one primitive that turns "I need these models" into "here is a ready engine", handling the
 * off-main-thread download, main-thread progress/result delivery, and cancellation that every
 * Android app otherwise hand-writes with {@code Thread}/{@code runOnUiThread}.
 *
 * <p>Given a list of {@link ModelSpec}s it downloads each (skipping files already present, into a
 * managed {@link ModelCache} directory) on a shared background executor, reports {@link
 * DownloadProgress} on the main thread, then runs the caller's {@link Builder} on the worker thread
 * to construct the engine(s) from the resolved directories and delivers the result (or error) on
 * the main thread.
 *
 * <p>Most apps do not need this: {@link MicTranscriber#load()}, {@link TextToSpeech#load()} and
 * {@link AgentFlow#load()} download what they need themselves, blocking on the caller's own
 * background thread. Reach for this when you want the callback-and-cancellation shape instead, or
 * need a per-file progress UI that the {@code 0..1} {@link ProgressCallback} does not give you.
 */
public final class CatalogLoader {

    /** Builds the engine(s) on the worker thread from the resolved per-spec directories. */
    public interface Builder<T> {
        T build(Map<ModelSpec, File> directories) throws Exception;
    }

    private static final ExecutorService EXECUTOR = Executors.newCachedThreadPool(runnable -> {
        Thread thread = new Thread(runnable, "moonshine-catalog-loader");
        thread.setDaemon(true);
        return thread;
    });

    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private CatalogLoader() {}

    /** Downloads {@code specs} into the managed cache, builds via {@code builder}, reports on main. */
    public static <T> Cancellable load(Context context, List<ModelSpec> specs,
            Builder<T> builder, LoadCallback<T> callback) {
        return load(context, specs, null, new AssetDownloader(), builder, callback);
    }

    /**
     * As {@link #load(Context, List, Builder, LoadCallback)}, but with an explicit cache root
     * (per-spec subdirectories are still derived under it) and a caller-supplied downloader.
     */
    public static <T> Cancellable load(Context context, List<ModelSpec> specs,
            @Nullable File cacheRoot, AssetDownloader downloader,
            Builder<T> builder, LoadCallback<T> callback) {
        final Context appContext = context.getApplicationContext();
        final AtomicBoolean cancelled = new AtomicBoolean(false);
        final CancellableHandle handle = new CancellableHandle(cancelled);

        Future<?> future = EXECUTOR.submit(() -> {
            try {
                Map<ModelSpec, File> directories = new LinkedHashMap<>();
                for (ModelSpec spec : specs) {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        throw new InterruptedException("Model download cancelled");
                    }
                    File directory = ModelCache.directoryFor(appContext, spec, cacheRoot);
                    downloader.ensureModelPresent(directory, spec,
                            (relativePath, fileIndex, totalFiles, bytesDownloaded, bytesTotal) -> {
                                final DownloadProgress progress = new DownloadProgress(
                                        relativePath, fileIndex, totalFiles, bytesDownloaded,
                                        bytesTotal);
                                MAIN.post(() -> callback.onProgress(progress));
                            });
                    directories.put(spec, directory);
                }
                final T engine = builder.build(directories);
                MAIN.post(() -> callback.onSuccess(engine));
            } catch (Throwable error) {
                MAIN.post(() -> callback.onError(error));
            }
        });
        handle.setFuture(future);
        return handle;
    }

    private static final class CancellableHandle implements Cancellable {
        private final AtomicBoolean cancelled;
        @Nullable private volatile Future<?> future;

        CancellableHandle(AtomicBoolean cancelled) {
            this.cancelled = cancelled;
        }

        void setFuture(Future<?> future) {
            this.future = future;
            if (cancelled.get()) {
                future.cancel(true);
            }
        }

        @Override
        public void cancel() {
            cancelled.set(true);
            Future<?> current = future;
            if (current != null) {
                current.cancel(true);
            }
        }

        @Override
        public boolean isCancelled() {
            return cancelled.get();
        }
    }
}
