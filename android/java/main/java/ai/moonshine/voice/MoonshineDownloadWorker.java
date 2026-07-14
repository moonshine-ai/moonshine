package ai.moonshine.voice;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.BackoffPolicy;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Runs {@link AssetDownloader#ensureModelPresent} under WorkManager so model downloads survive
 * process death, honor network constraints (e.g. unmetered-only), and retry with exponential
 * backoff. Progress is published through {@link androidx.work.WorkInfo#getProgress()} using the
 * {@code PROGRESS_*} keys.
 *
 * <p>Enqueue with {@link #buildRequest} and observe via WorkManager, for example:
 * <pre>{@code
 * File root = new File(context.getFilesDir(), "moonshine/tiny-en");
 * OneTimeWorkRequest request =
 *     MoonshineDownloadWorker.buildRequest(root, ModelSpec.stt("en"), true);
 * WorkManager.getInstance(context).enqueue(request);
 * }</pre>
 */
public final class MoonshineDownloadWorker extends Worker {

    // Input keys (set by buildRequest / toInputData).
    public static final String KEY_ROOT = "moonshine.root";
    public static final String KEY_TYPE = "moonshine.type";
    public static final String KEY_PRIMARY = "moonshine.primary";
    public static final String KEY_MODEL_ARCH = "moonshine.model_arch";
    public static final String KEY_INCLUDE_SPELLING = "moonshine.include_spelling";
    public static final String KEY_VOICE = "moonshine.voice";
    public static final String KEY_VARIANT = "moonshine.variant";

    // Progress keys (published via setProgressAsync).
    public static final String PROGRESS_RELATIVE_PATH = "moonshine.progress.relative_path";
    public static final String PROGRESS_FILE_INDEX = "moonshine.progress.file_index";
    public static final String PROGRESS_TOTAL_FILES = "moonshine.progress.total_files";
    public static final String PROGRESS_BYTES_DOWNLOADED = "moonshine.progress.bytes_downloaded";
    public static final String PROGRESS_BYTES_TOTAL = "moonshine.progress.bytes_total";

    // Output key on failure.
    public static final String OUTPUT_ERROR = "moonshine.error";

    /** Sentinel meaning "no explicit model_arch" so a default can be used. */
    private static final int NO_MODEL_ARCH = Integer.MIN_VALUE;
    private static final int MAX_ATTEMPTS = 3;

    public MoonshineDownloadWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    /**
     * Builds a one-time work request that downloads {@code spec} into {@code root}.
     *
     * @param requireUnmetered when true, only run over an unmetered (e.g. Wi-Fi) connection.
     */
    public static OneTimeWorkRequest buildRequest(File root, ModelSpec spec,
                                                  boolean requireUnmetered) {
        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(
                        requireUnmetered ? NetworkType.UNMETERED : NetworkType.CONNECTED)
                .build();
        return new OneTimeWorkRequest.Builder(MoonshineDownloadWorker.class)
                .setInputData(toInputData(root, spec))
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL,
                        OneTimeWorkRequest.MIN_BACKOFF_MILLIS, TimeUnit.MILLISECONDS)
                .build();
    }

    /** Serializes {@code root} + {@code spec} into WorkManager input {@link Data}. */
    public static Data toInputData(File root, ModelSpec spec) {
        Data.Builder builder = new Data.Builder()
                .putString(KEY_ROOT, root.getAbsolutePath())
                .putString(KEY_TYPE, spec.type.name())
                .putBoolean(KEY_INCLUDE_SPELLING, spec.includeSpelling)
                .putInt(KEY_MODEL_ARCH, spec.modelArch != null ? spec.modelArch : NO_MODEL_ARCH);
        if (spec.primary != null) {
            builder.putString(KEY_PRIMARY, spec.primary);
        }
        if (spec.voice != null) {
            builder.putString(KEY_VOICE, spec.voice);
        }
        if (spec.variant != null) {
            builder.putString(KEY_VARIANT, spec.variant);
        }
        return builder.build();
    }

    private static ModelSpec specFromData(Data data) {
        ModelSpec.Type type = ModelSpec.Type.valueOf(data.getString(KEY_TYPE));
        String primary = data.getString(KEY_PRIMARY);
        switch (type) {
            case STT: {
                int arch = data.getInt(KEY_MODEL_ARCH, NO_MODEL_ARCH);
                Integer modelArch = arch == NO_MODEL_ARCH ? null : arch;
                return ModelSpec.stt(primary, modelArch, data.getBoolean(KEY_INCLUDE_SPELLING, false));
            }
            case TTS:
                return ModelSpec.tts(primary, data.getString(KEY_VOICE));
            case INTENT:
                return ModelSpec.intent(primary, data.getString(KEY_VARIANT));
            case G2P:
            default:
                return ModelSpec.g2p(primary);
        }
    }

    @NonNull
    @Override
    public Result doWork() {
        Data input = getInputData();
        String rootPath = input.getString(KEY_ROOT);
        if (rootPath == null) {
            return Result.failure(
                    new Data.Builder().putString(OUTPUT_ERROR, "missing root path").build());
        }
        File root = new File(rootPath);
        ModelSpec spec = specFromData(input);

        AssetDownloader downloader = new AssetDownloader();
        try {
            downloader.ensureModelPresent(root, spec,
                    (relativePath, fileIndex, totalFiles, bytesDownloaded, bytesTotal) ->
                            setProgressAsync(new Data.Builder()
                                    .putString(PROGRESS_RELATIVE_PATH, relativePath)
                                    .putInt(PROGRESS_FILE_INDEX, fileIndex)
                                    .putInt(PROGRESS_TOTAL_FILES, totalFiles)
                                    .putLong(PROGRESS_BYTES_DOWNLOADED, bytesDownloaded)
                                    .putLong(PROGRESS_BYTES_TOTAL, bytesTotal)
                                    .build()));
            return Result.success();
        } catch (IOException e) {
            // Transient failures (connectivity, partial writes) get a bounded number of retries;
            // beyond that, surface a terminal failure so observers stop waiting.
            if (getRunAttemptCount() + 1 < MAX_ATTEMPTS) {
                return Result.retry();
            }
            String message = e.getMessage() != null ? e.getMessage() : "download failed";
            return Result.failure(new Data.Builder().putString(OUTPUT_ERROR, message).build());
        }
    }
}
