package ai.moonshine.voice;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Downloads the model/data files a Moonshine engine needs into an app-chosen directory, then hands
 * back that directory for loading with {@link Transcriber}, {@link TextToSpeech}, or
 * {@link IntentRecognizer}.
 *
 * <p>This is <b>opt-in</b>: apps that bundle their models never need it and default behavior is
 * unchanged. File lists come from the native dependency catalog (via
 * {@link Transcriber#getSttDependencies}, {@link IntentRecognizer#getIntentDependencies},
 * {@link TextToSpeech#getTtsDependencies}, {@link TextToSpeech#getG2pDependencies}), downloads are
 * written atomically (through a {@code .part} file), resume across interruptions with HTTP Range,
 * and report progress through an optional listener.
 *
 * <p>{@link #ensureModelPresent} performs blocking network I/O; call it off the main thread (or use
 * {@link MoonshineDownloadWorker} to run it under WorkManager with network constraints and retry).
 */
public final class AssetDownloader {

    /** CDN root for TTS / G2P canonical asset keys. STT/embedding manifests embed absolute URLs. */
    private static final String TTS_CDN_BASE = "https://download.moonshine.ai/tts/";
    private static final String PART_SUFFIX = ".part";
    private static final long PROGRESS_CHUNK_BYTES = 256 * 1024;
    private static final long SPACE_MARGIN_BYTES = 8L * 1024 * 1024;

    /** Progress callback invoked (possibly off the caller's thread) as files download. */
    public interface ProgressListener {
        /**
         * @param relativePath path of the current file relative to the root (e.g.
         *                     {@code encoder_model.ort} or {@code en_us/dict.tsv}).
         * @param fileIndex 1-based index of the file being downloaded in this run.
         * @param totalFiles total number of files that will be downloaded in this run.
         * @param bytesDownloaded bytes written for the current file so far.
         * @param bytesTotal total bytes for the current file, or {@code -1} if unknown.
         */
        void onProgress(String relativePath, int fileIndex, int totalFiles,
                        long bytesDownloaded, long bytesTotal);
    }

    private final OkHttpClient client;

    public AssetDownloader() {
        this(new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .callTimeout(0, TimeUnit.SECONDS)
                .build());
    }

    /** Use a caller-provided client (e.g. to inject a mock server in tests or share a pool). */
    public AssetDownloader(OkHttpClient client) {
        this.client = client;
    }

    /** Returns true when every file required by {@code spec} already exists under {@code root}. */
    public boolean isModelPresent(File root, ModelSpec spec) {
        try {
            List<ResolvedFile> files = resolveFiles(root, spec);
            if (files.isEmpty()) {
                return false;
            }
            for (ResolvedFile file : files) {
                if (!new File(root, file.relativePath).exists()) {
                    return false;
                }
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Ensures every file required by {@code spec} is present under {@code root}, downloading any
     * that are missing, and returns {@code root} (the directory to pass to the engine loader).
     *
     * @throws IOException on network / filesystem failures.
     */
    public File ensureModelPresent(File root, ModelSpec spec, @Nullable ProgressListener listener)
            throws IOException {
        List<ResolvedFile> files = resolveFiles(root, spec);
        List<ResolvedFile> missing = new ArrayList<>();
        for (ResolvedFile file : files) {
            if (!new File(root, file.relativePath).exists()) {
                missing.add(file);
            }
        }
        for (int i = 0; i < missing.size(); i++) {
            if (Thread.currentThread().isInterrupted()) {
                throw new IOException("Download interrupted");
            }
            downloadOne(missing.get(i), root, i + 1, missing.size(), listener);
        }
        return root;
    }

    // -- Manifest resolution -------------------------------------------------

    private static final class ResolvedFile {
        final String url;
        final String relativePath;

        ResolvedFile(String url, String relativePath) {
            this.url = url;
            this.relativePath = relativePath;
        }
    }

    private List<ResolvedFile> resolveFiles(File root, ModelSpec spec) throws IOException {
        switch (spec.type) {
            case STT: {
                String json = Transcriber.getSttDependencies(spec.primary, spec.toOptions(null));
                return filesFromGroupManifest(json);
            }
            case INTENT: {
                String json = IntentRecognizer.getIntentDependencies(
                        spec.primary, spec.toOptions(null));
                return filesFromGroupManifest(json);
            }
            case TTS: {
                String json = TextToSpeech.getTtsDependencies(
                        spec.primary, spec.toOptions(root.getAbsolutePath()));
                return filesFromKeyArray(json);
            }
            case G2P: {
                String csv = TextToSpeech.getG2pDependencies(
                        spec.primary, spec.toOptions(root.getAbsolutePath()));
                List<String> keys = new ArrayList<>();
                for (String key : csv.split(",")) {
                    keys.add(key);
                }
                return filesFromKeyList(keys);
            }
            default:
                throw new IOException("Unsupported model spec type: " + spec.type);
        }
    }

    /** Parses the {@code {"groups":[{"base_url":..,"files":[..]}]}} STT/intent manifest. */
    private List<ResolvedFile> filesFromGroupManifest(String json) throws IOException {
        List<ResolvedFile> result = new ArrayList<>();
        try {
            JSONObject object = new JSONObject(json);
            JSONArray groups = object.getJSONArray("groups");
            for (int g = 0; g < groups.length(); g++) {
                JSONObject group = groups.getJSONObject(g);
                String baseUrl = group.getString("base_url");
                JSONArray fileArray = group.getJSONArray("files");
                for (int f = 0; f < fileArray.length(); f++) {
                    String file = fileArray.getString(f);
                    result.add(new ResolvedFile(baseUrl + "/" + file, file));
                }
            }
        } catch (Exception e) {
            throw new IOException("Invalid download manifest: " + json, e);
        }
        return result;
    }

    /** Parses the flat JSON array of canonical keys emitted by the TTS dependency API. */
    private List<ResolvedFile> filesFromKeyArray(String json) throws IOException {
        try {
            JSONArray array = new JSONArray(json);
            List<String> keys = new ArrayList<>(array.length());
            for (int i = 0; i < array.length(); i++) {
                keys.add(array.optString(i, ""));
            }
            return filesFromKeyList(keys);
        } catch (Exception e) {
            throw new IOException("Invalid TTS manifest: " + json, e);
        }
    }

    private List<ResolvedFile> filesFromKeyList(List<String> keys) {
        List<ResolvedFile> result = new ArrayList<>();
        for (String rawKey : keys) {
            String key = rawKey.trim();
            if (key.isEmpty() || !key.contains("/")) {
                continue;
            }
            result.add(new ResolvedFile(TTS_CDN_BASE + encodeKey(key), key));
        }
        return result;
    }

    /** Percent-encode each path segment; the library expects canonical {@code /}-joined keys. */
    private static String encodeKey(String key) {
        String[] segments = key.split("/", -1);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < segments.length; i++) {
            if (i > 0) {
                builder.append('/');
            }
            try {
                builder.append(URLEncoder.encode(segments[i], "UTF-8").replace("+", "%20"));
            } catch (IOException e) {
                builder.append(segments[i]);
            }
        }
        return builder.toString();
    }

    // -- Single-file download ------------------------------------------------

    private void downloadOne(ResolvedFile file, File root, int fileIndex, int totalFiles,
                             @Nullable ProgressListener listener) throws IOException {
        File destination = new File(root, file.relativePath);
        File directory = destination.getParentFile();
        if (directory != null && !directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create directory: " + directory);
        }
        File partFile = new File(destination.getAbsolutePath() + PART_SUFFIX);

        long existingBytes = partFile.exists() ? partFile.length() : 0;

        Request.Builder requestBuilder = new Request.Builder().url(file.url);
        if (existingBytes > 0) {
            requestBuilder.header("Range", "bytes=" + existingBytes + "-");
        }

        try (Response response = client.newCall(requestBuilder.build()).execute()) {
            int code = response.code();
            if (code < 200 || code > 299) {
                throw new IOException("HTTP " + code + " fetching " + file.url);
            }
            // 206 => server honored our Range; anything else means start over.
            boolean resuming = existingBytes > 0 && code == 206;
            if (!resuming) {
                existingBytes = 0;
                // Truncate any stale partial content.
                if (partFile.exists() && !partFile.delete()) {
                    throw new IOException("Failed to reset partial file: " + partFile);
                }
            }

            ResponseBody body = response.body();
            if (body == null) {
                throw new IOException("Empty response body for " + file.url);
            }
            long remaining = body.contentLength();  // -1 if unknown
            long totalBytes = remaining >= 0 ? existingBytes + remaining : -1;

            ensureSpaceAvailable(root, remaining, file.url);

            try (RandomAccessFile out = new RandomAccessFile(partFile, "rw");
                 InputStream input = body.byteStream()) {
                out.seek(existingBytes);

                if (listener != null) {
                    listener.onProgress(
                            file.relativePath, fileIndex, totalFiles, existingBytes, totalBytes);
                }

                byte[] buffer = new byte[64 * 1024];
                long downloaded = existingBytes;
                long lastReported = existingBytes;
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    if (Thread.currentThread().isInterrupted()) {
                        throw new IOException("Download interrupted");
                    }
                    out.write(buffer, 0, read);
                    downloaded += read;
                    if (listener != null && downloaded - lastReported >= PROGRESS_CHUNK_BYTES) {
                        listener.onProgress(
                                file.relativePath, fileIndex, totalFiles, downloaded, totalBytes);
                        lastReported = downloaded;
                    }
                }
                if (listener != null) {
                    listener.onProgress(
                            file.relativePath, fileIndex, totalFiles, downloaded, totalBytes);
                }
            }
        }

        if (destination.exists() && !destination.delete()) {
            throw new IOException("Failed to replace existing file: " + destination);
        }
        if (!partFile.renameTo(destination)) {
            partFile.delete();
            throw new IOException("Failed to move download into place: " + destination);
        }
    }

    /** Best-effort free-space precheck before writing a file whose size the server reported. */
    private void ensureSpaceAvailable(File root, long needBytes, String url) throws IOException {
        if (needBytes <= 0) {
            return;
        }
        long available = root.getUsableSpace();
        if (available > 0 && available < needBytes + SPACE_MARGIN_BYTES) {
            throw new IOException(
                    "Not enough disk space for " + url + ": need " + needBytes
                            + " bytes, " + available + " available");
        }
    }
}
