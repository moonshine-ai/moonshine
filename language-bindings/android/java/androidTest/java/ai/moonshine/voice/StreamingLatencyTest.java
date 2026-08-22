package ai.moonshine.voice;

import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.test.InstrumentationRegistry;
import androidx.test.filters.LargeTest;

import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Measures Tiny / Small / Medium Streaming end-of-phrase latency on-device with
 * the same metric as {@code core/benchmark} / the README table: average
 * {@link TranscriptLine#lastTranscriptionLatencyMs} over completed lines while
 * feeding {@code two_cities.wav} in small chunks (as fast as the device can).
 *
 * <p>Models are downloaded from the CDN via {@link AssetDownloader} (network
 * required). Parseable summary for {@code scripts/test-mobile-latency.sh}:
 * {@code MOONSHINE_LATENCY platform=android device=... model=... avg_ms=...}
 *
 * <p>Pass {@code -e keyterms "Kubernetes,Ceph"} (and optionally
 * {@code -e keyterm_boost 4.0}) to measure the same latency with contextual
 * biasing switched on, so its per-token cost can be compared against a
 * baseline run on the same device. A list of more than a few thousand terms
 * will not fit in an instrumentation argument, so those go in a file pushed to
 * the device and named with {@code -e keyterms_file /data/local/tmp/terms.txt}.
 */
@LargeTest
@RunWith(Parameterized.class)
public class StreamingLatencyTest {
    private static final String TAG = "MoonshineLatency";

    @Parameterized.Parameter(0)
    public String modelName;

    @Parameterized.Parameter(1)
    public int modelArch;

    @Parameterized.Parameter(2)
    public int maxAvgLatencyMs;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> models() {
        return Arrays.asList(new Object[][] {
                {"tiny-streaming-en", JNI.MOONSHINE_MODEL_ARCH_TINY_STREAMING, 250},
                {"small-streaming-en", JNI.MOONSHINE_MODEL_ARCH_SMALL_STREAMING, 750},
                {"medium-streaming-en", JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, 1400},
        });
    }

    private Path tempDir;

    @Before
    public void setUp() throws IOException {
        JNI.ensureLibraryLoaded();
        // Medium Streaming (~245M params) routinely OOMs / crashes the process on
        // the API 26 CI emulator used by scripts/test-android.sh. Tiny/Small still
        // exercise that path; Medium is covered on physical devices via
        // scripts/test-mobile-latency.sh.
        Assume.assumeFalse(
                "medium-streaming-en skipped on API < 28 (CI emulator OOM)",
                "medium-streaming-en".equals(modelName) && Build.VERSION.SDK_INT < 28);
        // Pixel 16 test packages can be PARTIALLY_DIRECT_BOOT_AWARE and never
        // launched, so credential-encrypted /data/user/0/<pkg> does not exist
        // (ceDataInode=0). Device-protected storage does.
        Context ctx = InstrumentationRegistry.getInstrumentation()
                .getTargetContext()
                .createDeviceProtectedStorageContext();
        File base = ctx.getFilesDir();
        if (base == null || (!base.isDirectory() && !base.mkdirs())) {
            throw new IOException("no device-protected files dir for "
                    + ctx.getPackageName());
        }
        File dir = new File(base, "voice-streaming-latency-" + System.nanoTime());
        if (!dir.mkdirs()) {
            throw new IOException("Failed to create " + dir.getAbsolutePath());
        }
        tempDir = dir.toPath();
    }

    @Test
    public void testStreamingLatencyTwoCities() throws IOException {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        Utils.WavData wavData = Utils.loadWavFromAssets(testContext, "two_cities.wav");
        assertTrue(wavData.data != null && wavData.data.length > 0);

        File root = new File(tempDir.toFile(), modelName);
        assertTrue(root.mkdirs() || root.isDirectory());
        AssetDownloader downloader = new AssetDownloader();
        ModelSpec spec = ModelSpec.stt("en", modelArch, false);
        Log.i(TAG, "downloading " + modelName + " ...");
        downloader.ensureModelPresent(root, spec, (path, fileIndex, totalFiles, got, total) -> {
            if (total > 0 && (got == total || got % (1024 * 1024) < 256 * 1024)) {
                Log.i(TAG, String.format(Locale.US, "  %s %d/%d %.0f%%",
                        path, fileIndex, totalFiles, 100.0 * got / total));
            }
        });
        assertTrue(downloader.isModelPresent(root, spec));

        // Contextual biasing is off unless the runner passes key terms, so the
        // default run stays the plain latency baseline.
        List<TranscriberOption> options = new ArrayList<>();
        android.os.Bundle arguments = InstrumentationRegistry.getArguments();
        String keyterms = arguments.getString("keyterms");
        String keytermBoost = arguments.getString("keyterm_boost");
        String keytermsFile = arguments.getString("keyterms_file");
        if (keytermsFile != null && !keytermsFile.isEmpty()) {
            keyterms = readKeytermsFile(keytermsFile);
        }
        if (keyterms != null && !keyterms.isEmpty()) {
            options.add(new TranscriberOption("keyterms", keyterms));
            if (keytermBoost != null && !keytermBoost.isEmpty()) {
                options.add(new TranscriberOption("keyterm_boost", keytermBoost));
            }
            Log.i(TAG, "key-term biasing enabled: "
                    + keyterms.split(",").length + " terms, boost "
                    + (keytermBoost == null ? "default" : keytermBoost));
        }

        Transcriber transcriber = new Transcriber(options);
        long loadStartNs = System.nanoTime();
        transcriber.loadFromFiles(root.getAbsolutePath() + "/", modelArch);
        double loadMs = (System.nanoTime() - loadStartNs) / 1e6;
        transcriber.start();

        List<Integer> latencies = new ArrayList<>();
        StringBuilder allText = new StringBuilder();
        AtomicReference<String> error = new AtomicReference<>(null);

        transcriber.addListener(event -> event.accept(new TranscriptEventListener() {
            @Override
            public void onLineCompleted(TranscriptEvent.LineCompleted e) {
                latencies.add(e.line.lastTranscriptionLatencyMs);
                allText.append(e.line.text).append("\n");
            }

            @Override
            public void onError(TranscriptEvent.Error e) {
                error.set(e.cause != null ? e.cause.getMessage() : "unknown");
            }
        }));

        final float chunkDurationSeconds = 0.0214f;
        final int chunkSize = Math.max(1, (int) (chunkDurationSeconds * wavData.sampleRate));
        long wallStartNs = System.nanoTime();
        for (int i = 0; i < wavData.data.length; i += chunkSize) {
            float[] chunk = Arrays.copyOfRange(
                    wavData.data, i, Math.min(i + chunkSize, wavData.data.length));
            transcriber.addAudio(chunk, wavData.sampleRate);
        }
        transcriber.stop();
        double wallSeconds = (System.nanoTime() - wallStartNs) / 1e9;

        assertTrue("transcription error: " + error.get(), error.get() == null);
        assertTrue("expected completed lines", !latencies.isEmpty());
        String lower = allText.toString().toLowerCase(Locale.US);
        assertTrue(lower.contains("best of times"));
        assertTrue(lower.contains("worst of times"));

        long sum = 0;
        for (int ms : latencies) {
            sum += ms;
        }
        double avgMs = sum / (double) latencies.size();
        String device = android.os.Build.MODEL.replace(' ', '_');
        String summary = String.format(Locale.US,
                "MOONSHINE_LATENCY platform=android device=%s model=%s avg_ms=%.0f load_ms=%.0f lines=%d wall_s=%.2f keyterms=%d",
                device, modelName, avgMs, loadMs, latencies.size(), wallSeconds,
                (keyterms == null || keyterms.isEmpty()) ? 0 : keyterms.split(",").length);
        Log.i(TAG, summary);
        System.out.println(summary);

        assertTrue(
                String.format(Locale.US,
                        "%s avg latency %.0fms exceeds regression ceiling %dms",
                        modelName, avgMs, maxAvgLatencyMs),
                avgMs <= maxAvgLatencyMs);
    }

    /**
     * Reads a key terms file of comma or newline separated terms into the
     * comma-separated form the option expects.
     */
    private static String readKeytermsFile(String path) {
        StringBuilder terms = new StringBuilder();
        try (java.io.BufferedReader reader =
                new java.io.BufferedReader(new java.io.FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                if (terms.length() > 0) {
                    terms.append(',');
                }
                terms.append(line);
            }
        } catch (java.io.IOException e) {
            throw new AssertionError("Failed to read key terms file " + path, e);
        }
        return terms.toString();
    }
}
