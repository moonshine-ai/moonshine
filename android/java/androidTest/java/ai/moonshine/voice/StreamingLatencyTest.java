package ai.moonshine.voice;

import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.util.Log;

import androidx.test.InstrumentationRegistry;
import androidx.test.filters.LargeTest;

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
        tempDir = Files.createTempDirectory("voice-streaming-latency");
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

        Transcriber transcriber = new Transcriber();
        transcriber.loadFromFiles(root.getAbsolutePath() + "/", modelArch);
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
                "MOONSHINE_LATENCY platform=android device=%s model=%s avg_ms=%.0f lines=%d wall_s=%.2f",
                device, modelName, avgMs, latencies.size(), wallSeconds);
        Log.i(TAG, summary);
        System.out.println(summary);

        assertTrue(
                String.format(Locale.US,
                        "%s avg latency %.0fms exceeds regression ceiling %dms",
                        modelName, avgMs, maxAvgLatencyMs),
                avgMs <= maxAvgLatencyMs);
    }
}
