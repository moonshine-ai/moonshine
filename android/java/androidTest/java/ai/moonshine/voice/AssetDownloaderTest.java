package ai.moonshine.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.os.Bundle;

import androidx.test.InstrumentationRegistry;

import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

/**
 * End-to-end tests that exercise {@link AssetDownloader} against the <b>real</b> CDN
 * (https://download.moonshine.ai): each downloads a model into an empty directory and then loads
 * and runs the matching engine, proving the whole download-then-load path works on a device, not
 * just manifest parsing.
 *
 * <p>These require a network connection and pull tens to hundreds of MB, so they are opt-in: they
 * run only when the instrumentation argument {@code moonshineDownloadTests} is set to a truthy
 * value ({@code 1}, {@code true}, or {@code yes}), e.g.
 * {@code ./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.moonshineDownloadTests=1}.
 * Without it they skip so the default instrumentation suite stays offline.
 */
public class AssetDownloaderTest {

    private Path tempDir;

    @Before
    public void setUp() throws IOException {
        JNI.ensureLibraryLoaded();
        Assume.assumeTrue(
                "Set -Pandroid.testInstrumentationRunnerArguments.moonshineDownloadTests=1 to run "
                        + "network download tests against the CDN",
                downloadsEnabled());
        tempDir = Files.createTempDirectory("moonshine-download-network-test");
    }

    /** True when the network download tests have been explicitly enabled via instrumentation args. */
    private static boolean downloadsEnabled() {
        Bundle args = InstrumentationRegistry.getArguments();
        if (args == null) {
            return false;
        }
        String value = args.getString("moonshineDownloadTests");
        return value != null
                && (value.equals("1") || value.equalsIgnoreCase("true")
                        || value.equalsIgnoreCase("yes"));
    }

    @Test
    public void testDownloadsAndRunsSttModel() throws IOException {
        Context ctx = InstrumentationRegistry.getInstrumentation().getContext();
        Utils.WavData wav = Utils.loadWavFromAssets(ctx, "two_cities.wav");
        assertTrue(wav.data != null && wav.data.length > 0);

        File root = tempDir.toFile();
        AssetDownloader downloader = new AssetDownloader();
        ModelSpec spec = ModelSpec.stt("en", JNI.MOONSHINE_MODEL_ARCH_TINY, false);

        assertFalse(downloader.isModelPresent(root, spec));
        downloader.ensureModelPresent(root, spec, null);
        assertTrue("every STT file should be present after download",
                downloader.isModelPresent(root, spec));

        Transcriber transcriber = new Transcriber();
        transcriber.loadFromFiles(root.getAbsolutePath() + "/", JNI.MOONSHINE_MODEL_ARCH_TINY);

        Transcript transcript = transcriber.transcribeWithoutStreaming(wav.data, wav.sampleRate);
        assertTrue(transcript != null);
        assertTrue(transcript.lines.size() > 0);
        StringBuilder sb = new StringBuilder();
        for (TranscriptLine line : transcript.lines) {
            sb.append(line.text.toLowerCase()).append(' ');
        }
        String text = sb.toString();
        assertTrue("unexpected transcript: " + text, text.contains("best of times"));
        assertTrue("unexpected transcript: " + text, text.contains("worst of times"));
    }

    @Test
    public void testDownloadsAndRunsTtsVoice() throws IOException {
        File root = tempDir.toFile();
        AssetDownloader downloader = new AssetDownloader();
        ModelSpec spec = ModelSpec.tts("en_us", "kokoro_af_heart");

        downloader.ensureModelPresent(root, spec, null);
        assertTrue(downloader.isModelPresent(root, spec));

        TextToSpeech tts = new TextToSpeech("en_us", root.getAbsolutePath(),
                Collections.singletonList(new TranscriberOption("voice", "kokoro_af_heart")));
        try {
            TtsSynthesisResult result = tts.synthesize("Hello from the download test.");
            assertTrue(result != null);
            assertTrue("synthesis produced no audio", result.samples.length > 0);
            assertTrue(result.sampleRateHz > 0);
        } finally {
            tts.close();
        }
    }

    @Test
    public void testDownloadsAndRunsIntentModel() throws IOException {
        File root = tempDir.toFile();
        AssetDownloader downloader = new AssetDownloader();
        ModelSpec spec = ModelSpec.intent(null, "q4");

        downloader.ensureModelPresent(root, spec, null);
        assertTrue(downloader.isModelPresent(root, spec));

        IntentRecognizer recognizer = new IntentRecognizer(
                root.getAbsolutePath(), JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M);
        try {
            recognizer.registerIntent("turn on the lights");
            List<IntentMatch> matches = recognizer.getClosestIntents("turn on the lights", 0.0f);
            assertFalse(matches.isEmpty());
            assertEquals("turn on the lights", matches.get(0).canonicalPhrase);
        } finally {
            recognizer.close();
        }
    }
}
