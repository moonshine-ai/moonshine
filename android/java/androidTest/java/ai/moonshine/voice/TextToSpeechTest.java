package ai.moonshine.voice;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * ZipVoice TTS coverage for the Android JNI binding.
 *
 * <p>The catalog / engine-selection tests need no model assets and always run. The synthesis tests
 * require the ZipVoice ONNX bundle under a {@code g2p_root} tree with a {@code zipvoice/} subdir; they
 * skip cleanly (via assumptions) when it is not bundled with the test APK, since the models are ~250 MB.
 */
public class TextToSpeechTest {
    @Before
    public void setUp() {
        JNI.ensureLibraryLoaded();
    }

    @Test
    public void testZipVoiceDependencies() {
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("voice", "zipvoice_american_female"));
        String json = TextToSpeech.getTtsDependencies("en_us", opts);
        assertTrue(json != null);
        assertTrue("expected zipvoice text encoder key",
                json.contains("zipvoice/text_encoder.ort"));
        assertTrue("expected zipvoice fm decoder key",
                json.contains("zipvoice/fm_decoder.ort"));
        assertTrue("expected zipvoice vocoder key", json.contains("zipvoice/vocoder.ort"));
        assertTrue("expected zipvoice tokens key", json.contains("zipvoice/tokens.txt"));
        assertFalse("should not mix in kokoro assets", json.contains("kokoro/model.onnx"));
        assertFalse("should not mix in piper assets", json.contains("piper-voices"));
    }

    @Test
    public void testZipVoiceVoicesListing() {
        List<TranscriberOption> opts = new ArrayList<>();
        opts.add(new TranscriberOption("voice", "zipvoice_american_female"));
        opts.add(new TranscriberOption("g2p_root", "/data/local/tmp"));
        String json = TextToSpeech.getTtsVoices("en_us", opts);
        assertTrue(json != null);
        assertTrue("expected built-in ZipVoice voice id",
                json.contains("zipvoice_american_female"));
        assertTrue("expected a second built-in ZipVoice voice id",
                json.contains("zipvoice_indian_male"));
        assertFalse("zipvoice engine should not list kokoro voices", json.contains("kokoro_"));
    }

    /** Resolves a bundled ZipVoice model tree, or null when it is not present in the test APK. */
    private static String findZipVoiceRoot() {
        Context ctx = InstrumentationRegistry.getInstrumentation().getContext();
        try {
            String[] assets = ctx.getAssets().list("tts-data/zipvoice");
            if (assets == null || assets.length == 0) {
                return null;
            }
        } catch (IOException e) {
            return null;
        }
        Path tempDir;
        try {
            tempDir = Files.createTempDirectory("moonshine-zipvoice-test");
        } catch (IOException e) {
            return null;
        }
        String[] files = {
            "zipvoice/text_encoder.ort", "zipvoice/fm_decoder.ort", "zipvoice/vocoder.ort",
            "zipvoice/tokens.txt", "zipvoice/model.json",
            "en_us/dict_filtered_heteronyms.tsv", "en_us/g2p-config.json",
        };
        for (String f : files) {
            try {
                Utils.copyAssetToTempDir(ctx, tempDir, "tts-data/" + f);
            } catch (RuntimeException ignored) {
                // Optional G2P files may be absent in a minimal bundle.
            }
        }
        File root = new File(tempDir.toFile(), "tts-data");
        File zv = new File(root, "zipvoice/text_encoder.ort");
        return zv.exists() ? root.getAbsolutePath() : null;
    }

    @Test
    public void testZipVoiceBuiltinVoiceSynthesizes() {
        String root = findZipVoiceRoot();
        org.junit.Assume.assumeTrue("ZipVoice model bundle not present in test assets", root != null);
        TextToSpeech tts = new TextToSpeech("en_us", root,
                java.util.Collections.singletonList(
                        new TranscriberOption("voice", "zipvoice_american_female")));
        try {
            TtsSynthesisResult result = tts.synthesize("Hello from ZipVoice on Android.");
            assertTrue(result != null);
            assertTrue(result.samples.length > 0);
            assertTrue(result.sampleRateHz == 24000);
        } finally {
            tts.close();
        }
    }

    @Test
    public void testZipVoiceClonePcmSynthesizes() {
        String root = findZipVoiceRoot();
        org.junit.Assume.assumeTrue("ZipVoice model bundle not present in test assets", root != null);
        float[] pcm = new float[24000];
        for (int i = 0; i < pcm.length; i++) {
            pcm[i] = (float) (0.05 * Math.sin(2.0 * Math.PI * 150.0 * i / 24000.0));
        }
        TextToSpeech tts = TextToSpeech.fromZipVoiceClone("en_us", pcm, 24000,
                "This is a reference clip.", root, null);
        try {
            TtsSynthesisResult result = tts.synthesize("Cloning a custom voice.");
            assertTrue(result != null);
            assertTrue(result.sampleRateHz == 24000);
        } finally {
            tts.close();
        }
    }
}
