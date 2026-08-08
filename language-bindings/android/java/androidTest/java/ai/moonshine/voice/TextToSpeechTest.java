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
        assertFalse("should not mix in kokoro assets", json.contains("kokoro/model.ort"));
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

    /** Real speech, so the clone path's voice-activity search has something to find. */
    private static Utils.WavData speechSamples() {
        Context ctx = InstrumentationRegistry.getInstrumentation().getContext();
        try {
            return Utils.loadWavFromAssets(ctx, "beckett.wav");
        } catch (IOException e) {
            throw new RuntimeException("beckett.wav is missing from the test assets", e);
        }
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
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        TextToSpeech tts = new TextToSpeech(context)
                .language("en_us")
                .voice("zipvoice_american_female")
                .modelsFrom(new File(root));
        try {
            tts.load();
            TtsSynthesisResult result = tts.synthesize("Hello from ZipVoice on Android.");
            assertTrue(result != null);
            assertTrue(result.samples.length > 0);
            assertTrue(result.sampleRateHz == 24000);
        } finally {
            tts.close();
        }
    }

    @Test
    public void testCloneFromPcmSynthesizes() {
        String root = findZipVoiceRoot();
        org.junit.Assume.assumeTrue("ZipVoice model bundle not present in test assets", root != null);
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Utils.WavData reference = speechSamples();
        TextToSpeech tts = new TextToSpeech(context)
                .language("en_us")
                .modelsFrom(new File(root))
                .cloning(true);
        try {
            tts.load();
            tts.cloneFrom(reference.data, reference.sampleRate, "This is a reference clip.");
            assertTrue("cloneFrom should mark the engine as cloned", tts.isCloned());
            TtsSynthesisResult result = tts.synthesize("Cloning a custom voice.");
            assertTrue(result != null);
            assertTrue(result.sampleRateHz == 24000);
        } finally {
            tts.close();
        }
    }
}
