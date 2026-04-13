package ai.moonshine.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import androidx.test.InstrumentationRegistry;

import org.junit.Test;

import java.io.File;

public class IntentRecognizerTest {

  @Test(expected = RuntimeException.class)
  public void testCreateIntentRecognizer_invalidPath_throws() {
    JNI.ensureLibraryLoaded();
    new IntentRecognizer("/nonexistent/moonshine/intent/model",
        JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M);
  }

  @Test
  public void testGetClosestIntents_whenModelPresent() {
    JNI.ensureLibraryLoaded();
    File filesDir = InstrumentationRegistry.getTargetContext().getFilesDir();
    File modelDir = new File(filesDir, "embeddinggemma-300m-ONNX");
    if (!modelDir.isDirectory()) {
      return;
    }
    IntentRecognizer r =
        new IntentRecognizer(modelDir.getAbsolutePath(), JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M);
    try {
      assertEquals(0, r.getIntentCount());
      r.registerIntent("turn on the lights");
      assertEquals(1, r.getIntentCount());
      java.util.List<IntentMatch> matches = r.getClosestIntents("turn on the lights", 0.0f);
      assertFalse(matches.isEmpty());
      assertEquals("turn on the lights", matches.get(0).canonicalPhrase);
      assertFalse(r.unregisterIntent("unknown phrase"));
      assertTrue(r.unregisterIntent("turn on the lights"));
      assertEquals(0, r.getIntentCount());
      r.clearIntents();
    } finally {
      r.close();
    }
  }
}
