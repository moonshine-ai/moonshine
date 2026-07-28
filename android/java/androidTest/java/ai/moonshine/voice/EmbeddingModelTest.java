package ai.moonshine.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import androidx.test.InstrumentationRegistry;

import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;

public class EmbeddingModelTest {

  @Test(expected = RuntimeException.class)
  public void testCreateEmbeddingModel_invalidPath_throws() {
    JNI.ensureLibraryLoaded();
    new EmbeddingModel("/nonexistent/moonshine/embedding/model",
        JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M);
  }

  @Test
  public void testMatchesClosestPhrase_whenModelPresent() {
    JNI.ensureLibraryLoaded();
    File filesDir = InstrumentationRegistry.getTargetContext().getFilesDir();
    File modelDir = new File(filesDir, "embeddinggemma-300m-ONNX");
    if (!modelDir.isDirectory()) {
      return;
    }
    EmbeddingModel model = new EmbeddingModel(modelDir.getAbsolutePath(),
        JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M);
    try {
      float[] lights = model.calculateEmbedding("turn on the lights");
      assertTrue(lights.length > 0);
      float[] garage = model.calculateEmbedding("close the garage door");
      assertTrue(model.distance(lights, lights) > model.distance(lights, garage));

      PhraseMatcher matcher = new PhraseMatcher(model);
      String[] phrases = {"turn on the lights", "close the garage door"};
      assertEquals("turn on the lights",
          matcher.match("switch the lights on", phrases, 0.5f));
    } finally {
      model.close();
    }
  }

  @Test
  public void testPhraseMatcherFallsBackToSubstrings() {
    PhraseMatcher matcher = new PhraseMatcher(null);
    assertEquals("weather", matcher.match("what's the weather like",
        Collections.singletonList(new PhraseMatcher.Group("weather",
            Arrays.asList("the weather", "forecast"))), 0.7f));
    assertNull(matcher.match("play some music",
        Collections.singletonList(new PhraseMatcher.Group("weather",
            Arrays.asList("the weather", "forecast"))), 0.7f));
  }
}
