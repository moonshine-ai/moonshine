package ai.moonshine.voice;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Semantic intent recognizer: registers canonical phrases and ranks them against
 * an utterance via {@link JNI#moonshineGetClosestIntents}.
 */
public class IntentRecognizer {
  private int handle = -1;

  /**
   * @param modelRootDir directory containing the embedding ONNX bundle
   * @param embeddingModelArch e.g. {@link JNI#MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M}
   * @param modelVariant e.g. {@code "q4"}; pass null for native default
   */
  public IntentRecognizer(String modelRootDir, int embeddingModelArch, String modelVariant) {
    JNI.ensureLibraryLoaded();
    String variant = modelVariant != null ? modelVariant : "q4";
    this.handle = JNI.moonshineCreateIntentRecognizer(modelRootDir, embeddingModelArch, variant);
    if (this.handle < 0) {
      throw new RuntimeException("Failed to create intent recognizer from path: " + modelRootDir);
    }
  }

  public IntentRecognizer(String modelRootDir, int embeddingModelArch) {
    this(modelRootDir, embeddingModelArch, "q4");
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  public void close() {
    if (handle >= 0) {
      JNI.moonshineFreeIntentRecognizer(handle);
      handle = -1;
    }
  }

  public void registerIntent(String canonicalPhrase) {
    checkHandle();
    int err = JNI.moonshineRegisterIntent(handle, canonicalPhrase);
    if (err != JNI.MOONSHINE_ERROR_NONE) {
      throw new RuntimeException("moonshineRegisterIntent failed: " + err);
    }
  }

  /** @return true if the phrase was removed */
  public boolean unregisterIntent(String canonicalPhrase) {
    checkHandle();
    int err = JNI.moonshineUnregisterIntent(handle, canonicalPhrase);
    if (err == JNI.MOONSHINE_ERROR_NONE) {
      return true;
    }
    if (err == JNI.MOONSHINE_ERROR_INVALID_ARGUMENT) {
      return false;
    }
    throw new RuntimeException("moonshineUnregisterIntent failed: " + err);
  }

  public List<IntentMatch> getClosestIntents(String utterance, float toleranceThreshold) {
    checkHandle();
    IntentMatch[] arr =
        JNI.moonshineGetClosestIntents(handle, utterance, toleranceThreshold);
    if (arr == null) {
      throw new RuntimeException("moonshineGetClosestIntents failed");
    }
    return Collections.unmodifiableList(Arrays.asList(arr));
  }

  public int getIntentCount() {
    checkHandle();
    int n = JNI.moonshineGetIntentCount(handle);
    if (n < 0) {
      throw new RuntimeException("moonshineGetIntentCount failed: " + n);
    }
    return n;
  }

  public void clearIntents() {
    checkHandle();
    int err = JNI.moonshineClearIntents(handle);
    if (err != JNI.MOONSHINE_ERROR_NONE) {
      throw new RuntimeException("moonshineClearIntents failed: " + err);
    }
  }

  private void checkHandle() {
    if (handle < 0) {
      throw new IllegalStateException("IntentRecognizer is closed");
    }
  }
}
