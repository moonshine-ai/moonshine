package ai.moonshine.voice;

import java.util.List;

/**
 * Turns text into embedding vectors and scores them against each other.
 *
 * <p>Internal to the library. {@link AgentFlow} is the supported way to match
 * spoken phrases; it owns a model and compares utterances to phrases itself.
 */
class EmbeddingModel {
  private int handle = -1;

  /**
   * @param modelRootDir directory containing the embedding ONNX bundle
   * @param embeddingModelArch e.g. {@link JNI#MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M}
   * @param modelVariant e.g. {@code "q4"}; pass null for native default
   */
  EmbeddingModel(String modelRootDir, int embeddingModelArch, String modelVariant) {
    JNI.ensureLibraryLoaded();
    String variant = modelVariant != null ? modelVariant : "q4";
    this.handle = JNI.moonshineCreateEmbeddingModel(modelRootDir, embeddingModelArch, variant);
    if (this.handle < 0) {
      throw new RuntimeException("Failed to create embedding model from path: " + modelRootDir);
    }
  }

  EmbeddingModel(String modelRootDir, int embeddingModelArch) {
    this(modelRootDir, embeddingModelArch, "q4");
  }

  /**
   * Returns the embedding model download manifest as a JSON object string (see
   * {@code moonshine_get_embedding_dependencies}). Shape:
   * {@code {"groups":[{"base_url":"...","files":["a","b",...]}]}}. Use with
   * {@link AssetDownloader} to fetch the files, then construct an
   * {@link EmbeddingModel} pointing at the download directory.
   *
   * @param modelName Embedding model id (e.g. {@code "embeddinggemma-300m"}), or
   *                  {@code null} for the default model.
   * @param options   Optional options; recognizes {@code variant}.
   */
  static String getEmbeddingDependencies(String modelName, List<TranscriberOption> options) {
    JNI.ensureLibraryLoaded();
    TranscriberOption[] optionsArray =
        (options == null || options.isEmpty())
            ? null
            : options.toArray(new TranscriberOption[0]);
    String json = JNI.moonshineGetEmbeddingDependencies(modelName, optionsArray);
    if (json == null) {
      throw new RuntimeException("moonshineGetEmbeddingDependencies failed");
    }
    return json;
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  void close() {
    if (handle >= 0) {
      JNI.moonshineFreeEmbeddingModel(handle);
      handle = -1;
    }
  }

  /** The embedding vector for {@code sentence}. */
  float[] calculateEmbedding(String sentence) {
    checkHandle();
    float[] embedding = JNI.moonshineCalculateEmbedding(handle, sentence);
    if (embedding == null) {
      throw new RuntimeException("moonshineCalculateEmbedding failed");
    }
    return embedding;
  }

  /** Cosine similarity between two embeddings of equal length, in {@code [-1, 1]}. */
  float distance(float[] embeddingA, float[] embeddingB) {
    checkHandle();
    if (embeddingA == null || embeddingB == null || embeddingA.length != embeddingB.length) {
      throw new IllegalArgumentException("Embeddings must be non-null and the same length");
    }
    return JNI.moonshineCalculateEmbeddingDistance(handle, embeddingA, embeddingB);
  }

  private void checkHandle() {
    if (handle < 0) {
      throw new IllegalStateException("EmbeddingModel is closed");
    }
  }
}
