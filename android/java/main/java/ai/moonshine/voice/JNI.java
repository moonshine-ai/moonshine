package ai.moonshine.voice;

public class JNI {
    public static final int MOONSHINE_ERROR_NONE = 0;
    public static final int MOONSHINE_ERROR_UNKNOWN = -1;
    public static final int MOONSHINE_ERROR_INVALID_HANDLE = -2;
    public static final int MOONSHINE_ERROR_INVALID_ARGUMENT = -3;

    public static final int MOONSHINE_MODEL_ARCH_TINY = 0;
    public static final int MOONSHINE_MODEL_ARCH_BASE = 1;
    public static final int MOONSHINE_MODEL_ARCH_TINY_STREAMING = 2;
    public static final int MOONSHINE_MODEL_ARCH_BASE_STREAMING = 3;
    public static final int MOONSHINE_MODEL_ARCH_SMALL_STREAMING = 4;
    public static final int MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING = 5;

    public static final int MOONSHINE_FLAG_FORCE_UPDATE = 1 << 0;
    /**
     * Run the alphanumeric spelling-fusion path on completed lines. Requires
     * the transcriber to have been built with a {@code spelling_model_path}
     * option (or a non-null spelling model byte array passed to
     * {@link #moonshineLoadTranscriberFromMemory}); without one, this flag
     * is a no-op.
     */
    public static final int MOONSHINE_FLAG_SPELLING_MODE = 1 << 1;

    /** Embedding model architecture (Gemma 300M). */
    static final int MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M = 0;

    /** Pass to TTS/G2P create calls; must match native {@code moonshine-c-api.h}. */
    public static final int MOONSHINE_HEADER_VERSION = 30000;

    public static native int moonshineGetVersion();

    public static native String moonshineErrorToString(int error);

    public static native String moonshineTranscriptToString(Transcript transcript);

    public static native int moonshineLoadTranscriberFromFiles(String path, int model_arch, Object[] options);

    /**
     * Loads a transcriber from in-memory model buffers.
     *
     * @param spelling_model_data Optional spelling-CNN {@code .ort} payload; pass
     *                            {@code null} when not using
     *                            {@link #MOONSHINE_FLAG_SPELLING_MODE}. When provided,
     *                            the buffer is referenced (not copied) for the
     *                            transcriber's lifetime.
     */
    public static native int moonshineLoadTranscriberFromMemory(byte[] encoder_model_data, byte[] decoder_model_data,
            byte[] tokenizer_data, byte[] spelling_model_data, int model_arch, Object[] options);

    /**
     * Loads a transcriber from in-memory model buffers keyed by their canonical
     * filename (matching {@link #moonshineGetSttDependencies}). This is the
     * general in-memory loader: it supports every architecture (including
     * streaming), the word-timestamp decoders, and the spelling model
     * ({@code spelling_cnn.ort}). {@code filenames} and {@code memory} must be
     * the same length; each buffer is copied and kept alive for the
     * transcriber's lifetime.
     */
    public static native int moonshineLoadTranscriberFromMemoryFiles(String[] filenames, byte[][] memory,
            int model_arch, Object[] options);

    public static native void moonshineFreeTranscriber(int transcriber_handle);

    public static native Transcript moonshineTranscribeWithoutStreaming(int transcriber_handle,
            float[] audio_data,
            int sample_rate, int flags);

    public static native int moonshineCreateStream(int transcriber_handle, int flags);

    public static native int moonshineFreeStream(int transcriber_handle, int stream_handle);

    public static native int moonshineStartStream(int transcriber_handle, int stream_handle);

    public static native int moonshineStopStream(int transcriber_handle, int stream_handle);

    public static native int moonshineAddAudioToStream(int transcriber_handle,
            int stream_handle,
            float[] audio_data,
            int sample_rate,
            int flags);

    public static native Transcript moonshineTranscribeStream(int transcriber_handle,
            int stream_handle, int flags);

    // Text embeddings back AgentFlow's phrase matching and are not part of the
    // library's public surface, so these stay package-private.

    static native int moonshineCreateEmbeddingModel(String model_path,
            int embedding_model_arch, String model_variant);

    static native void moonshineFreeEmbeddingModel(int embedding_model_handle);

    /** Returns null on failure. */
    static native float[] moonshineCalculateEmbedding(int embedding_model_handle,
            String sentence);

    /**
     * Cosine similarity of two equal-length embeddings, in {@code [-1, 1]}.
     * Returns 0 when the arrays are null, empty, or of differing lengths.
     */
    static native float moonshineCalculateEmbeddingDistance(int embedding_model_handle,
            float[] embedding_a, float[] embedding_b);

    public static native int moonshineCreateTtsSynthesizerFromFiles(String language,
            String[] filenames, TranscriberOption[] options);

    public static native int moonshineCreateTtsSynthesizerFromMemory(String language,
            String[] filenames, byte[][] memory, TranscriberOption[] options);

    public static native void moonshineFreeTtsSynthesizer(int tts_synthesizer_handle);

    public static native String moonshineGetG2pDependencies(String languages,
            TranscriberOption[] options);

    public static native String moonshineGetTtsDependencies(String languages,
            TranscriberOption[] options);

    /**
     * Returns the speech-to-text model download manifest as a JSON object string,
     * or {@code null} on failure. Shape:
     * {@code {"groups":[{"base_url":"...","files":["a","b",...]}]}}. Download each
     * file from {@code base_url + "/" + file}.
     *
     * @param language Language code (e.g. {@code "en"}) or English name; must not
     *                 be empty.
     * @param options  Optional options; recognizes {@code model_arch} (decimal
     *                 string of a {@code MOONSHINE_MODEL_ARCH_*} value) and
     *                 {@code include_spelling} (bool).
     */
    public static native String moonshineGetSttDependencies(String language,
            TranscriberOption[] options);

    /**
     * Returns the embedding model download manifest as a JSON object string
     * (same shape as {@link #moonshineGetSttDependencies}), or {@code null} on
     * failure.
     *
     * @param modelName Embedding model id (e.g. {@code "embeddinggemma-300m"}), or
     *                  {@code null} for the default model.
     * @param options   Optional options; recognizes {@code variant}.
     */
    static native String moonshineGetEmbeddingDependencies(String modelName,
            TranscriberOption[] options);

    /**
     * Returns the speaker diarization download manifest as a JSON object string
     * (same shape as {@link #moonshineGetSttDependencies}), or {@code null} on
     * failure. There is one set of models and it takes no options.
     */
    static native String moonshineGetDiarizationDependencies();

    public static native String moonshineGetTtsVoices(String languages,
            TranscriberOption[] options);

    /**
     * Finds the best short window of speech in a recording, for voice cloning
     * (see {@code moonshine_extract_speech_clip}). Returns a {@link SpeechClip}
     * whose {@code audio} is null until enough speech has been heard, which is
     * how incremental capture knows to keep listening.
     */
    public static native SpeechClip moonshineExtractSpeechClip(float[] audioData, int sampleRate,
            float clipDurationSeconds, float minimumSpeechSeconds);

    public static native TtsSynthesisResult moonshineTextToSpeech(int tts_synthesizer_handle,
            String text, TranscriberOption[] options);

    public static native TtsSynthesisResult moonshinePhonemesToSpeech(int tts_synthesizer_handle,
            String phonemes, TranscriberOption[] options);

    public static native int moonshineCreateGraphemeToPhonemizerFromFiles(String language,
            String[] filenames, TranscriberOption[] options);

    public static native int moonshineCreateGraphemeToPhonemizerFromMemory(String language,
            String[] filenames, byte[][] memory, TranscriberOption[] options);

    public static native void moonshineFreeGraphemeToPhonemizer(
            int grapheme_to_phonemizer_handle);

    public static native String moonshineTextToPhonemes(int grapheme_to_phonemizer_handle,
            String text, TranscriberOption[] options);

    static boolean isLibraryLoaded = false;

    public static void ensureLibraryLoaded() {
        if (isLibraryLoaded) {
            return;
        }
        try {
            System.loadLibrary("moonshine-jni");
            isLibraryLoaded = true;
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Failed to load moonshine-jni library", e);
        }
    }
}
