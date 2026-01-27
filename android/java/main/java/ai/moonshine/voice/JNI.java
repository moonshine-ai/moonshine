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

    public static native int moonshineGetVersion();

    public static native String moonshineErrorToString(int error);

    public static native String moonshineTranscriptToString(Transcript transcript);

    public static native int moonshineLoadTranscriberFromFiles(String path, int model_arch, Object[] options);

    public static native int moonshineLoadTranscriberFromMemory(byte[] encoder_model_data, byte[] decoder_model_data,
            byte[] tokenizer_data, int model_arch, Object[] options);

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
            int sample_rate);

    public static native Transcript moonshineTranscribeStream(int transcriber_handle,
            int stream_handle, int flags);

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
