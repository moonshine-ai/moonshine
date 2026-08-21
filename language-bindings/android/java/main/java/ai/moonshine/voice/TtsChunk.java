package ai.moonshine.voice;

/** One piece of streamed audio from {@link TextToSpeech#nextChunk()}. */
public class TtsChunk {
    /** Mono PCM float samples, approximately in the range -1..1. */
    public float[] samples;
    /** Sample rate in Hz (typically 24000). */
    public int sampleRateHz;
    /** The text this chunk covers, or {@code ""} when the engine cannot attribute it. */
    public String text = "";
    /** Which queued utterance this chunk belongs to, counting from zero. */
    public long utteranceId;
    /** True for the last chunk of an utterance. */
    public boolean isFinal;

    /**
     * Native status: {@code 0} when this chunk holds audio, {@code 1} when no complete
     * utterance is buffered yet, {@code 2} once input ended and the queue drained,
     * {@code 3} once after a cancel discarded the reply, and negative for an error.
     */
    public int status;

    public TtsChunk() {}
}
