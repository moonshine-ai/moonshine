package ai.moonshine.voice;

/** PCM float samples (~-1..1) and sample rate from {@link TextToSpeech#synthesize}. */
public class TtsSynthesisResult {
    public float[] samples;
    public int sampleRateHz;

    public TtsSynthesisResult() {}

    public TtsSynthesisResult(float[] samples, int sampleRateHz) {
        this.samples = samples;
        this.sampleRateHz = sampleRateHz;
    }
}
