package ai.moonshine.voice;

import androidx.annotation.Nullable;

/**
 * A short window of mostly-speech audio pulled out of a longer recording, as
 * used for zero-shot voice cloning.
 *
 * <p>{@link #audio} is null until the detector has heard enough speech, so a
 * caller recording incrementally can keep feeding audio in and use
 * {@link #speechDuration} to show progress in the meantime.
 */
public final class SpeechClip {

    /** 16 kHz mono PCM, or null when {@link #isComplete} is false. */
    @Nullable
    public final float[] audio;

    /** Where the window starts in the input recording, in seconds. */
    public final float startTime;

    /** How much of the window is speech, in seconds. */
    public final float speechDuration;

    /** True once a window with enough speech in it was found. */
    public final boolean isComplete;

    /** Transcript when the TTS owns clone ASR and refine ran; null otherwise. */
    @Nullable
    public final String transcript;

    public SpeechClip(@Nullable float[] audio, float startTime, float speechDuration,
            boolean isComplete, @Nullable String transcript) {
        this.audio = audio;
        this.startTime = startTime;
        this.speechDuration = speechDuration;
        this.isComplete = isComplete;
        this.transcript = transcript;
    }

    /** Sample rate of {@link #audio}. */
    public static final int SAMPLE_RATE = 16000;
}
