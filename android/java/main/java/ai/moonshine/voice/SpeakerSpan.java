package ai.moonshine.voice;

/**
 * One contiguous span of speech within a line attributed to a single speaker.
 * Only populated when the identify_speakers option is enabled.
 *
 * Spans are mutable: the diarization algorithm re-clusters the audio history
 * as more speech arrives, so the spans of any line - including completed
 * ones - can be revised on any transcription update. Watch
 * TranscriptLine.haveSpeakersChanged to detect revisions.
 */
public class SpeakerSpan {
    /** Time offset from the start of the audio or stream in seconds. */
    public float startTime;
    /** Length of the span in seconds. */
    public float duration;
    /** Stable identifier for the speaker within this stream. */
    public long speakerId;
    /** The order the speaker first appeared in the transcript, starting at 0. */
    public int speakerIndex;
    /** UTF-8 byte offset into the line text where this span begins (inclusive). */
    public long startChar;
    /** UTF-8 byte offset into the line text where this span ends (exclusive). */
    public long endChar;

    public String toString() {
        return "SpeakerSpan(startTime=" + startTime + ", duration=" + duration
                + ", speakerId=" + speakerId + ", speakerIndex=" + speakerIndex
                + ", startChar=" + startChar + ", endChar=" + endChar + ")";
    }
}
