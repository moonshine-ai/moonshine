package ai.moonshine.voice;

/**
 * One contiguous span of speech within a line attributed to a single speaker.
 * Only populated when the identify_speakers option is enabled.
 *
 * Spans for recent audio are mutable: streaming diarization re-clusters a
 * sliding window (diarization_cluster_window_sec, default 120s) as more
 * speech arrives. Assignments for older audio are frozen. Watch
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
