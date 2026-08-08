package ai.moonshine.voice;

import java.util.List;

public class TranscriptLine {
    public String text;
    public float[] audioData;
    public float startTime;
    public float duration;
    public long id;
    public boolean isComplete;
    public boolean isUpdated;
    public boolean isNew;
    public boolean hasTextChanged;
    /**
     * Whether the speaker spans of the line have changed since the previous
     * call. Unlike the other change flags, this can fire for lines that are
     * already complete, since diarization refines speaker assignments
     * retroactively as more audio arrives.
     */
    public boolean haveSpeakersChanged;
    /**
     * Speaker spans covering this line, ordered by start time and clipped to
     * the line's time range. Null unless the identify_speakers option is
     * enabled and speech has been attributed to a speaker.
     */
    public List<SpeakerSpan> speakerSpans;
    public int lastTranscriptionLatencyMs;
    public List<WordTiming> words;

    public String toString() {
        return "TranscriptLine(text=" + text + ", audioData.length=" + audioData.length + ", startTime=" + startTime
                + ", duration=" + duration + ", id=" + id + ", isComplete=" + isComplete + ", isUpdated=" + isUpdated
                + ", isNew=" + isNew + ", hasTextChanged=" + hasTextChanged + ", haveSpeakersChanged="
                + haveSpeakersChanged + ", speakerSpans=" + (speakerSpans != null ? speakerSpans.size() : 0)
                + ", lastTranscriptionLatencyMs=" + lastTranscriptionLatencyMs + ", words="
                + (words != null ? words.size() : 0) + ")";
    }
}
