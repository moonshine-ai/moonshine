package ai.moonshine.voice;

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

    public String toString() {
        return "TranscriptLine(text=" + text + ", audioData.length=" + audioData.length + ", startTime=" + startTime
                + ", duration=" + duration + ", id=" + id + ", isComplete=" + isComplete + ", isUpdated=" + isUpdated
                + ", isNew=" + isNew + ", hasTextChanged=" + hasTextChanged + ")";
    }
}
