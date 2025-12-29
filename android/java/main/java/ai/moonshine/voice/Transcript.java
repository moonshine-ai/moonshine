package ai.moonshine.voice;

import java.util.List;

public class Transcript {
    public List<TranscriptLine> lines;
    public String text() {
        StringBuilder text = new StringBuilder();
        for (TranscriptLine line : lines) {
            text.append(line.text).append("\n");
        }
        return text.toString();
    }
}
