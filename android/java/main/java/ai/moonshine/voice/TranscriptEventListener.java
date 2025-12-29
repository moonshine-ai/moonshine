package ai.moonshine.voice;

public abstract class TranscriptEventListener implements TranscriptEvent.Visitor {
    @Override public void onLineStarted(TranscriptEvent.LineStarted event) {}
    @Override public void onLineUpdated(TranscriptEvent.LineUpdated event) {}
    @Override public void onLineTextChanged(TranscriptEvent.LineTextChanged event) {}
    @Override public void onLineCompleted(TranscriptEvent.LineCompleted event) {}
    @Override public void onError(TranscriptEvent.Error event) {}
}