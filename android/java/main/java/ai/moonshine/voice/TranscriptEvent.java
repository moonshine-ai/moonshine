package ai.moonshine.voice;

public interface TranscriptEvent {
    void accept(Visitor visitor);
    
    interface Visitor {
        void onLineStarted(LineStarted event);
        void onLineUpdated(LineUpdated event);
        void onLineTextChanged(LineTextChanged event);
        void onLineCompleted(LineCompleted event);
        void onError(Error event);
    }
    
    class LineStarted implements TranscriptEvent {
        public final TranscriptLine line;
        public final int streamHandle;
        
        public LineStarted(TranscriptLine line, int streamHandle) { 
            this.line = line;
            this.streamHandle = streamHandle;
        }
        
        @Override
        public void accept(Visitor v) { 
            v.onLineStarted(this); 
        }
    }
    
    class LineUpdated implements TranscriptEvent {
        public final TranscriptLine line;
        public final int streamHandle;
        
        public LineUpdated(TranscriptLine line, int streamHandle) {
            this.line = line;
            this.streamHandle = streamHandle;
        }
        
        @Override
        public void accept(Visitor v) { 
            v.onLineUpdated(this);
        }
    }
    
    class LineTextChanged implements TranscriptEvent {
        public final TranscriptLine line;
        public final int streamHandle;
        
        public LineTextChanged(TranscriptLine line, int streamHandle) {
            this.line = line;
            this.streamHandle = streamHandle;
        }
        
        @Override
        public void accept(Visitor v) { 
            v.onLineTextChanged(this); 
        }
    }
    
    class LineCompleted implements TranscriptEvent {
        public final TranscriptLine line;
        public final int streamHandle;
        
        public LineCompleted(TranscriptLine line, int streamHandle) {
            this.line = line;
            this.streamHandle = streamHandle;
        }
        
        @Override
        public void accept(Visitor v) { 
            v.onLineCompleted(this);
        }
    }
    
    class Error implements TranscriptEvent {
        public final Throwable cause;
        public final int streamHandle;
        
        public Error(Throwable cause, int streamHandle) {
            this.cause = cause;
            this.streamHandle = streamHandle;
        }

        @Override
        public void accept(Visitor v) { 
            v.onError(this); 
        }
    }
}