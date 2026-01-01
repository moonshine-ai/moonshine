import Foundation

/// Base protocol for all transcript events.
public protocol TranscriptEvent {
    /// The transcript line associated with this event.
    var line: TranscriptLine { get }
    
    /// The handle of the stream that emitted this event.
    var streamHandle: Int32 { get }
}

/// Event emitted when a new transcription line starts.
public struct LineStarted: TranscriptEvent {
    public let line: TranscriptLine
    public let streamHandle: Int32
    
    public init(line: TranscriptLine, streamHandle: Int32) {
        self.line = line
        self.streamHandle = streamHandle
    }
}

/// Event emitted when an existing transcription line is updated.
public struct LineUpdated: TranscriptEvent {
    public let line: TranscriptLine
    public let streamHandle: Int32
    
    public init(line: TranscriptLine, streamHandle: Int32) {
        self.line = line
        self.streamHandle = streamHandle
    }
}

/// Event emitted when the text of a transcription line changes.
public struct LineTextChanged: TranscriptEvent {
    public let line: TranscriptLine
    public let streamHandle: Int32
    
    public init(line: TranscriptLine, streamHandle: Int32) {
        self.line = line
        self.streamHandle = streamHandle
    }
}

/// Event emitted when a transcription line is completed.
public struct LineCompleted: TranscriptEvent {
    public let line: TranscriptLine
    public let streamHandle: Int32
    
    public init(line: TranscriptLine, streamHandle: Int32) {
        self.line = line
        self.streamHandle = streamHandle
    }
}

/// Event emitted when an error occurs.
public struct TranscriptError: TranscriptEvent {
    private let _line: TranscriptLine?
    public let streamHandle: Int32
    public let error: Error
    
    /// Convenience property to satisfy TranscriptEvent protocol
    public var line: TranscriptLine {
        return _line ?? TranscriptLine(
            text: "",
            startTime: 0,
            duration: 0,
            lineId: 0,
            isComplete: false
        )
    }
    
    public init(line: TranscriptLine?, streamHandle: Int32, error: Error) {
        self._line = line
        self.streamHandle = streamHandle
        self.error = error
    }
}

