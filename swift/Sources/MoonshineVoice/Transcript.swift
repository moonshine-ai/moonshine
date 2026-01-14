import Foundation

/// A single line of transcription.
public struct TranscriptLine {
    /// UTF-8 encoded transcription text.
    public let text: String
    
    /// Time offset from the start of the audio in seconds.
    public let startTime: Float
    
    /// Duration of the segment in seconds.
    public let duration: Float
    
    /// Stable identifier for the line.
    public let lineId: UInt64
    
    /// Whether the line is complete (streaming only).
    public let isComplete: Bool
    
    /// Whether the line has been updated since the previous call (streaming only).
    public let isUpdated: Bool
    
    /// Whether the line was newly added since the previous call (streaming only).
    public let isNew: Bool
    
    /// Whether the text of the line has changed since the previous call (streaming only).
    public let hasTextChanged: Bool
    
    /// Audio data for this line, if available.
    public let audioData: [Float]?
    
    internal init(
        text: String,
        startTime: Float,
        duration: Float,
        lineId: UInt64,
        isComplete: Bool,
        isUpdated: Bool = false,
        isNew: Bool = false,
        hasTextChanged: Bool = false,
        audioData: [Float]? = nil
    ) {
        self.text = text
        self.startTime = startTime
        self.duration = duration
        self.lineId = lineId
        self.isComplete = isComplete
        self.isUpdated = isUpdated
        self.isNew = isNew
        self.hasTextChanged = hasTextChanged
        self.audioData = audioData
    }

    public var description: String {
        return "TranscriptLine(text: \(text), startTime: \(startTime), duration: \(duration), lineId: \(lineId), isComplete: \(isComplete), isUpdated: \(isUpdated), isNew: \(isNew), hasTextChanged: \(hasTextChanged))"
    }
}

/// A complete transcript containing multiple lines.
public struct Transcript {
    /// All lines of the transcript.
    public let lines: [TranscriptLine]
    
    public init(lines: [TranscriptLine] = []) {
        self.lines = lines
    }
}

extension Transcript: CustomStringConvertible {
    public var description: String {
        return "Transcript(lines: \(lines.map { $0.description }.joined(separator: "\n")))"
    }
}

