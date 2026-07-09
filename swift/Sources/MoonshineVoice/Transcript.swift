import Foundation

/// A single word with timing information.
public struct WordTiming {
    /// The word text.
    public let word: String
    /// Start time in seconds (absolute, from start of audio/stream).
    public let start: Float
    /// End time in seconds.
    public let end: Float
    /// Model confidence score, 0.0 to 1.0.
    public let confidence: Float
}

/// One contiguous span of speech within a line attributed to a single
/// speaker. Only populated when the `identify_speakers` option is enabled.
///
/// Spans for recent audio are mutable: streaming diarization re-clusters a
/// sliding window (`diarization_cluster_window_sec`, default 120s) as more
/// speech arrives. Assignments for older audio are frozen. Watch
/// `TranscriptLine.haveSpeakersChanged` to detect revisions.
public struct SpeakerSpan {
    /// Time offset from the start of the audio or stream in seconds.
    public let startTime: Float
    /// Length of the span in seconds.
    public let duration: Float
    /// Stable identifier for the speaker within this stream.
    public let speakerId: UInt64
    /// The order the speaker first appeared in the transcript, starting at 0.
    public let speakerIndex: UInt32
    /// UTF-8 byte offset into the line text where this span begins (inclusive).
    public let startChar: UInt64
    /// UTF-8 byte offset into the line text where this span ends (exclusive).
    public let endChar: UInt64
}

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

    /// Whether the speaker spans of the line have changed since the previous
    /// call. Unlike the other change flags, this can fire for lines that are
    /// already complete, since diarization refines speaker assignments
    /// retroactively as more audio arrives.
    public let haveSpeakersChanged: Bool

    /// Speaker spans covering this line, ordered by start time and clipped to
    /// the line's time range. Empty unless the `identify_speakers` option is
    /// enabled and speech has been attributed to a speaker.
    public let speakerSpans: [SpeakerSpan]
    
    /// Audio data for this line, if available.
    public let audioData: [Float]?

    /// Word-level timestamps. Empty if word_timestamps option is not enabled.
    public let words: [WordTiming]

    internal init(
        text: String,
        startTime: Float,
        duration: Float,
        lineId: UInt64,
        isComplete: Bool,
        isUpdated: Bool = false,
        isNew: Bool = false,
        hasTextChanged: Bool = false,
        haveSpeakersChanged: Bool = false,
        speakerSpans: [SpeakerSpan] = [],
        audioData: [Float]? = nil,
        words: [WordTiming] = []
    ) {
        self.text = text
        self.startTime = startTime
        self.duration = duration
        self.lineId = lineId
        self.isComplete = isComplete
        self.isUpdated = isUpdated
        self.isNew = isNew
        self.hasTextChanged = hasTextChanged
        self.haveSpeakersChanged = haveSpeakersChanged
        self.speakerSpans = speakerSpans
        self.audioData = audioData
        self.words = words
    }

    public var description: String {
        let spans = speakerSpans.map {
            "(start: \($0.startTime), duration: \($0.duration), speakerId: \($0.speakerId), speakerIndex: \($0.speakerIndex))"
        }.joined(separator: ", ")
        return "TranscriptLine(text: \(text), startTime: \(startTime), duration: \(duration), lineId: \(lineId), isComplete: \(isComplete), isUpdated: \(isUpdated), isNew: \(isNew), hasTextChanged: \(hasTextChanged), haveSpeakersChanged: \(haveSpeakersChanged), speakerSpans: [\(spans)])"
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

