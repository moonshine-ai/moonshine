import Foundation

/// Result of a text-to-speech synthesis call.
public struct TtsSynthesisResult {
    /// Mono PCM float samples, approximately in the range -1.0 to 1.0.
    public let samples: [Float]
    /// Sample rate in Hz (typically 24000).
    public let sampleRateHz: Int32
}

/// One piece of streamed audio from ``TextToSpeech/stream(_:)``.
public struct TtsChunk: Sendable {
    /// Mono PCM float samples, approximately in the range -1.0 to 1.0.
    public let samples: [Float]
    /// Sample rate in Hz (typically 24000).
    public let sampleRateHz: Int32
    /// The text this chunk covers, or `""` when the engine cannot attribute it.
    public let text: String
    /// Which queued utterance this chunk belongs to, counting from zero.
    public let utteranceId: UInt64
    /// True for the last chunk of an utterance.
    public let isFinal: Bool
}
