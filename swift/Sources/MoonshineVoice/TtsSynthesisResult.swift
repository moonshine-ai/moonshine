import Foundation

/// Result of a text-to-speech synthesis call.
public struct TtsSynthesisResult {
    /// Mono PCM float samples, approximately in the range -1.0 to 1.0.
    public let samples: [Float]
    /// Sample rate in Hz (typically 24000).
    public let sampleRateHz: Int32
}
