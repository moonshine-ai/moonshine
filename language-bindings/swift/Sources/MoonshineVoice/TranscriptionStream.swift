import Foundation

/// The subset of ``Stream`` that ``MicTranscriber`` drives.
///
/// Abstracting the stream behind a protocol lets tests inject a stand-in that
/// makes the (normally native) transcription observable and deterministically
/// slow, so we can verify that transcription never runs on the audio capture
/// thread (see issue #196).
protocol TranscriptionStream: AnyObject {
    func start() throws
    func stop() throws
    func close()
    func addAudio(_ audioData: [Float], sampleRate: Int32) throws
    func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void)
    func addListener(_ listener: TranscriptEventListener)
    func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void)
    func removeListener(_ listener: TranscriptEventListener)
    func removeAllListeners()
}

extension Stream: TranscriptionStream {}
