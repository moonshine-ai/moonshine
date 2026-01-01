import Foundation

/// Main transcriber class for Moonshine Voice.
public class Transcriber {
    private let api: MoonshineAPI
    internal let handle: Int32
    private let modelPath: String
    private let modelArch: ModelArch
    
    /// Moonshine header version constant.
    public static let moonshineHeaderVersion: Int32 = 20000
    
    /// Initialize a transcriber from model files on disk.
    /// - Parameters:
    ///   - modelPath: Path to the directory containing model files
    ///   - modelArch: Model architecture to use (default: `.base`)
    ///   - options: Optional transcriber options for advanced configuration
    /// - Throws: `MoonshineError` if the transcriber cannot be loaded
    public init(
        modelPath: String,
        modelArch: ModelArch = .base,
        options: [TranscriberOption]? = nil
    ) throws {
        self.api = MoonshineAPI.shared
        self.modelPath = modelPath
        self.modelArch = modelArch
        
        self.handle = try api.loadTranscriberFromFiles(
            path: modelPath,
            modelArch: modelArch,
            options: options,
            moonshineVersion: Transcriber.moonshineHeaderVersion
        )
    }
    
    deinit {
        close()
    }
    
    /// Free the transcriber resources.
    public func close() {
        api.freeTranscriber(handle)
    }
    
    /// Transcribe audio data without streaming.
    /// - Parameters:
    ///   - audioData: Array of PCM audio samples (float, -1.0 to 1.0)
    ///   - sampleRate: Sample rate in Hz (default: 16000)
    ///   - flags: Flags for transcription (default: 0)
    /// - Returns: A `Transcript` object containing the transcription lines
    /// - Throws: `MoonshineError` if transcription fails
    public func transcribeWithoutStreaming(
        audioData: [Float],
        sampleRate: Int32 = 16000,
        flags: UInt32 = 0
    ) throws -> Transcript {
        return try api.transcribeWithoutStreaming(
            transcriberHandle: handle,
            audioData: audioData,
            sampleRate: sampleRate,
            flags: flags
        )
    }
    
    /// Get the version of the loaded Moonshine library.
    /// - Returns: The version number
    public func getVersion() -> Int32 {
        return api.getVersion()
    }
    
    /// Create a new stream for real-time transcription.
    /// - Parameters:
    ///   - updateInterval: Interval in seconds between automatic updates (default: 0.5)
    ///   - flags: Flags for stream creation (default: 0)
    /// - Returns: A `Stream` object for real-time transcription
    /// - Throws: `MoonshineError` if stream creation fails
    public func createStream(
        updateInterval: TimeInterval = 0.5,
        flags: UInt32 = 0
    ) throws -> Stream {
        let streamHandle = try api.createStream(transcriberHandle: handle, flags: flags)
        return Stream(
            transcriber: self,
            handle: streamHandle,
            updateInterval: updateInterval,
            flags: flags
        )
    }
}

