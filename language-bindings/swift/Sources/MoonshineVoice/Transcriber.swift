import Foundation

/// Main transcriber class for Moonshine Voice.
public class Transcriber {
    private let api: MoonshineAPI
    internal let handle: Int32
    private let modelPath: String
    private let modelArch: ModelArch
    private var defaultStream: Stream? = nil

    /// Moonshine header version constant.
    public static let moonshineHeaderVersion: Int32 = 30000

    /// Get the bundle for the moonshine framework (where resources are located).
    /// - Returns: The framework bundle, or nil if not found
    public static var frameworkBundle: Bundle? {
        // Try bundle identifier first
        if let bundle = Bundle(identifier: "ai.moonshine.voice") {
            return bundle
        }
        // Fallback: try to find the framework bundle by searching
        if let frameworkPath = Bundle(for: Transcriber.self).path(
            forResource: "moonshine", ofType: "framework"),
            let bundle = Bundle(path: frameworkPath)
        {
            return bundle
        }
        // Last resort: return the module bundle
        return Bundle(for: Transcriber.self)
    }

    /// Initialize a transcriber from model files on disk.
    /// - Parameters:
    ///   - modelPath: Path to the directory containing model files
    ///   - modelArch: Model architecture to use (default: `.base`)
    ///   - options: Optional transcriber options for advanced configuration
    ///   - spellingModelPath: Convenience shortcut for the
    ///     ``"spelling_model_path"`` option used by the alphanumeric
    ///     spelling-fusion path (see
    ///     ``TranscribeStreamFlags.flagSpellingMode``). When non-nil,
    ///     a corresponding entry is appended to ``options``.
    /// - Throws: `MoonshineError` if the transcriber cannot be loaded
    public init(
        modelPath: String,
        modelArch: ModelArch = .base,
        options: [TranscriberOption]? = nil,
        spellingModelPath: String? = nil
    ) throws {
        self.api = MoonshineAPI.shared
        self.modelPath = modelPath
        self.modelArch = modelArch

        var resolvedOptions = options ?? []
        if let spellingModelPath = spellingModelPath, !spellingModelPath.isEmpty {
            resolvedOptions.append(
                TranscriberOption(name: "spelling_model_path",
                                  value: spellingModelPath))
        }

        self.handle = try api.loadTranscriberFromFiles(
            path: modelPath,
            modelArch: modelArch,
            options: resolvedOptions.isEmpty ? nil : resolvedOptions,
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
    ///   - transcribeFlags: Flags applied to every implicit
    ///     ``updateTranscription`` call the stream issues from
    ///     ``addAudio`` / ``stop``. Pass
    ///     ``TranscribeStreamFlags.flagSpellingMode`` to drive the C++
    ///     spelling-CNN fusion on live mic audio (default: 0).
    /// - Returns: A `Stream` object for real-time transcription
    /// - Throws: `MoonshineError` if stream creation fails
    public func createStream(
        updateInterval: TimeInterval = 0.5,
        flags: UInt32 = 0,
        transcribeFlags: UInt32 = 0
    ) throws -> Stream {
        let streamHandle = try api.createStream(transcriberHandle: handle, flags: flags)
        return Stream(
            transcriber: self,
            handle: streamHandle,
            updateInterval: updateInterval,
            flags: flags,
            transcribeFlags: transcribeFlags
        )
    }

    /// Bias the decoder towards a list of terms, replacing any previous list.
    ///
    /// Useful for jargon, product names and proper nouns the model would
    /// otherwise be unlikely to produce. No retraining is involved, so the list
    /// can follow whatever the user is looking at and can be changed while a
    /// stream is running; it takes effect on the next transcription and does not
    /// rewrite text already emitted.
    ///
    /// Match the capitalization and spelling you want to see in the output.
    /// Pass an empty array to turn biasing off. Set the strength with the
    /// ``keyterm_boost`` option at load time.
    /// - Parameter keyterms: Terms to bias towards, e.g. `["Kubernetes", "Ceph"]`.
    ///   Commas are the delimiter used internally, so terms must not contain them.
    /// - Throws: ``MoonshineError`` if a term contains a comma, or if the loaded
    ///   model is not a streaming architecture — only those decode through a
    ///   path that can apply the bias.
    public func setKeyterms(_ keyterms: [String]) throws {
        for term in keyterms where term.contains(",") {
            throw MoonshineError.custom(
                message: "Key terms cannot contain commas, which separate them: \(term)",
                code: Int32(-3))  // MOONSHINE_ERROR_INVALID_ARGUMENT
        }
        try api.setKeyterms(transcriberHandle: handle, keyterms: keyterms.joined(separator: ","))
    }

    /// Pick the key terms out of a passage of text and bias towards them,
    /// replacing any previous list.
    ///
    /// Where ``setKeyterms(_:)`` wants a list, this wants context: pass the
    /// document on screen, the agenda for the meeting, the last few messages in
    /// the thread, and the unusual words in it are found for you. A word counts
    /// as unusual when the model's own tokenizer has no single symbol for it,
    /// which is the case biasing helps with, so the judgment follows the
    /// language of the loaded model with no word lists involved.
    ///
    /// Like ``setKeyterms(_:)``, this can be called while a stream is running,
    /// takes effect on the next transcription, and does not rewrite text already
    /// emitted. The capitalization in the passage is what gets asked for in the
    /// transcript.
    /// - Parameters:
    ///   - context: The passage to read terms out of. Pass an empty string to
    ///     turn biasing off.
    ///   - maxTerms: Most terms to take, 200 by default. Worth keeping modest:
    ///     a long list costs accuracy on the words you did not ask for, so the
    ///     terms the passage leans on hardest are kept and its long tail is
    ///     dropped.
    /// - Throws: ``MoonshineError`` if the loaded model is not a streaming
    ///   architecture — only those decode through a path that can apply the bias.
    public func setContext(_ context: String, maxTerms: Int32 = 0) throws {
        try api.setContext(transcriberHandle: handle, context: context, maxTerms: maxTerms)
    }

    /// Get the default stream handle.
    public func getDefaultStream() throws -> Stream {
        if defaultStream == nil {
            do {
                defaultStream = try createStream()
            } catch {
                throw MoonshineError.custom(message: "Failed to create default stream", code: -1)
            }
        }
        return defaultStream!
    }

    public func start() throws {
        try getDefaultStream().start()
    }

    public func stop() throws {
        try getDefaultStream().stop()
    }

    public func addAudio(_ audioData: [Float], sampleRate: Int32) throws {
        try getDefaultStream().addAudio(audioData, sampleRate: sampleRate)
    }

    public func updateTranscription() throws -> Transcript {
        return try getDefaultStream().updateTranscription()
    }

    public func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void) throws {
        try getDefaultStream().addListener(listener)
    }

    public func addListener(_ listener: TranscriptEventListener) throws {
        try getDefaultStream().addListener(listener)
    }

    public func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void) throws {
        try getDefaultStream().removeListener(listener)
    }

    public func removeListener(_ listener: TranscriptEventListener) throws {
        try getDefaultStream().removeListener(listener)
    }

    public func removeAllListeners() throws {
        try getDefaultStream().removeAllListeners()
    }
}
