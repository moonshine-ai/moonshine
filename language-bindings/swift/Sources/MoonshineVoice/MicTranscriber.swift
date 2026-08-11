@preconcurrency import AVFoundation
import Foundation

/// MicTranscriber is a class that transcribes audio from a microphone.
///
/// Marked `@unchecked Sendable` because it manages its own cross-thread
/// safety: captured audio crosses from the capture thread to a serial
/// ``audioQueue`` guarded by ``pendingLock``, and lifecycle calls
/// (``start``/``stop``) are expected from a single controlling thread.
public final class MicTranscriber: @unchecked Sendable {
    private var transcriber: Transcriber?
    private var ownsTranscriber = true
    private var micStream: TranscriptionStream?
    private var audioEngine: AVAudioEngine?
    private var isListening: Bool = false
    private let sampleRate: Double
    private let channels: Int
    private let bufferSize: AVAudioFrameCount

    // Deferred configuration, applied by ``load()``.
    private var languageCode = "en"
    private var arch: ModelArch = .mediumStreaming
    private var modelDirectory: URL?
    private var streamUpdateInterval: TimeInterval = 0.5
    private var transcriberOptions: [TranscriberOption] = []
    private var transcribeFlags: UInt32 = 0
    private var progressHandler: (@Sendable (Double, String) -> Void)?
    private var isMuted = false

    /// Listeners registered before the model finished loading, attached to the
    /// stream by ``prepareStream()``.
    private enum PendingListener {
        case closure((TranscriptEvent) throws -> Void)
        case object(TranscriptEventListener)
    }
    private var pendingListeners: [PendingListener] = []

    // Captured audio is handed to this serial queue, which runs the blocking
    // transcription off the time-critical capture thread (see issue #196).
    private let audioQueue = DispatchQueue(label: "ai.moonshine.MicTranscriber.audio")
    private let pendingLock = NSLock()
    private var pendingChunks: [(audio: [Float], sampleRate: Int32)] = []

    /// Creates a transcriber that has not loaded a model yet.
    ///
    /// Configure it with the chainable setters below, then `try await load()`:
    ///
    /// ```swift
    /// let mic = MicTranscriber()
    ///     .onText { print($0, terminator: "\r") }
    ///     .onLine { print($0.text) }
    ///
    /// try await mic.load()
    /// try mic.start()
    /// ```
    public init(
        sampleRate: Double = 16000,
        channels: Int = 1,
        bufferSize: AVAudioFrameCount = 1024
    ) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.bufferSize = bufferSize
    }

    /// Initialize a MicTranscriber from a model already on disk.
    /// - Parameters:
    ///   - modelPath: Path to the directory containing model files
    ///   - modelArch: Model architecture to use (default: `.tiny`)
    ///   - updateInterval: Interval in seconds between automatic updates (default: 0.5)
    ///   - sampleRate: Sample rate in Hz (default: 16000)
    ///   - channels: Number of audio channels (default: 1)
    ///   - bufferSize: Buffer size in frames (default: 1024)
    ///   - options: Optional transcriber options for advanced configuration
    ///   - spellingModelPath: Convenience shortcut for the
    ///     ``"spelling_model_path"`` option used by the alphanumeric
    ///     spelling-fusion path.
    ///   - transcribeFlags: Flags applied to every implicit
    ///     transcription update the underlying stream issues. Pass
    ///     ``TranscribeStreamFlags.flagSpellingMode`` to enable the
    ///     C++ spelling-CNN fusion path on live mic audio (requires
    ///     ``spellingModelPath`` to be set).
    /// - Throws: `MoonshineError` if the transcriber cannot be loaded
    public init(
        modelPath: String,
        modelArch: ModelArch = .tiny,
        updateInterval: TimeInterval = 0.5,
        sampleRate: Double = 16000,
        channels: Int = 1,
        bufferSize: AVAudioFrameCount = 1024,
        options: [TranscriberOption]? = nil,
        spellingModelPath: String? = nil,
        transcribeFlags: UInt32 = 0
    ) throws {
        let transcriber = try Transcriber(
            modelPath: modelPath,
            modelArch: modelArch,
            options: options,
            spellingModelPath: spellingModelPath)
        self.transcriber = transcriber
        self.micStream = try transcriber.createStream(
            updateInterval: updateInterval,
            transcribeFlags: transcribeFlags)
        self.sampleRate = sampleRate
        self.channels = channels
        self.bufferSize = bufferSize
    }

    /// Test-only initializer that injects a stream stand-in and puts the
    /// transcriber into the "listening" state without opening an
    /// ``AVAudioEngine``. This lets tests drive ``feedCapturedAudio`` directly
    /// (simulating the capture tap) without real microphone hardware or model
    /// assets. The injected stream is started immediately so it behaves like a
    /// running mic stream.
    internal init(
        testStream: TranscriptionStream,
        sampleRate: Double = 16000,
        channels: Int = 1,
        bufferSize: AVAudioFrameCount = 1024
    ) throws {
        self.transcriber = nil
        self.ownsTranscriber = false
        self.micStream = testStream
        self.sampleRate = sampleRate
        self.channels = channels
        self.bufferSize = bufferSize
        try testStream.start()
        self.isListening = true
    }

    deinit {
        close()
    }

    // MARK: - Configuration

    /// Speech-to-text language. Defaults to `"en"`.
    @discardableResult
    public func language(_ code: String) -> Self {
        languageCode = code
        return self
    }

    /// Overrides the streaming model. Defaults to `.mediumStreaming`.
    @discardableResult
    public func modelArch(_ arch: ModelArch) -> Self {
        self.arch = arch
        return self
    }

    /// Loads the model from a directory you supply rather than the Moonshine CDN.
    @discardableResult
    public func modelsFrom(_ directory: URL) -> Self {
        modelDirectory = directory
        return self
    }

    /// Reuses an already-loaded transcriber rather than loading another.
    @discardableResult
    public func useTranscriber(_ transcriber: Transcriber) -> Self {
        self.transcriber = transcriber
        self.ownsTranscriber = false
        return self
    }

    /// Seconds between automatic streaming updates. Defaults to 0.5.
    @discardableResult
    public func updateInterval(_ seconds: TimeInterval) -> Self {
        streamUpdateInterval = seconds
        return self
    }

    /// Escape hatch for options the chainable setters don't cover.
    @discardableResult
    public func options(_ options: [TranscriberOption]) -> Self {
        transcriberOptions.append(contentsOf: options)
        return self
    }

    /// Called with the in-progress text of the line currently being spoken.
    @discardableResult
    public func onText(_ handler: @escaping (String) -> Void) -> Self {
        addListener { event in
            if let changed = event as? LineTextChanged {
                handler(changed.line.text)
            }
        }
        return self
    }

    /// Called once per finished line.
    @discardableResult
    public func onLine(_ handler: @escaping (TranscriptLine) -> Void) -> Self {
        addListener { event in
            if let completed = event as? LineCompleted {
                handler(completed.line)
            }
        }
        return self
    }

    @discardableResult
    public func onError(_ handler: @escaping (Error) -> Void) -> Self {
        addListener { event in
            if let failure = event as? TranscriptError {
                handler(failure.error)
            }
        }
        return self
    }

    /// Model download progress, as a `0..1` fraction plus the file being fetched.
    @discardableResult
    public func onProgress(_ handler: @escaping @Sendable (Double, String) -> Void) -> Self {
        progressHandler = handler
        return self
    }

    /// Every completed line, as an `AsyncSequence`.
    ///
    /// ```swift
    /// for try await line in mic.transcript {
    ///     print(line.text)
    /// }
    /// ```
    ///
    /// Each access registers a fresh listener, so hold on to the sequence rather
    /// than re-reading the property in a loop.
    public var transcript: AsyncThrowingStream<TranscriptLine, Error> {
        AsyncThrowingStream { continuation in
            addListener { event in
                if let completed = event as? LineCompleted {
                    continuation.yield(completed.line)
                } else if let failure = event as? TranscriptError {
                    continuation.finish(throwing: failure.error)
                }
            }
        }
    }

    // MARK: - Loading

    /// Downloads the model if needed and prepares the transcriber.
    @available(iOS 15.0, macOS 12.0, *)
    public func load() async throws {
        guard transcriber == nil else {
            try prepareStream()
            return
        }
        let directory: URL
        if let modelDirectory {
            directory = modelDirectory
        } else {
            let spec = ModelSpec.stt(language: languageCode, modelArch: arch)
            directory = try ModelCache.directory(for: spec)
            _ = try await AssetDownloader().ensureModelPresent(
                root: directory, spec: spec, onProgress: fractionReporter(progressHandler))
        }
        let options = try await withDiarizationModels(
            transcriberOptions.isEmpty ? nil : transcriberOptions,
            downloader: AssetDownloader(),
            onProgress: fractionReporter(progressHandler))
        transcriber = try Transcriber(
            modelPath: directory.path,
            modelArch: arch,
            options: options)
        ownsTranscriber = true
        try prepareStream()
    }

    /// Builds the streaming session and attaches any listeners registered before
    /// the model finished loading.
    private func prepareStream() throws {
        guard micStream == nil, let transcriber else { return }
        let stream = try transcriber.createStream(
            updateInterval: streamUpdateInterval, transcribeFlags: transcribeFlags)
        for listener in pendingListeners {
            switch listener {
            case .closure(let closure): stream.addListener(closure)
            case .object(let object): stream.addListener(object)
            }
        }
        pendingListeners.removeAll()
        micStream = stream
    }

    /// Drops incoming audio without tearing down the microphone. Used to stop
    /// an assistant transcribing its own synthesized speech.
    public func mute(_ muted: Bool = true) {
        isMuted = muted
    }

    /// Biases the decoder towards a list of terms while listening, replacing any
    /// previous list. See ``Transcriber/setKeyterms(_:)``; this can be called
    /// between phrases to follow whatever the user is looking at. To start with a
    /// list instead, pass the ``keyterms`` transcriber option to ``options(_:)``.
    /// - Throws: ``MoonshineError`` if no model is loaded yet, if a term contains
    ///   a comma, or if the model is not a streaming architecture.
    public func setKeyterms(_ keyterms: [String]) throws {
        guard let transcriber else {
            throw MoonshineError.custom(
                message: "No model loaded. Call load() before setKeyterms().", code: -1)
        }
        try transcriber.setKeyterms(keyterms)
    }

    /// Picks the key terms out of a passage of text and biases towards them while
    /// listening. See ``Transcriber/setContext(_:maxTerms:)``; this can be called
    /// between phrases to follow the document or thread the user is in. To start
    /// with a passage instead, pass the ``context`` transcriber option to
    /// ``options(_:)``.
    /// - Throws: ``MoonshineError`` if no model is loaded yet, or if the model is
    ///   not a streaming architecture.
    public func setContext(_ context: String, maxTerms: Int32 = 0) throws {
        guard let transcriber else {
            throw MoonshineError.custom(
                message: "No model loaded. Call load() before setContext().", code: -1)
        }
        try transcriber.setContext(context, maxTerms: maxTerms)
    }

    /// Start listening to the microphone and begin transcription.
    /// - Throws: `MoonshineError` or `AVAudioSessionError` if starting fails
    public func start() throws {
        guard !isListening else { return }
        try prepareStream()
        guard let micStream else {
            throw MoonshineError.custom(
                message: "No model loaded. Call load() before start().", code: -1)
        }

        try ensureMicrophonePermission()

        // Start the stream
        try micStream.start()

        // Set up audio engine
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        // Create target format
        guard
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels),
                interleaved: false
            )
        else {
            throw MoonshineError.custom(message: "Failed to create target audio format", code: -1)
        }

        // Check if conversion is needed
        let needsConversion =
            inputFormat.sampleRate != targetFormat.sampleRate
            || inputFormat.channelCount != targetFormat.channelCount
            || inputFormat.commonFormat != targetFormat.commonFormat

        let converter: AVAudioConverter? =
            needsConversion ? AVAudioConverter(from: inputFormat, to: targetFormat) : nil

        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) {
            [weak self] (buffer, time) in
            guard let self = self, self.isListening else { return }

            var audioData: [Float] = []
            var finalSampleRate: Double = self.sampleRate

            if let converter = converter {
                // Convert buffer to target format
                let capacity = AVAudioFrameCount(
                    Double(buffer.frameLength) * targetFormat.sampleRate / inputFormat.sampleRate)
                guard
                    let convertedBuffer = AVAudioPCMBuffer(
                        pcmFormat: targetFormat, frameCapacity: capacity)
                else {
                    return
                }

                var error: NSError?
                let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                    outStatus.pointee = .haveData
                    return buffer
                }

                converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

                if let error = error {
                    print("MicTranscriber: Audio conversion error: \(error.localizedDescription)")
                    return
                }

                // Extract audio data from converted buffer
                guard let channelData = convertedBuffer.floatChannelData else { return }
                let frameLength = Int(convertedBuffer.frameLength)
                audioData.reserveCapacity(frameLength * channels)

                if channels == 1 {
                    // Mono: copy single channel
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                } else {
                    // Multi-channel: use first channel
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                }
            } else {
                // No conversion needed, extract directly from buffer
                guard let channelData = buffer.floatChannelData else { return }
                let frameLength = Int(buffer.frameLength)
                audioData.reserveCapacity(frameLength * channels)

                if channels == 1 {
                    // Mono: copy single channel
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                } else {
                    // Multi-channel: use first channel
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                }
                finalSampleRate = inputFormat.sampleRate
            }

            // Hand the captured audio to the (time-critical-safe) sink.
            self.feedCapturedAudio(audioData, sampleRate: Int32(finalSampleRate))
        }

        // Start the audio engine
        try engine.start()
        self.audioEngine = engine
        self.isListening = true
    }

    /// Sink for a single captured audio buffer.
    ///
    /// Called from the ``AVAudioEngine`` capture tap (a high-priority audio
    /// thread), so it must return promptly and must never run transcription
    /// inline, or the capture buffer overflows and audio is dropped (see
    /// issue #196). The buffer is queued for a worker to transcribe and this
    /// returns immediately.
    internal func feedCapturedAudio(_ audioData: [Float], sampleRate: Int32) {
        pendingLock.lock()
        pendingChunks.append((audioData, sampleRate))
        pendingLock.unlock()
        audioQueue.async { [weak self] in
            self?.drainPendingAudio()
        }
    }

    /// Drain everything queued so far into the stream, running on ``audioQueue``.
    ///
    /// Consecutive chunks sharing a sample rate are concatenated so a backlog
    /// (which builds while a slow transcription is running) is transcribed in a
    /// single pass instead of once per chunk.
    private func drainPendingAudio() {
        pendingLock.lock()
        let batch = pendingChunks
        pendingChunks.removeAll(keepingCapacity: true)
        pendingLock.unlock()
        guard !batch.isEmpty else { return }

        var runAudio: [Float] = []
        var runRate: Int32? = nil
        for chunk in batch {
            if let rate = runRate, rate != chunk.sampleRate {
                addRun(runAudio, sampleRate: rate)
                runAudio.removeAll(keepingCapacity: true)
            }
            runAudio.append(contentsOf: chunk.audio)
            runRate = chunk.sampleRate
        }
        if let rate = runRate {
            addRun(runAudio, sampleRate: rate)
        }
    }

    private func addRun(_ audioData: [Float], sampleRate: Int32) {
        guard !isMuted, let micStream else { return }
        do {
            try micStream.addAudio(audioData, sampleRate: sampleRate)
        } catch {
            print("MicTranscriber: Error adding audio to stream: \(error.localizedDescription)")
        }
    }

    /// Stop listening to the microphone and stop transcription.
    /// - Throws: `MoonshineError` if stopping fails
    public func stop() throws {
        guard isListening else { return }

        isListening = false

        // Remove tap and stop engine
        if let engine = audioEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            audioEngine = nil
        }

        // Drain any queued audio and run the final flush on the audio queue,
        // so every stream call is serialized on one thread and no transcription
        // is still running on a worker when we stop.
        var flushError: Error?
        audioQueue.sync {
            self.drainPendingAudio()
            do {
                try self.micStream?.stop()
            } catch {
                flushError = error
            }
        }
        releaseMicrophoneSession()

        if let flushError {
            throw flushError
        }
    }

    /// Close the transcriber and free resources.
    public func close() {
        do {
            try stop()
        } catch {
            // Ignore errors during cleanup
        }
        micStream?.close()
        micStream = nil
        if ownsTranscriber {
            transcriber?.close()
        }
        transcriber = nil
    }

    /// Add an event listener. Safe to call before ``load()``; listeners
    /// registered early are attached to the stream once it exists.
    /// - Parameter listener: Either a `TranscriptEventListener` instance or a closure
    public func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        if let micStream {
            micStream.addListener(listener)
        } else {
            pendingListeners.append(.closure(listener))
        }
    }

    /// Add a `TranscriptEventListener` instance to the stream.
    /// - Parameter listener: The listener object
    public func addListener(_ listener: TranscriptEventListener) {
        if let micStream {
            micStream.addListener(listener)
        } else {
            pendingListeners.append(.object(listener))
        }
    }

    /// Remove an event listener from the stream.
    /// - Parameter listener: The listener to remove
    public func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        micStream?.removeListener(listener)
    }

    /// Remove a `TranscriptEventListener` instance from the stream.
    /// - Parameter listener: The listener object to remove
    public func removeListener(_ listener: TranscriptEventListener) {
        micStream?.removeListener(listener)
    }

    /// Remove all event listeners from the stream.
    public func removeAllListeners() {
        micStream?.removeAllListeners()
        pendingListeners.removeAll()
    }
}
