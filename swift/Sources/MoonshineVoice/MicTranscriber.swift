import AVFoundation
import Foundation

/// MicTranscriber is a class that transcribes audio from a microphone.
public class MicTranscriber {
    private let transcriber: Transcriber
    private let micStream: Stream
    private var audioEngine: AVAudioEngine?
    private var isListening: Bool = false
    private let sampleRate: Double
    private let channels: Int
    private let bufferSize: AVAudioFrameCount

    /// Initialize a MicTranscriber.
    /// - Parameters:
    ///   - modelPath: Path to the directory containing model files
    ///   - modelArch: Model architecture to use (default: `.tiny`)
    ///   - updateInterval: Interval in seconds between automatic updates (default: 0.5)
    ///   - sampleRate: Sample rate in Hz (default: 16000)
    ///   - channels: Number of audio channels (default: 1)
    ///   - bufferSize: Buffer size in frames (default: 1024)
    ///   - options: Optional transcriber options for advanced configuration
    /// - Throws: `MoonshineError` if the transcriber cannot be loaded
    public init(
        modelPath: String,
        modelArch: ModelArch = .tiny,
        updateInterval: TimeInterval = 0.5,
        sampleRate: Double = 16000,
        channels: Int = 1,
        bufferSize: AVAudioFrameCount = 1024,
        options: [TranscriberOption]? = nil
    ) throws {
        self.transcriber = try Transcriber(
            modelPath: modelPath, modelArch: modelArch, options: options)
        self.micStream = try transcriber.createStream(updateInterval: updateInterval)
        self.sampleRate = sampleRate
        self.channels = channels
        self.bufferSize = bufferSize
    }

    deinit {
        close()
    }

    /// Start listening to the microphone and begin transcription.
    /// - Throws: `MoonshineError` or `AVAudioSessionError` if starting fails
    public func start() throws {
        guard !isListening else { return }

        // Request microphone permission (platform-specific)
        #if os(iOS) || os(tvOS) || os(watchOS)
        // iOS/tvOS/watchOS: Use AVAudioSession
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .default)
        try audioSession.setActive(true)

        // Check permission status
        let permissionStatus = audioSession.recordPermission
        if permissionStatus == .denied {
            throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
        }

        if permissionStatus == .undetermined {
            // Request permission asynchronously
            var permissionGranted = false
            let semaphore = DispatchSemaphore(value: 0)

            audioSession.requestRecordPermission { granted in
                permissionGranted = granted
                semaphore.signal()
            }

            semaphore.wait()

            if !permissionGranted {
                throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
            }
        }
        #elseif os(macOS)
        // macOS: Use AVCaptureDevice for permission checking
        let permissionStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        if permissionStatus == .denied {
            throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
        }

        if permissionStatus == .notDetermined {
            // Request permission asynchronously
            var permissionGranted = false
            let semaphore = DispatchSemaphore(value: 0)

            AVCaptureDevice.requestAccess(for: .audio) { granted in
                permissionGranted = granted
                semaphore.signal()
            }

            semaphore.wait()

            if !permissionGranted {
                throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
            }
        }
        #endif

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

            // Feed audio to stream
            do {
                try self.micStream.addAudio(audioData, sampleRate: Int32(finalSampleRate))
            } catch {
                print("MicTranscriber: Error adding audio to stream: \(error.localizedDescription)")
            }
        }

        // Start the audio engine
        try engine.start()
        self.audioEngine = engine
        self.isListening = true
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

        // Stop the stream
        try micStream.stop()

        // Deactivate audio session (iOS/tvOS/watchOS only)
        #if os(iOS) || os(tvOS) || os(watchOS)
        try? AVAudioSession.sharedInstance().setActive(false)
        #endif
    }

    /// Close the transcriber and free resources.
    public func close() {
        do {
            try stop()
        } catch {
            // Ignore errors during cleanup
        }
        micStream.close()
        transcriber.close()
    }

    /// Add an event listener to the stream.
    /// - Parameter listener: Either a `TranscriptEventListener` instance or a closure
    public func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        micStream.addListener(listener)
    }

    /// Add a `TranscriptEventListener` instance to the stream.
    /// - Parameter listener: The listener object
    public func addListener(_ listener: TranscriptEventListener) {
        micStream.addListener(listener)
    }

    /// Remove an event listener from the stream.
    /// - Parameter listener: The listener to remove
    public func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        micStream.removeListener(listener)
    }

    /// Remove a `TranscriptEventListener` instance from the stream.
    /// - Parameter listener: The listener object to remove
    public func removeListener(_ listener: TranscriptEventListener) {
        micStream.removeListener(listener)
    }

    /// Remove all event listeners from the stream.
    public func removeAllListeners() {
        micStream.removeAllListeners()
    }
}
