import AVFoundation
import Foundation

#if os(macOS)
import CoreAudio
#else
/// Placeholder type on non-macOS platforms where CoreAudio's AudioDeviceID is unavailable.
public typealias AudioDeviceID = UInt32
#endif

/// On-device text-to-speech using the Moonshine native API (Kokoro / Piper).
///
/// Provide a `g2pRoot` directory containing G2P and vocoder assets. Use
/// ``synthesize(text:options:)`` to get raw PCM samples, or ``say(_:device:options:)``
/// to queue text for synthesis and playback.
///
/// Usage:
/// ```swift
/// let tts = try TextToSpeech(language: "en_us", g2pRoot: "/path/to/assets")
/// let result = try tts.synthesize(text: "Hello world!")
/// // or play directly (returns immediately):
/// tts.say("Hello world!")
/// tts.wait()   // block until done
/// tts.stop()   // cancel all pending + halt playback
/// tts.close()
/// ```
public class TextToSpeech: @unchecked Sendable {
    private let api: MoonshineAPI
    private var handle: Int32
    private let _language: String

    private let sayLock = NSLock()
    private var sayEngine: AVAudioEngine?
    private var sayPlayerNode: AVAudioPlayerNode?
    #if os(macOS)
    private var sayCachedDeviceID: AudioDeviceID?
    #endif
    private var sayCachedSampleRate: Int32 = 0

    // Queue infrastructure: two serial GCD queues form a pipeline.
    // synthQueue synthesizes the next utterance while playbackQueue plays the current one.
    private let synthQueue = DispatchQueue(label: "ai.moonshine.tts.synth")
    private let playbackQueue = DispatchQueue(label: "ai.moonshine.tts.play")
    private let stateLock = NSLock()
    private var stopGeneration: UInt64 = 0
    private let pendingCondition = NSCondition()
    private var pendingCount = 0

    private struct PlayItem {
        let samples: [Float]
        let sampleRate: Int32
        let deviceID: AudioDeviceID?
    }

    /// Moonshine header version constant.
    public static let moonshineHeaderVersion: Int32 = 20000

    /// Initialize a TTS synthesizer from asset files on disk.
    ///
    /// - Parameters:
    ///   - language: Moonshine language tag (e.g. `en_us`, `de`, `fr`).
    ///   - g2pRoot: Path to the directory containing G2P and vocoder assets.
    ///   - voice: Optional voice ID (e.g. `kokoro_af_heart`, `piper_en`).
    ///   - options: Additional options for advanced configuration.
    /// - Throws: `MoonshineError` if the synthesizer cannot be created.
    public init(
        language: String,
        g2pRoot: String,
        voice: String? = nil,
        options: [TranscriberOption]? = nil
    ) throws {
        self.api = MoonshineAPI.shared
        self._language = language

        var allOptions = options ?? []
        allOptions.append(TranscriberOption(name: "g2p_root", value: g2pRoot))
        if let voice = voice {
            allOptions.append(TranscriberOption(name: "voice", value: voice))
        }

        self.handle = try api.createTtsSynthesizerFromFiles(
            language: language,
            options: allOptions,
            moonshineVersion: TextToSpeech.moonshineHeaderVersion
        )
    }

    deinit {
        close()
    }

    /// The language tag this synthesizer was created with.
    public var language: String {
        return _language
    }

    /// Synthesize text to mono PCM float samples and sample rate.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - options: Optional per-call options (e.g. `speed`).
    /// - Returns: A ``TtsSynthesisResult`` with PCM samples and sample rate.
    /// - Throws: `MoonshineError` if synthesis fails.
    public func synthesize(
        text: String,
        options: [TranscriberOption]? = nil
    ) throws -> TtsSynthesisResult {
        return try api.textToSpeech(
            ttsHandle: handle,
            text: text,
            options: options
        )
    }

    // MARK: - Queued say / stop / wait / isTalking

    /// Queue ``text`` for synthesis and playback, returning immediately.
    ///
    /// ``text`` may be a single string or an array. An array is equivalent to calling
    /// ``say`` once per element in order. Utterances are played in order; synthesis of the
    /// next utterance is pipelined with playback of the current one. Call ``stop()`` to
    /// cancel all pending utterances and halt the currently-playing audio.
    ///
    /// - Parameters:
    ///   - text: The text to speak (single string).
    ///   - options: Optional per-call synthesis options (e.g. `speed`).
    public func say(
        _ text: String,
        options: [TranscriberOption]? = nil
    ) {
        enqueueSay(text: text, deviceID: nil, options: options)
    }

    /// Queue each string for synthesis and playback, returning immediately.
    ///
    /// - Parameters:
    ///   - texts: An array of strings to speak in order.
    ///   - options: Optional per-call synthesis options.
    public func say(
        _ texts: [String],
        options: [TranscriberOption]? = nil
    ) {
        for text in texts {
            enqueueSay(text: text, deviceID: nil, options: options)
        }
    }

    #if os(macOS)
    /// Queue ``text`` for synthesis and playback on a specific device, returning immediately.
    ///
    /// - Parameters:
    ///   - text: The text to speak.
    ///   - device: An `AudioDeviceID` to route output to, or `nil` for the
    ///     system default output device.
    ///   - options: Optional per-call synthesis options.
    public func say(
        _ text: String,
        device: AudioDeviceID?,
        options: [TranscriberOption]? = nil
    ) {
        enqueueSay(text: text, deviceID: device, options: options)
    }

    /// Queue each string for synthesis and playback on a specific device, returning immediately.
    ///
    /// - Parameters:
    ///   - texts: An array of strings to speak in order.
    ///   - device: An `AudioDeviceID` to route output to, or `nil` for the
    ///     system default output device.
    ///   - options: Optional per-call synthesis options.
    public func say(
        _ texts: [String],
        device: AudioDeviceID?,
        options: [TranscriberOption]? = nil
    ) {
        for text in texts {
            enqueueSay(text: text, deviceID: device, options: options)
        }
    }
    #endif

    /// Block until all queued utterances have been synthesized and played.
    public func wait() {
        pendingCondition.lock()
        while pendingCount > 0 {
            pendingCondition.wait()
        }
        pendingCondition.unlock()
    }

    /// Clear the utterance queue and stop any audio currently playing.
    ///
    /// Returns once all pending utterances are discarded and the active playback (if any)
    /// has been halted. It is safe to call ``say`` again afterwards.
    public func stop() {
        stateLock.lock()
        stopGeneration += 1
        stateLock.unlock()

        sayLock.lock()
        sayPlayerNode?.stop()
        sayLock.unlock()
    }

    /// Returns `true` if utterances are queued, being synthesized, or currently playing.
    public func isTalking() -> Bool {
        pendingCondition.lock()
        let count = pendingCount
        pendingCondition.unlock()
        return count > 0
    }

    // MARK: - Queue internals

    private func enqueueSay(text: String, deviceID: AudioDeviceID?, options: [TranscriberOption]?) {
        pendingCondition.lock()
        pendingCount += 1
        pendingCondition.unlock()

        stateLock.lock()
        let gen = stopGeneration
        stateLock.unlock()

        synthQueue.async { [self] in
            guard self.isGenerationCurrent(gen) else {
                self.decrementPending()
                return
            }

            guard let result = try? self.synthesize(text: text, options: options),
                  result.sampleRateHz > 0,
                  !result.samples.isEmpty else {
                self.decrementPending()
                return
            }

            guard self.isGenerationCurrent(gen) else {
                self.decrementPending()
                return
            }

            let item = PlayItem(
                samples: result.samples,
                sampleRate: result.sampleRateHz,
                deviceID: deviceID
            )

            self.playbackQueue.async { [self] in
                defer { self.decrementPending() }
                guard self.isGenerationCurrent(gen) else { return }
                self.playOneItem(item, generation: gen)
            }
        }
    }

    private func isGenerationCurrent(_ gen: UInt64) -> Bool {
        stateLock.lock()
        let current = stopGeneration
        stateLock.unlock()
        return current == gen
    }

    private func decrementPending() {
        pendingCondition.lock()
        pendingCount -= 1
        if pendingCount <= 0 {
            pendingCount = 0
            pendingCondition.broadcast()
        }
        pendingCondition.unlock()
    }

    private func playOneItem(_ item: PlayItem, generation gen: UInt64) {
        let semaphore: DispatchSemaphore

        sayLock.lock()
        do {
            _ = try obtainEngine(sampleRate: item.sampleRate, device: item.deviceID)
        } catch {
            sayLock.unlock()
            return
        }
        guard let playerNode = sayPlayerNode else {
            sayLock.unlock()
            return
        }

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(item.sampleRate),
            channels: 1,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(item.samples.count)
        ) else {
            sayLock.unlock()
            return
        }
        buffer.frameLength = AVAudioFrameCount(item.samples.count)

        let channelData = buffer.floatChannelData!
        item.samples.withUnsafeBufferPointer { src in
            channelData[0].update(from: src.baseAddress!, count: item.samples.count)
        }

        semaphore = DispatchSemaphore(value: 0)
        playerNode.stop()
        playerNode.scheduleBuffer(buffer) {
            semaphore.signal()
        }
        playerNode.play()
        sayLock.unlock()

        while true {
            let result = semaphore.wait(timeout: .now() + 0.05)
            if result == .success { break }
            if !isGenerationCurrent(gen) {
                sayLock.lock()
                sayPlayerNode?.stop()
                sayLock.unlock()
                return
            }
        }
    }

    // MARK: - Static helpers

    /// Get TTS voice availability as a JSON string for the given languages.
    ///
    /// - Parameters:
    ///   - languages: Comma-separated language tags (e.g. `"en_us,de"`).
    ///     Pass an empty string for all languages.
    ///   - options: Optional options (set `g2p_root` for accurate on-disk state).
    /// - Returns: JSON string mapping language tags to voice arrays.
    /// - Throws: `MoonshineError` on failure.
    public static func getVoices(
        languages: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        return try MoonshineAPI.shared.getTtsVoices(
            languages: languages,
            options: options
        )
    }

    /// Get TTS asset dependency keys as a JSON string for the given languages.
    ///
    /// - Parameters:
    ///   - languages: Comma-separated language tags. Pass an empty string for all.
    ///   - options: Optional options.
    /// - Returns: JSON array string of canonical asset keys.
    /// - Throws: `MoonshineError` on failure.
    public static func getDependencies(
        languages: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        return try MoonshineAPI.shared.getTtsDependencies(
            languages: languages,
            options: options
        )
    }

    /// Release all resources held by this synthesizer.
    public func close() {
        stateLock.lock()
        stopGeneration += 1
        stateLock.unlock()

        sayLock.lock()
        releaseEngine()
        sayLock.unlock()

        if handle >= 0 {
            api.freeTtsSynthesizer(handle)
            handle = -1
        }
    }

    // MARK: - Audio Engine Management

    private func obtainEngine(
        sampleRate: Int32,
        device: AudioDeviceID?
    ) throws -> AVAudioEngine {
        #if os(macOS)
        let deviceChanged = (device != sayCachedDeviceID)
        #else
        let deviceChanged = false
        #endif
        let rateChanged = (sampleRate != sayCachedSampleRate)

        if let engine = sayEngine, !deviceChanged, !rateChanged {
            return engine
        }

        releaseEngine()

        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()

        engine.attach(playerNode)

        #if os(macOS)
        if let deviceID = device {
            setOutputDevice(engine: engine, deviceID: deviceID)
        }
        sayCachedDeviceID = device
        #endif

        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        )!

        engine.connect(playerNode, to: engine.mainMixerNode, format: outputFormat)

        do {
            try engine.start()
        } catch {
            throw MoonshineError.custom(
                message: "Failed to start audio engine: \(error.localizedDescription)",
                code: -1
            )
        }

        sayEngine = engine
        sayPlayerNode = playerNode
        sayCachedSampleRate = sampleRate

        return engine
    }

    private func releaseEngine() {
        if let playerNode = sayPlayerNode {
            playerNode.stop()
        }
        if let engine = sayEngine {
            engine.stop()
        }
        sayPlayerNode = nil
        sayEngine = nil
        sayCachedSampleRate = 0
        #if os(macOS)
        sayCachedDeviceID = nil
        #endif
    }

    #if os(macOS)
    private func setOutputDevice(engine: AVAudioEngine, deviceID: AudioDeviceID) {
        let outputNode = engine.outputNode
        let audioUnit = outputNode.audioUnit!
        var deviceID = deviceID
        AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
    }
    #endif

    /// List available audio output devices on macOS.
    /// Returns an array of `(id, name)` tuples.
    #if os(macOS)
    public static func getAudioOutputDevices() -> [(id: AudioDeviceID, name: String)] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize
        )
        guard status == noErr, dataSize > 0 else { return [] }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &deviceIDs
        )
        guard status == noErr else { return [] }

        var results: [(id: AudioDeviceID, name: String)] = []
        for deviceID in deviceIDs {
            // Check if device has output channels
            var outputAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreamConfiguration,
                mScope: kAudioDevicePropertyScopeOutput,
                mElement: kAudioObjectPropertyElementMain
            )
            var outputSize: UInt32 = 0
            status = AudioObjectGetPropertyDataSize(
                deviceID, &outputAddress, 0, nil, &outputSize)
            guard status == noErr, outputSize > 0 else { continue }

            let bufferListPtr = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { bufferListPtr.deallocate() }
            status = AudioObjectGetPropertyData(
                deviceID, &outputAddress, 0, nil, &outputSize, bufferListPtr)
            guard status == noErr else { continue }

            let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPtr)
            let outputChannels = bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
            guard outputChannels > 0 else { continue }

            // Get device name
            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var name: Unmanaged<CFString>?
            var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            status = AudioObjectGetPropertyData(
                deviceID, &nameAddress, 0, nil, &nameSize, &name)
            let deviceName: String
            if status == noErr, let cfName = name?.takeUnretainedValue() {
                deviceName = cfName as String
            } else {
                deviceName = "Unknown"
            }

            results.append((id: deviceID, name: deviceName))
        }
        return results
    }
    #endif
}
