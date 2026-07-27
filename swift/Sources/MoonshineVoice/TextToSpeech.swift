import AVFoundation
import Foundation

#if os(macOS)
import CoreAudio
#else
/// Placeholder type on non-macOS platforms where CoreAudio's AudioDeviceID is unavailable.
public typealias AudioDeviceID = UInt32
#endif

/// On-device text-to-speech using the Moonshine native API (Kokoro / Piper / ZipVoice).
///
/// ```swift
/// let tts = TextToSpeech()
/// try await tts.load()
/// try await tts.say("Hello world!")
/// ```
///
/// Cloning a voice is two more lines, and the awkward parts — finding the speech
/// in the reference recording, and transcribing it so the vocoder knows what was
/// said — happen inside the library:
///
/// ```swift
/// try await tts.cloneFrom(url)
/// try await tts.say("Hello in your voice!")
/// ```
///
/// ``say(_:)`` plays audio and returns when playback finishes; ``synthesize(_:)``
/// returns the raw PCM instead, for callers doing their own mixing or encoding.
public class TextToSpeech: @unchecked Sendable {
    private let api: MoonshineAPI
    private var handle: Int32 = -1
    private var _language: String
    /// Retained reference-clip PCM buffer for the ZipVoice-from-memory path (native layer does not copy it).
    private var cloneBuffer: UnsafeMutablePointer<UInt8>?
    private var cloneBufferCount: Int = 0

    // Deferred configuration, applied by ``load()``.
    private var voiceId: String?
    private var assetDirectory: URL?
    private var extraOptions: [TranscriberOption] = []
    private var progressHandler: (@Sendable (Double, String) -> Void)?
    private var cloningWanted = false

    /// The clip the current voice was cloned from, if any.
    private var cloneSamples: [Float]?
    private var cloneTranscript: String?
    /// Loaded lazily, the first time a clone clip needs transcribing.
    private var clipTranscriber: Transcriber?

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
    public static let moonshineHeaderVersion: Int32 = 30000

    /// Canonical asset key under which a ZipVoice clone reference clip is supplied.
    private static let cloneAudioKey = "zipvoice/clone_audio"
    /// The only engine that can clone an arbitrary reference voice.
    private static let cloneVoice = "zipvoice"
    /// Reference clips are resampled to this rate before cloning.
    private static let cloneSampleRate: Int32 = 16000

    /// Creates a synthesizer that has not loaded any assets yet.
    ///
    /// Configure it with the chainable setters below, then `try await load()`.
    public init() {
        self.api = MoonshineAPI.shared
        self._language = "en"
    }

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

    /// Initialize a ZipVoice synthesizer that clones ``clonePCM`` (mono float samples in -1..1).
    ///
    /// - Parameters:
    ///   - language: Moonshine language tag (English only for now, e.g. `en_us`).
    ///   - g2pRoot: Directory containing the ZipVoice model assets.
    ///   - clonePCM: Reference clip to clone as mono float PCM.
    ///   - cloneSampleRate: Sample rate of ``clonePCM``.
    ///   - cloneTranscript: Transcript of the clip (recommended; may be nil).
    ///   - options: Additional options.
    /// - Throws: `MoonshineError` if the synthesizer cannot be created.
    public init(
        language: String,
        g2pRoot: String,
        clonePCM: [Float],
        cloneSampleRate: Int32 = 24000,
        cloneTranscript: String? = nil,
        options: [TranscriberOption]? = nil
    ) throws {
        self.api = MoonshineAPI.shared
        self._language = language

        // Convert to little-endian float32 bytes in a stable heap buffer that outlives the synthesizer
        // (the native layer keeps a pointer to it rather than copying).
        let byteCount = clonePCM.count * MemoryLayout<Float>.size
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: max(byteCount, 1))
        clonePCM.withUnsafeBytes { src in
            if let base = src.baseAddress, byteCount > 0 {
                buffer.update(from: base.assumingMemoryBound(to: UInt8.self), count: byteCount)
            }
        }
        self.cloneBuffer = buffer
        self.cloneBufferCount = byteCount

        var allOptions = options ?? []
        allOptions.append(TranscriberOption(name: "g2p_root", value: g2pRoot))
        allOptions.append(TranscriberOption(name: "voice", value: "zipvoice"))
        allOptions.append(TranscriberOption(
            name: "zipvoice_clone_sample_rate", value: String(cloneSampleRate)))
        if let transcript = cloneTranscript, !transcript.isEmpty {
            allOptions.append(TranscriberOption(name: "zipvoice_clone_transcript", value: transcript))
        }

        do {
            self.handle = try api.createTtsSynthesizerFromMemory(
                language: language,
                filenames: ["zipvoice/clone_audio"],
                memoryPtrs: [UnsafePointer(buffer)],
                memorySizes: [UInt64(byteCount)],
                options: allOptions,
                moonshineVersion: TextToSpeech.moonshineHeaderVersion
            )
        } catch {
            buffer.deallocate()
            self.cloneBuffer = nil
            self.cloneBufferCount = 0
            throw error
        }
    }

    deinit {
        close()
    }

    /// The language tag this synthesizer was created with.
    public var languageTag: String {
        return _language
    }

    // MARK: - Configuration

    /// Synthesis language, e.g. `"en"` or `"en_us"`. Defaults to `"en"`.
    @discardableResult
    public func language(_ code: String) -> Self {
        _language = code
        return self
    }

    /// Voice id, e.g. `"kokoro_af_heart"`. Defaults to the engine's own default.
    @discardableResult
    public func voice(_ id: String) -> Self {
        voiceId = id
        return self
    }

    /// Loads voice assets from a directory you supply rather than the Moonshine CDN.
    @discardableResult
    public func modelsFrom(_ directory: URL) -> Self {
        assetDirectory = directory
        return self
    }

    /// Fetches the cloning engine during ``load()`` rather than on the first
    /// ``cloneFrom(_:transcript:)``, so the first clone is quick.
    @discardableResult
    public func cloning(_ enabled: Bool = true) -> Self {
        cloningWanted = enabled
        return self
    }

    /// Asset download progress, as a `0..1` fraction plus the file being fetched.
    @discardableResult
    public func onProgress(_ handler: @escaping @Sendable (Double, String) -> Void) -> Self {
        progressHandler = handler
        return self
    }

    /// Escape hatch for options the chainable setters don't cover.
    @discardableResult
    public func options(_ options: [TranscriberOption]) -> Self {
        extraOptions.append(contentsOf: options)
        return self
    }

    // MARK: - Loading

    /// Downloads the voice assets if needed and prepares the synthesizer.
    @available(iOS 15.0, macOS 12.0, *)
    public func load() async throws {
        if cloningWanted, cloneSamples == nil {
            // The cloning engine can't exist until there's a voice to clone, so
            // all load() can usefully do is fetch its assets into the cache that
            // the first cloneFrom() reads from.
            _ = try await ensureAssets(voice: Self.cloneVoice)
            return
        }
        try await build(voice: voiceId)
    }

    /// True once a voice has been cloned into this synthesizer.
    public var isCloned: Bool {
        return cloneSamples != nil
    }

    /// Clones the voice in the recording at `url` and uses it for subsequent
    /// synthesis. `url` may be a local file or a remote WAV.
    ///
    /// The library trims the recording down to a few seconds of actual speech and
    /// transcribes that clip for the vocoder, downloading a small speech-to-text
    /// model the first time it needs to. Callers who already know what was said
    /// can skip that by passing `transcript`.
    @available(iOS 15.0, macOS 12.0, *)
    public func cloneFrom(_ url: URL, transcript: String? = nil) async throws {
        let wav: WAVData
        if url.isFileURL {
            wav = try loadWAVFile(url.path)
        } else {
            let (data, _) = try await URLSession.shared.data(from: url)
            let temporary = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".wav")
            try data.write(to: temporary)
            defer { try? FileManager.default.removeItem(at: temporary) }
            wav = try loadWAVFile(temporary.path)
        }
        try await cloneFrom(
            samples: wav.audioData, sampleRate: Int32(wav.sampleRate), transcript: transcript)
    }

    /// Clones the voice in the recording at `path`.
    @available(iOS 15.0, macOS 12.0, *)
    public func cloneFrom(_ path: String, transcript: String? = nil) async throws {
        try await cloneFrom(URL(fileURLWithPath: path), transcript: transcript)
    }

    /// Clones the voice in `samples` (mono float PCM in -1..1).
    @available(iOS 15.0, macOS 12.0, *)
    public func cloneFrom(
        samples: [Float], sampleRate: Int32, transcript: String? = nil
    ) async throws {
        let clip = try clipForCloning(samples: samples, sampleRate: sampleRate)
        cloneSamples = clip
        if let transcript {
            cloneTranscript = transcript
        } else {
            cloneTranscript = await transcribeClip(clip)
        }
        try await build(voice: Self.cloneVoice)
    }

    /// Clones the voice captured by a ``VoiceClone``.
    @available(iOS 15.0, macOS 12.0, *)
    public func cloneFrom(_ clone: VoiceClone, transcript: String? = nil) async throws {
        guard let audio = clone.audio else {
            throw MoonshineError.custom(
                message:
                    "That VoiceClone has not captured enough speech yet — wait for onReady.",
                code: -1)
        }
        try await cloneFrom(
            samples: audio, sampleRate: clone.sampleRate, transcript: transcript)
    }

    /// Starts capturing a reference voice from the microphone, for cloning.
    /// The returned object reports when it has heard enough.
    public func startCloning(
        clipDurationSeconds: Float = 4, minimumSpeechSeconds: Float = 2
    ) -> VoiceClone {
        return VoiceClone(
            clipDurationSeconds: clipDurationSeconds,
            minimumSpeechSeconds: minimumSpeechSeconds)
    }

    /// Synthesize text to mono PCM float samples and sample rate, without
    /// playing it. Use ``say(_:options:)`` to hear it instead.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - options: Optional per-call options (e.g. `speed`).
    /// - Returns: A ``TtsSynthesisResult`` with PCM samples and sample rate.
    /// - Throws: `MoonshineError` if synthesis fails.
    public func synthesize(
        _ text: String,
        options: [TranscriberOption]? = nil
    ) throws -> TtsSynthesisResult {
        guard handle >= 0 else {
            throw MoonshineError.custom(
                message: cloningWanted
                    ? "Call cloneFrom() before synthesizing with a cloned voice."
                    : "Call load() before synthesizing.",
                code: -1)
        }
        return try api.textToSpeech(
            ttsHandle: handle,
            text: text,
            options: options
        )
    }

    /// Synthesize speech directly from IPA phonemes, skipping grapheme-to-phoneme conversion.
    ///
    /// - Parameters:
    ///   - phonemes: An IPA phoneme string, as produced by the `moonshine_text_to_phonemes`
    ///     C API. Passing the phonemes for the same language yields audio equivalent to
    ///     ``synthesize(text:options:)`` on the original text, but lets you inspect or edit the
    ///     phonemes in between (e.g. to fix a name's pronunciation).
    ///   - options: Optional per-call options (e.g. `speed`).
    /// - Returns: A ``TtsSynthesisResult`` with PCM samples and sample rate.
    /// - Throws: `MoonshineError` if synthesis fails.
    public func synthesizeFromPhonemes(
        phonemes: String,
        options: [TranscriberOption]? = nil
    ) throws -> TtsSynthesisResult {
        return try api.phonemesToSpeech(
            ttsHandle: handle,
            phonemes: phonemes,
            options: options
        )
    }

    // MARK: - say / stop / wait / isTalking

    /// Speaks `text` out loud, returning once playback finishes.
    ///
    /// Utterances are played in the order they were requested; synthesis of the
    /// next one is pipelined with playback of the current one, so several
    /// concurrent `say` calls still come out in order without gaps. Call
    /// ``stop()`` to cancel everything queued and halt the audio playing now,
    /// which makes the pending calls return early.
    ///
    /// - Parameters:
    ///   - text: The text to speak.
    ///   - options: Optional per-call synthesis options (e.g. `speed`).
    public func say(
        _ text: String,
        options: [TranscriberOption]? = nil
    ) async throws {
        try await speak(text, deviceID: nil, options: options)
    }

    /// Speaks each string in order, returning once the last one finishes.
    public func say(
        _ texts: [String],
        options: [TranscriberOption]? = nil
    ) async throws {
        for text in texts {
            try await speak(text, deviceID: nil, options: options)
        }
    }

    /// Queues `text` without waiting for it, for callers that just want the
    /// audio to start and have somewhere else to be.
    public func sayInBackground(
        _ text: String,
        options: [TranscriberOption]? = nil
    ) {
        enqueueSay(text: text, deviceID: nil, options: options)
    }

    #if os(macOS)
    /// Speaks `text` on a specific output device, returning once playback finishes.
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
    ) async throws {
        try await speak(text, deviceID: device, options: options)
    }

    /// Speaks each string in order on a specific output device.
    public func say(
        _ texts: [String],
        device: AudioDeviceID?,
        options: [TranscriberOption]? = nil
    ) async throws {
        for text in texts {
            try await speak(text, deviceID: device, options: options)
        }
    }
    #endif

    private func speak(
        _ text: String, deviceID: AudioDeviceID?, options: [TranscriberOption]?
    ) async throws {
        guard !text.isEmpty else { return }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            enqueueSay(text: text, deviceID: deviceID, options: options) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

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

    // MARK: - Load internals

    /// (Re)creates the native synthesizer for `voice`, downloading its assets.
    ///
    /// The old engine is only torn down once the new one exists, so a failed
    /// clone leaves the caller with a working synthesizer.
    @available(iOS 15.0, macOS 12.0, *)
    private func build(voice: String?) async throws {
        let directory = try await ensureAssets(voice: voice)
        var allOptions = extraOptions
        allOptions.append(TranscriberOption(name: "g2p_root", value: directory.path))

        if let clip = cloneSamples {
            allOptions.append(TranscriberOption(name: "voice", value: Self.cloneVoice))
            allOptions.append(
                TranscriberOption(
                    name: "zipvoice_clone_sample_rate", value: String(Self.cloneSampleRate)))
            if let cloneTranscript, !cloneTranscript.isEmpty {
                allOptions.append(
                    TranscriberOption(name: "zipvoice_clone_transcript", value: cloneTranscript))
            }

            let byteCount = clip.count * MemoryLayout<Float>.size
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: max(byteCount, 1))
            clip.withUnsafeBytes { source in
                if let base = source.baseAddress, byteCount > 0 {
                    buffer.update(from: base.assumingMemoryBound(to: UInt8.self), count: byteCount)
                }
            }
            let next: Int32
            do {
                next = try api.createTtsSynthesizerFromMemory(
                    language: _language,
                    filenames: [Self.cloneAudioKey],
                    memoryPtrs: [UnsafePointer(buffer)],
                    memorySizes: [UInt64(byteCount)],
                    options: allOptions,
                    moonshineVersion: Self.moonshineHeaderVersion)
            } catch {
                buffer.deallocate()
                throw error
            }
            replaceHandle(next)
            cloneBuffer?.deallocate()
            cloneBuffer = buffer
            cloneBufferCount = byteCount
            return
        }

        if let voice {
            allOptions.append(TranscriberOption(name: "voice", value: voice))
        }
        let next = try api.createTtsSynthesizerFromFiles(
            language: _language,
            options: allOptions,
            moonshineVersion: Self.moonshineHeaderVersion)
        replaceHandle(next)
    }

    private func replaceHandle(_ next: Int32) {
        if handle >= 0 {
            api.freeTtsSynthesizer(handle)
        }
        handle = next
    }

    @available(iOS 15.0, macOS 12.0, *)
    private func ensureAssets(voice: String?) async throws -> URL {
        if let assetDirectory {
            return assetDirectory
        }
        let spec = ModelSpec.tts(language: _language, voice: voice)
        let directory = try ModelCache.directory(for: spec)
        _ = try await AssetDownloader().ensureModelPresent(
            root: directory, spec: spec, onProgress: fractionReporter(progressHandler))
        return directory
    }

    /// Trims a reference recording to the few seconds of speech ZipVoice wants,
    /// resampling to 16 kHz on the way.
    private func clipForCloning(samples: [Float], sampleRate: Int32) throws -> [Float] {
        if sampleRate == Self.cloneSampleRate, samples.count <= Int(Self.cloneSampleRate) * 10 {
            return samples
        }
        if let audio = try api.extractSpeechClip(
            audioData: samples, sampleRate: sampleRate, clipDurationSeconds: 4,
            minimumSpeechSeconds: 2
        ).audio {
            return audio
        }
        // Nothing clearly speech-like. Rather than refuse outright, take the best
        // window the detector found — a poor clone beats no clone for a caller
        // who explicitly handed us this recording.
        if let audio = try api.extractSpeechClip(
            audioData: samples, sampleRate: sampleRate, clipDurationSeconds: 4,
            minimumSpeechSeconds: 0
        ).audio {
            return audio
        }
        throw MoonshineError.custom(
            message: "Couldn't find enough speech in that recording to clone from.", code: -1)
    }

    /// Transcribes a clone clip so the vocoder knows what the reference voice
    /// said. The speech-to-text model this needs is an implementation detail, so
    /// it is loaded here rather than being the caller's problem. Cloning still
    /// works without a transcript, just less faithfully, so a failure here is
    /// swallowed rather than sinking the whole operation.
    @available(iOS 15.0, macOS 12.0, *)
    private func transcribeClip(_ clip: [Float]) async -> String? {
        do {
            if clipTranscriber == nil {
                clipTranscriber = try await Transcriber.load(
                    language: sttLanguage(for: _language),
                    modelArch: .base,
                    onProgress: fractionReporter(progressHandler))
            }
            let transcript = try clipTranscriber?.transcribeWithoutStreaming(
                audioData: clip, sampleRate: Self.cloneSampleRate)
            let text =
                transcript?.lines.map { $0.text }.joined(separator: " ").trimmingCharacters(
                    in: .whitespacesAndNewlines) ?? ""
            return text.isEmpty ? nil : text
        } catch {
            return nil
        }
    }

    // MARK: - Queue internals

    private func enqueueSay(
        text: String,
        deviceID: AudioDeviceID?,
        options: [TranscriberOption]?,
        completion: (@Sendable (Error?) -> Void)? = nil
    ) {
        pendingCondition.lock()
        pendingCount += 1
        pendingCondition.unlock()

        stateLock.lock()
        let gen = stopGeneration
        stateLock.unlock()

        synthQueue.async { [self] in
            // A bumped generation means stop() ran, which is a cancellation
            // rather than a failure, so the waiting caller just returns.
            guard self.isGenerationCurrent(gen) else {
                self.finish(completion, error: nil)
                return
            }

            let result: TtsSynthesisResult
            do {
                result = try self.synthesize(text, options: options)
            } catch {
                self.finish(completion, error: error)
                return
            }
            guard result.sampleRateHz > 0, !result.samples.isEmpty else {
                self.finish(completion, error: nil)
                return
            }

            guard self.isGenerationCurrent(gen) else {
                self.finish(completion, error: nil)
                return
            }

            let item = PlayItem(
                samples: result.samples,
                sampleRate: result.sampleRateHz,
                deviceID: deviceID
            )

            self.playbackQueue.async { [self] in
                defer { self.finish(completion, error: nil) }
                guard self.isGenerationCurrent(gen) else { return }
                self.playOneItem(item, generation: gen)
            }
        }
    }

    private func finish(_ completion: (@Sendable (Error?) -> Void)?, error: Error?) {
        decrementPending()
        completion?(error)
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
        clipTranscriber?.close()
        clipTranscriber = nil
        if let buffer = cloneBuffer {
            buffer.deallocate()
            cloneBuffer = nil
            cloneBufferCount = 0
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

/// TTS languages are regional (`en_us`); speech-to-text ones are not (`en`).
private func sttLanguage(for ttsLanguage: String) -> String {
    let base = ttsLanguage.split(whereSeparator: { $0 == "_" || $0 == "-" }).first
    return base.map(String.init) ?? "en"
}
