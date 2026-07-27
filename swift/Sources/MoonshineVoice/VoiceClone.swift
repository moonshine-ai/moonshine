@preconcurrency import AVFoundation
import Foundation

/// Captures the short reference clip that zero-shot voice cloning needs.
///
/// ```swift
/// let clone = tts.startCloning()
/// clone.onReady { status.stringValue = "Got it — you can stop talking." }
/// try await clone.fromMicrophone()
/// try await tts.cloneFrom(clone)
/// ```
///
/// Finding a usable clip means locating a window of the recording that is mostly
/// speech rather than silence or breathing. That search runs in the core, so the
/// browser, iOS and Android bindings all agree on what a good clip looks like.
/// No model download is involved: the voice-activity detector is compiled into
/// the library.
public final class VoiceClone: @unchecked Sendable {
    /// Sample rate of the clip handed back by ``audio``.
    public static let clipSampleRate: Int32 = 16000
    /// How much new audio to accumulate between speech searches.
    private static let searchIntervalSeconds = 0.25
    /// Give up looking for a good window after this much recording.
    public static let defaultMaxRecordSeconds = 20.0

    private let clipDurationSeconds: Float
    private let minimumSpeechSeconds: Float
    private let api = MoonshineAPI.shared

    private let lock = NSLock()
    private var recording: [Float] = []
    private var recordingSampleRate: Int32 = VoiceClone.clipSampleRate
    private var samplesSinceSearch = 0
    private var clip: [Float]?
    private var speech: Float = 0
    private var readyHandlers: [() -> Void] = []
    private var progressHandlers: [(Double, Double) -> Void] = []

    private var engine: AVAudioEngine?

    public init(clipDurationSeconds: Float = 4, minimumSpeechSeconds: Float = 2) {
        self.clipDurationSeconds = clipDurationSeconds
        self.minimumSpeechSeconds = minimumSpeechSeconds
    }

    deinit {
        stopCapture()
    }

    /// Fires once, as soon as enough speech has been captured.
    @discardableResult
    public func onReady(_ handler: @escaping () -> Void) -> Self {
        lock.lock()
        if clip != nil {
            lock.unlock()
            handler()
        } else {
            readyHandlers.append(handler)
            lock.unlock()
        }
        return self
    }

    /// Reports how long the caller has been recording and how much of the best
    /// window so far was speech, both in seconds.
    @discardableResult
    public func onProgress(_ handler: @escaping (Double, Double) -> Void) -> Self {
        lock.lock()
        progressHandlers.append(handler)
        lock.unlock()
        return self
    }

    /// True once ``audio`` holds a usable reference clip.
    public var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return clip != nil
    }

    /// The captured clip (16 kHz mono), or nil until ``isReady``.
    public var audio: [Float]? {
        lock.lock()
        defer { lock.unlock() }
        return clip
    }

    public var sampleRate: Int32 {
        return Self.clipSampleRate
    }

    /// Speech found in the best window so far, in seconds.
    public var speechSeconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return Double(speech)
    }

    public var recordedSeconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return Double(recording.count) / Double(recordingSampleRate)
    }

    /// Feeds captured audio in. Call this from your own audio pipeline; the
    /// search for a usable window runs a few times a second rather than on every
    /// chunk.
    public func addAudio(_ pcm: [Float], sampleRate: Int32) {
        lock.lock()
        guard clip == nil, !pcm.isEmpty, sampleRate > 0 else {
            lock.unlock()
            return
        }
        if sampleRate != recordingSampleRate {
            // Mixed rates in one buffer would make the clip come out at the
            // wrong speed, so a change starts the recording over.
            recording.removeAll(keepingCapacity: true)
            recordingSampleRate = sampleRate
            samplesSinceSearch = 0
        }
        recording.append(contentsOf: pcm)
        samplesSinceSearch += pcm.count
        let due = Double(samplesSinceSearch) >= Self.searchIntervalSeconds * Double(sampleRate)
        if due {
            samplesSinceSearch = 0
        }
        lock.unlock()
        if due {
            search()
        }
    }

    /// Opens the microphone and records until there is enough speech, or until
    /// `maxSeconds` have passed. Returns the clip, which is also available from
    /// ``audio``.
    @discardableResult
    public func fromMicrophone(maxSeconds: Double = VoiceClone.defaultMaxRecordSeconds) async throws
        -> [Float]
    {
        try ensureMicrophonePermission()
        try startCapture()
        defer {
            stopCapture()
            releaseMicrophoneSession()
        }

        let deadline = Date().addingTimeInterval(maxSeconds)
        while !isReady {
            if Date() >= deadline {
                // Out of patience: take the best window we have, even a quiet one.
                search(acceptAnything: true)
                break
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        guard let audio else {
            throw MoonshineError.custom(
                message:
                    "No speech detected in \(Int(maxSeconds))s of recording. Try again somewhere quieter.",
                code: -1)
        }
        return audio
    }

    /// Stops an in-flight ``fromMicrophone(maxSeconds:)`` capture.
    public func cancel() {
        stopCapture()
    }

    /// Throws away everything captured so far.
    public func reset() {
        lock.lock()
        recording.removeAll(keepingCapacity: false)
        samplesSinceSearch = 0
        clip = nil
        speech = 0
        lock.unlock()
    }

    // MARK: - Internals

    private func search(acceptAnything: Bool = false) {
        lock.lock()
        guard clip == nil else {
            lock.unlock()
            return
        }
        let samples = recording
        let rate = recordingSampleRate
        lock.unlock()

        guard
            let result = try? api.extractSpeechClip(
                audioData: samples,
                sampleRate: rate,
                clipDurationSeconds: clipDurationSeconds,
                minimumSpeechSeconds: acceptAnything ? 0 : minimumSpeechSeconds)
        else {
            return
        }

        lock.lock()
        speech = result.speechDuration
        let recorded = Double(samples.count) / Double(rate)
        let speechSoFar = Double(result.speechDuration)
        let progress = progressHandlers
        var ready: [() -> Void] = []
        if let audio = result.audio, !audio.isEmpty {
            clip = audio
            ready = readyHandlers
            readyHandlers.removeAll()
        }
        lock.unlock()

        for handler in progress {
            handler(recorded, speechSoFar)
        }
        for handler in ready {
            handler()
        }
    }

    private func startCapture() throws {
        guard engine == nil else { return }
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        let inputRate = Int32(inputFormat.sampleRate)

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) {
            [weak self] buffer, _ in
            guard let self, let channelData = buffer.floatChannelData else { return }
            let frames = Int(buffer.frameLength)
            guard frames > 0 else { return }
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frames))
            self.addAudio(samples, sampleRate: inputRate)
        }

        try engine.start()
        self.engine = engine
    }

    private func stopCapture() {
        guard let engine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        self.engine = nil
    }
}
