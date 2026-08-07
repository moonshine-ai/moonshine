import Foundation
import XCTest

@testable import MoonshineVoice

/// Covers the reference-clip capture used for voice cloning. These feed audio
/// in directly rather than opening a microphone, so they need no hardware.
final class VoiceCloneTests: XCTestCase {

    /// Real speech, which the detector should find a window in.
    private func speechSamples() throws -> ([Float], Int32) {
        let path = try TranscriberTests.getWAVFilePath("two_cities.wav")
        let wav = try loadWAVFile(path)
        return (wav.audioData, Int32(wav.sampleRate))
    }

    /// A loaded TTS whose synthesizer handle stays valid for the caller's
    /// lifetime. Returning only the Int32 handle would free the synthesizer in
    /// `deinit` and make `extractSpeechClip` / `VoiceClone` fail with
    /// `invalidHandle`.
    private func makeTts() throws -> TextToSpeech {
        let g2pRoot = "../core/moonshine-tts/data/"
        guard FileManager.default.fileExists(atPath: g2pRoot + "zipvoice/vocoder.ort") else {
            throw XCTSkip("zipvoice assets not available")
        }
        return try TextToSpeech(language: "en_us", g2pRoot: g2pRoot)
    }

    func testExtractsAClipFromSpeech() throws {
        let (samples, rate) = try speechSamples()
        let tts = try makeTts()
        let result = try MoonshineAPI.shared.extractSpeechClip(
            audioData: samples, sampleRate: rate, ttsSynthesizerHandle: tts.synthesizerHandle,
            clipDurationSeconds: 4,
            minimumSpeechSeconds: 2)

        let clip = try XCTUnwrap(result.audio, "should find speech in two_cities.wav")
        XCTAssertEqual(clip.count, 4 * 16000, "clip should be exactly the requested length")
        XCTAssertGreaterThanOrEqual(result.speechDuration, 2)
    }

    func testSilenceYieldsNoClip() throws {
        let tts = try makeTts()
        let silence = [Float](repeating: 0, count: 16000 * 6)
        let result = try MoonshineAPI.shared.extractSpeechClip(
            audioData: silence, sampleRate: 16000, ttsSynthesizerHandle: tts.synthesizerHandle,
            clipDurationSeconds: 4,
            minimumSpeechSeconds: 2)

        XCTAssertNil(result.audio, "silence should not produce a reference clip")
    }

    func testTooShortToFillAWindowYieldsNoClip() throws {
        let (samples, rate) = try speechSamples()
        let tts = try makeTts()
        let oneSecond = Array(samples.prefix(Int(rate)))
        let result = try MoonshineAPI.shared.extractSpeechClip(
            audioData: oneSecond, sampleRate: rate, ttsSynthesizerHandle: tts.synthesizerHandle,
            clipDurationSeconds: 4,
            minimumSpeechSeconds: 2)

        XCTAssertNil(result.audio)
    }

    func testIncrementalCaptureBecomesReady() throws {
        let (samples, rate) = try speechSamples()
        let tts = try makeTts()
        let clone = VoiceClone(ttsHandle: tts.synthesizerHandle)
        let readyCount = ReadyCounter()
        clone.onReady { readyCount.bump() }

        XCTAssertFalse(clone.isReady)

        // Feed the recording in quarter-second chunks, the way a capture
        // callback would.
        let chunk = Int(rate) / 4
        var offset = 0
        while offset < samples.count, !clone.isReady {
            let end = min(offset + chunk, samples.count)
            clone.addAudio(Array(samples[offset..<end]), sampleRate: rate)
            offset = end
        }

        XCTAssertTrue(clone.isReady, "should find a usable window in real speech")
        XCTAssertEqual(clone.audio?.count, 4 * 16000)
        XCTAssertEqual(clone.sampleRate, 16000)
        XCTAssertEqual(readyCount.value, 1, "onReady should fire exactly once")

        // Late subscribers still hear about a clip that already exists.
        let late = ReadyCounter()
        clone.onReady { late.bump() }
        XCTAssertEqual(late.value, 1)
    }

    func testResetDiscardsTheClip() throws {
        let (samples, rate) = try speechSamples()
        let tts = try makeTts()
        let clone = VoiceClone(ttsHandle: tts.synthesizerHandle)
        clone.addAudio(samples, sampleRate: rate)
        XCTAssertTrue(clone.isReady)

        clone.reset()

        XCTAssertFalse(clone.isReady)
        XCTAssertNil(clone.audio)
        XCTAssertEqual(clone.recordedSeconds, 0)
    }

    func testProgressReportsSpeechFound() throws {
        let (samples, rate) = try speechSamples()
        let tts = try makeTts()
        let clone = VoiceClone(ttsHandle: tts.synthesizerHandle)
        let updates = ProgressLog()
        clone.onProgress { recorded, speech in updates.add(recorded, speech) }

        clone.addAudio(samples, sampleRate: rate)

        XCTAssertFalse(updates.entries.isEmpty)
        XCTAssertGreaterThan(updates.entries.last?.speech ?? 0, 0)
    }

    private final class ReadyCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func bump() {
            lock.lock()
            count += 1
            lock.unlock()
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    private final class ProgressLog: @unchecked Sendable {
        private let lock = NSLock()
        private var log: [(recorded: Double, speech: Double)] = []
        func add(_ recorded: Double, _ speech: Double) {
            lock.lock()
            log.append((recorded, speech))
            lock.unlock()
        }
        var entries: [(recorded: Double, speech: Double)] {
            lock.lock()
            defer { lock.unlock() }
            return log
        }
    }
}
