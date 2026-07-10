import Foundation
import XCTest

@testable import MoonshineVoice

/// Regression test for issue #196, mirroring the Python
/// ``test_mic_transcriber_threading`` test.
///
/// ``MicTranscriber`` feeds captured audio to the stream from the
/// ``AVAudioEngine`` tap, which runs on a high-priority audio thread. The
/// stream performs the (slow) transcription every ``updateInterval``. If that
/// runs inline on the capture thread, the callback blocks for the full
/// inference time and the capture buffer overflows -- dropped audio, the same
/// defect the Python side hit on a Raspberry Pi.
///
/// ``FakeStream`` stands in for the native-backed ``Stream``: ``addAudio`` is
/// cheap except that every ``updateInterval`` of audio it blocks for
/// ``slowUpdate`` to emulate a slow on-device inference. ``feedCapturedAudio``
/// is driven directly at the real-time block cadence (no microphone, model, or
/// audio engine), timing how long each call takes to return.
final class MicTranscriberThreadingTests: XCTestCase {

    private static let sampleRate: Int32 = 16000
    private static let blockSize = 1024
    // Wall-clock span of one captured block. A capture callback that takes
    // longer than this cannot keep up with real-time capture.
    private static var blockPeriod: Double { Double(blockSize) / Double(sampleRate) }
    private static let updateInterval = 0.1
    // Emulated inference time per update; well above blockPeriod so an inline
    // update unambiguously blows the real-time budget.
    private static let slowUpdate = 0.25
    private static let numBlocks = 10

    /// Test double for ``Stream`` with a deterministically slow update.
    final class FakeStream: TranscriptionStream {
        private let updateInterval: Double
        private let slowUpdate: Double

        private let lock = NSLock()
        private var streamTime = 0.0
        private var lastUpdateTime = 0.0
        private var totalSamples_ = 0
        private var addAudioCalls_ = 0
        private var addAudioThreads_: [Thread] = []
        private var blockingUpdateThreads_: [Thread] = []

        init(updateInterval: Double, slowUpdate: Double) {
            self.updateInterval = updateInterval
            self.slowUpdate = slowUpdate
        }

        func start() throws {}
        func close() {}

        // The real stream flushes any trailing audio on stop.
        func stop() throws { runUpdate() }

        func addAudio(_ audioData: [Float], sampleRate: Int32) throws {
            lock.lock()
            addAudioCalls_ += 1
            totalSamples_ += audioData.count
            addAudioThreads_.append(Thread.current)
            streamTime += Double(audioData.count) / Double(sampleRate)
            let shouldUpdate = streamTime - lastUpdateTime >= updateInterval
            if shouldUpdate { lastUpdateTime = streamTime }
            lock.unlock()

            // Emulate the blocking transcription outside the lock.
            if shouldUpdate { runUpdate() }
        }

        private func runUpdate() {
            lock.lock()
            blockingUpdateThreads_.append(Thread.current)
            lock.unlock()
            Thread.sleep(forTimeInterval: slowUpdate)
        }

        func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {}
        func addListener(_ listener: TranscriptEventListener) {}
        func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {}
        func removeListener(_ listener: TranscriptEventListener) {}
        func removeAllListeners() {}

        // Thread-safe snapshots for assertions.
        var totalSamples: Int { lock.lock(); defer { lock.unlock() }; return totalSamples_ }
        var addAudioCalls: Int { lock.lock(); defer { lock.unlock() }; return addAudioCalls_ }
        var addAudioThreads: [Thread] {
            lock.lock(); defer { lock.unlock() }; return addAudioThreads_
        }
        var blockingUpdateThreads: [Thread] {
            lock.lock(); defer { lock.unlock() }; return blockingUpdateThreads_
        }
    }

    func testCaptureCallbackIsNotBlockedByTranscription() throws {
        let fake = FakeStream(
            updateInterval: Self.updateInterval, slowUpdate: Self.slowUpdate)
        let mic = try MicTranscriber(
            testStream: fake, sampleRate: Double(Self.sampleRate))

        let feedThread = Thread.current
        var callbackDurations: [Double] = []

        for _ in 0..<Self.numBlocks {
            let block = [Float](repeating: 0.01, count: Self.blockSize)
            let start = DispatchTime.now().uptimeNanoseconds
            mic.feedCapturedAudio(block, sampleRate: Self.sampleRate)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            callbackDurations.append(elapsed)
            // Maintain a real-time-ish cadence for the next block.
            let remaining = Self.blockPeriod - elapsed
            if remaining > 0 { Thread.sleep(forTimeInterval: remaining) }
        }

        try mic.stop()
        mic.close()

        let slowest = callbackDurations.max() ?? 0

        // Core assertion: no capture callback may exceed the real-time budget
        // for one audio block. With transcription running inline, the callbacks
        // that trigger an update block ~slowUpdate and this fails.
        XCTAssertLessThan(
            slowest, Self.blockPeriod,
            "slowest capture callback took \(Int(slowest * 1000)) ms, exceeding the "
                + "\(Int(Self.blockPeriod * 1000)) ms real-time budget for one block -- "
                + "transcription is running on the capture thread (see issue #196)")

        // A blocking update must have happened (otherwise the slow path was
        // never exercised), just never on the capture thread.
        XCTAssertFalse(
            fake.blockingUpdateThreads.isEmpty, "no transcription update ran")
        XCTAssertFalse(
            fake.addAudioThreads.contains { $0 === feedThread },
            "transcription ran on the capture callback thread")

        // The decoupling must not drop audio.
        XCTAssertEqual(fake.totalSamples, Self.numBlocks * Self.blockSize)

        // The backlog that builds while a slow transcription runs should be
        // coalesced into fewer add_audio calls than captured blocks.
        XCTAssertLessThan(fake.addAudioCalls, Self.numBlocks)
    }
}
