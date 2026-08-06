import XCTest
import UIKit
import MoonshineVoice

/// On-device Tiny / Small / Medium Streaming latency. Models download from the
/// CDN (network required); only `two_cities.wav` may be bundled.
@available(iOS 15.0, *)
final class StreamingLatencyTests: XCTestCase {
    private struct Case {
        let modelName: String
        let arch: ModelArch
        let maxAvgLatencyMs: Double
    }

    private static let cases: [Case] = [
        Case(modelName: "tiny-streaming-en", arch: .tinyStreaming, maxAvgLatencyMs: 250),
        Case(modelName: "small-streaming-en", arch: .smallStreaming, maxAvgLatencyMs: 750),
        Case(modelName: "medium-streaming-en", arch: .mediumStreaming, maxAvgLatencyMs: 1400),
    ]

    private static let twoCitiesURL = URL(
        string: "https://github.com/moonshine-ai/moonshine/raw/main/test-assets/two_cities.wav")!

    func testStreamingLatencyTwoCities() async throws {
        let wavURL = try await Self.ensureTwoCitiesWav()
        let wavData = try loadWAVFile(wavURL.path)
        let device = UIDevice.current.model.replacingOccurrences(of: " ", with: "_")

        for testCase in Self.cases {
            let transcriber = try await Transcriber.load(
                language: "en", modelArch: testCase.arch)
            defer { transcriber.close() }

            var latencies: [UInt32] = []
            var allText = ""
            var heardError: Error?

            try transcriber.addListener { event in
                if let completed = event as? LineCompleted {
                    latencies.append(completed.line.lastTranscriptionLatencyMs)
                    allText += completed.line.text + "\n"
                } else if let err = event as? TranscriptError {
                    heardError = err.error
                }
            }

            try transcriber.start()
            let chunkSize = max(1, Int(0.0214 * Double(wavData.sampleRate)))
            let wallStart = Date()
            var offset = 0
            while offset < wavData.audioData.count {
                let end = min(offset + chunkSize, wavData.audioData.count)
                try transcriber.addAudio(
                    Array(wavData.audioData[offset..<end]),
                    sampleRate: Int32(wavData.sampleRate))
                offset = end
            }
            try transcriber.stop()
            let wallSeconds = Date().timeIntervalSince(wallStart)

            if let heardError {
                XCTFail("\(testCase.modelName) error: \(heardError)")
                return
            }
            XCTAssertFalse(latencies.isEmpty, "\(testCase.modelName): expected lines")
            let lower = allText.lowercased()
            XCTAssertTrue(lower.contains("best of times"))
            XCTAssertTrue(lower.contains("worst of times"))

            let sum = latencies.reduce(0) { $0 + Int($1) }
            let avgMs = Double(sum) / Double(latencies.count)
            let summary = String(
                format: "MOONSHINE_LATENCY platform=ios device=%@ model=%@ avg_ms=%.0f lines=%d wall_s=%.2f",
                device, testCase.modelName, avgMs, latencies.count, wallSeconds)
            print(summary)
            fputs(summary + "\n", stderr)

            XCTAssertLessThanOrEqual(
                avgMs, testCase.maxAvgLatencyMs,
                String(format: "%@ avg %.0fms > ceiling %.0fms",
                       testCase.modelName, avgMs, testCase.maxAvgLatencyMs))
        }
    }

    private static func ensureTwoCitiesWav() async throws -> URL {
        if let url = Bundle.main.url(forResource: "two_cities", withExtension: "wav") {
            return url
        }
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("moonshine-two_cities.wav")
        if FileManager.default.fileExists(atPath: dest.path) {
            return dest
        }
        let (data, response) = try await URLSession.shared.data(from: twoCitiesURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw XCTSkip("Could not download two_cities.wav")
        }
        try data.write(to: dest, options: .atomic)
        return dest
    }
}
