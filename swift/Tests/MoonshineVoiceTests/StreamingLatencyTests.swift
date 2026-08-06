import Foundation
import XCTest
#if os(iOS)
import UIKit
#else
import Darwin
#endif

@testable import MoonshineVoice

/// Measures Tiny / Small / Medium Streaming end-of-phrase latency with the same
/// metric as `core/benchmark` / the README table. Models are downloaded from
/// the CDN via ``AssetDownloader`` (network required).
///
/// Parseable summary for `scripts/test-mobile-latency.sh`:
/// `MOONSHINE_LATENCY platform=macos|ios device=... model=... avg_ms=...`
@available(iOS 15.0, macOS 12.0, *)
final class StreamingLatencyTests: XCTestCase {

    private struct Case {
        let modelName: String
        let arch: ModelArch
        let maxAvgLatencyMs: Double
    }

    #if os(iOS)
    private static let cases: [Case] = [
        Case(modelName: "tiny-streaming-en", arch: .tinyStreaming, maxAvgLatencyMs: 250),
        Case(modelName: "small-streaming-en", arch: .smallStreaming, maxAvgLatencyMs: 750),
        Case(modelName: "medium-streaming-en", arch: .mediumStreaming, maxAvgLatencyMs: 1400),
    ]
    #else
    private static let cases: [Case] = [
        Case(modelName: "tiny-streaming-en", arch: .tinyStreaming, maxAvgLatencyMs: 100),
        Case(modelName: "small-streaming-en", arch: .smallStreaming, maxAvgLatencyMs: 200),
        Case(modelName: "medium-streaming-en", arch: .mediumStreaming, maxAvgLatencyMs: 300),
    ]
    #endif

    private static let twoCitiesURL = URL(
        string: "https://github.com/moonshine-ai/moonshine/raw/main/test-assets/two_cities.wav")!

    func testStreamingLatencyTwoCities() async throws {
        let wavURL = try await Self.ensureTwoCitiesWav()
        let wavData = try loadWAVFile(wavURL.path)
        let device = Self.deviceLabel().replacingOccurrences(of: " ", with: "_")
        #if os(iOS)
        let platform = "ios"
        #else
        let platform = "macos"
        #endif

        for testCase in Self.cases {
            let transcriber = try await Transcriber.load(
                language: "en",
                modelArch: testCase.arch,
                onProgress: { progress in
                    if progress.bytesTotal > 0 {
                        let pct = 100.0 * Double(progress.bytesDownloaded)
                            / Double(progress.bytesTotal)
                        if progress.bytesDownloaded == progress.bytesTotal
                            || progress.bytesDownloaded % (1024 * 1024) < 256 * 1024
                        {
                            print(String(
                                format: "download %@ %d/%d %.0f%%",
                                progress.relativePath, progress.fileIndex,
                                progress.totalFiles, pct))
                        }
                    }
                })
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
                XCTFail("\(testCase.modelName) transcription error: \(heardError)")
                return
            }
            XCTAssertFalse(latencies.isEmpty, "\(testCase.modelName): expected completed lines")
            let lower = allText.lowercased()
            XCTAssertTrue(lower.contains("best of times"), "\(testCase.modelName) missing phrase")
            XCTAssertTrue(lower.contains("worst of times"), "\(testCase.modelName) missing phrase")

            let sum = latencies.reduce(0) { $0 + Int($1) }
            let avgMs = Double(sum) / Double(latencies.count)
            let summary = String(
                format: "MOONSHINE_LATENCY platform=%@ device=%@ model=%@ avg_ms=%.0f lines=%d wall_s=%.2f",
                platform, device, testCase.modelName, avgMs, latencies.count, wallSeconds)
            print(summary)
            fputs(summary + "\n", stderr)

            XCTAssertLessThanOrEqual(
                avgMs, testCase.maxAvgLatencyMs,
                String(
                    format: "%@ avg latency %.0fms exceeds regression ceiling %.0fms",
                    testCase.modelName, avgMs, testCase.maxAvgLatencyMs))
        }
    }

    private static func ensureTwoCitiesWav() async throws -> URL {
        if let bundled = try? TranscriberTests.getWAVFilePath("two_cities.wav") {
            return URL(fileURLWithPath: bundled)
        }
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("moonshine-two_cities.wav")
        if FileManager.default.fileExists(atPath: dest.path) {
            return dest
        }
        let (data, response) = try await URLSession.shared.data(from: twoCitiesURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(
                domain: "StreamingLatencyTests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "failed to download two_cities.wav"])
        }
        try data.write(to: dest, options: .atomic)
        return dest
    }

    private static func deviceLabel() -> String {
        #if os(iOS)
        return UIDevice.current.model
        #else
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let cpu = String(
            decoding: brand.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        if !cpu.isEmpty {
            return cpu.replacingOccurrences(of: " ", with: "_")
        }
        return "Mac"
        #endif
    }
}
