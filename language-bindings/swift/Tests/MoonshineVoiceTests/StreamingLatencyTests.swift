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
///
/// Set `MOONSHINE_KEYTERMS` (comma-separated, and optionally
/// `MOONSHINE_KEYTERM_BOOST`) to measure the same latency with contextual
/// biasing switched on, so its per-token cost can be compared against a
/// baseline run on the same machine.
///
/// Set `MOONSHINE_LATENCY_OPTIONAL` to report a breached **iOS** ceiling as a
/// warning instead of a failure. macOS timings are always informational: they
/// are still measured and printed, but never fail the test. A release Mac is
/// usually hot from hours of building, and the same code has measured anywhere
/// from 95ms to 124ms for Tiny between runs on one machine — wider than any
/// useful ceiling. Grep the log for `MOONSHINE_LATENCY` afterwards.
@available(iOS 15.0, macOS 12.0, *)
final class StreamingLatencyTests: XCTestCase {

    private struct Case {
        let modelName: String
        let arch: ModelArch
        let maxAvgLatencyMs: Double?
    }

    #if os(iOS)
    private static let cases: [Case] = [
        Case(modelName: "tiny-streaming-en", arch: .tinyStreaming, maxAvgLatencyMs: 250),
        Case(modelName: "small-streaming-en", arch: .smallStreaming, maxAvgLatencyMs: 750),
        Case(modelName: "medium-streaming-en", arch: .mediumStreaming, maxAvgLatencyMs: 1400),
    ]
    #else
    private static let cases: [Case] = [
        Case(modelName: "tiny-streaming-en", arch: .tinyStreaming, maxAvgLatencyMs: nil),
        Case(modelName: "small-streaming-en", arch: .smallStreaming, maxAvgLatencyMs: nil),
        Case(modelName: "medium-streaming-en", arch: .mediumStreaming, maxAvgLatencyMs: nil),
    ]
    #endif

    private static var ceilingsAreAdvisory: Bool {
        let value = ProcessInfo.processInfo.environment["MOONSHINE_LATENCY_OPTIONAL"] ?? ""
        return !value.isEmpty
    }

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

        // Contextual biasing is off unless the environment names key terms, so
        // the default run stays the plain latency baseline.
        let environment = ProcessInfo.processInfo.environment
        let keyterms = environment["MOONSHINE_KEYTERMS"] ?? ""
        var options: [TranscriberOption] = []
        if !keyterms.isEmpty {
            options.append(TranscriberOption(name: "keyterms", value: keyterms))
            if let boost = environment["MOONSHINE_KEYTERM_BOOST"], !boost.isEmpty {
                options.append(TranscriberOption(name: "keyterm_boost", value: boost))
            }
        }
        let keytermCount = keyterms.isEmpty ? 0 : keyterms.split(separator: ",").count

        for testCase in Self.cases {
            let spec = ModelSpec.stt(
                language: "en", modelArch: testCase.arch)
            let directory = try ModelCache.directory(for: spec)
            _ = try await AssetDownloader().ensureModelPresent(
                root: directory, spec: spec,
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
            let loadStart = Date()
            let transcriber = try Transcriber(
                modelPath: directory.path,
                modelArch: testCase.arch,
                options: options.isEmpty ? nil : options)
            let loadMs = Date().timeIntervalSince(loadStart) * 1000
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
                format: "MOONSHINE_LATENCY platform=%@ device=%@ model=%@ avg_ms=%.0f load_ms=%.0f lines=%d wall_s=%.2f keyterms=%d",
                platform, device, testCase.modelName, avgMs, loadMs, latencies.count, wallSeconds,
                keytermCount)
            print(summary)
            fputs(summary + "\n", stderr)

            guard let ceiling = testCase.maxAvgLatencyMs else { continue }
            let ceilingMessage = String(
                format: "%@ avg latency %.0fms exceeds regression ceiling %.0fms",
                testCase.modelName, avgMs, ceiling)
            if Self.ceilingsAreAdvisory {
                if avgMs > ceiling {
                    let warning = "MOONSHINE_LATENCY_WARNING " + ceilingMessage
                        + " (MOONSHINE_LATENCY_OPTIONAL is set)"
                    print(warning)
                    fputs(warning + "\n", stderr)
                }
            } else {
                XCTAssertLessThanOrEqual(avgMs, ceiling, ceilingMessage)
            }
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
