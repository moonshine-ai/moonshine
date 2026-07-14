import Foundation
import XCTest

@testable import MoonshineVoice

/// End-to-end tests that exercise ``AssetDownloader`` against the **real** CDN
/// (https://download.moonshine.ai): they download a model into an empty directory and then load and
/// run the matching engine, proving the whole download-then-load path works, not just manifest
/// parsing (which ``AssetDownloaderTests`` covers with a mocked protocol).
///
/// These require a working network connection and pull tens to hundreds of MB, so they are opt-in:
/// they run only when the environment variable `MOONSHINE_DOWNLOAD_TESTS` is set to a truthy value
/// (`1`, `true`, or `yes`). `scripts/test-model-downloads.sh` sets it; a plain `swift test` skips
/// them so the default suite stays hermetic and offline.
@available(iOS 15.0, macOS 12.0, *)
final class AssetDownloaderNetworkTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        try Self.skipUnlessEnabled()
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("moonshine-download-network-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempRoot = tempRoot {
            try? FileManager.default.removeItem(at: tempRoot)
        }
        super.tearDown()
    }

    /// Skips (rather than fails) when the network download tests have not been explicitly enabled.
    private static func skipUnlessEnabled() throws {
        let raw = ProcessInfo.processInfo.environment["MOONSHINE_DOWNLOAD_TESTS"]?
            .lowercased() ?? ""
        guard ["1", "true", "yes"].contains(raw) else {
            throw XCTSkip(
                "Set MOONSHINE_DOWNLOAD_TESTS=1 to run network download tests against the CDN")
        }
    }

    // MARK: - STT

    /// Downloads the tiny English STT model, loads it with ``Transcriber``, and transcribes a
    /// bundled WAV, asserting on known words so a broken/partial download is caught.
    func testDownloadsAndRunsSttModel() async throws {
        let downloader = AssetDownloader()
        let spec = ModelSpec.stt(language: "en", modelArch: .tiny)

        XCTAssertFalse(downloader.isModelPresent(root: tempRoot, spec: spec))
        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        XCTAssertTrue(
            downloader.isModelPresent(root: tempRoot, spec: spec),
            "every STT file should be present after download")

        let transcriber = try Transcriber(modelPath: tempRoot.path, modelArch: .tiny)
        defer { transcriber.close() }

        let wavPath = try TranscriberTests.getWAVFilePath("two_cities.wav")
        let wav = try loadWAVFile(wavPath)
        let transcript = try transcriber.transcribeWithoutStreaming(
            audioData: wav.audioData, sampleRate: Int32(wav.sampleRate))

        let text = transcript.lines.map { $0.text }.joined(separator: " ").lowercased()
        XCTAssertTrue(text.contains("best of times"), "unexpected transcript: \(text)")
        XCTAssertTrue(text.contains("worst of times"), "unexpected transcript: \(text)")
    }

    // MARK: - TTS

    /// Downloads a Kokoro English voice plus its G2P assets, then synthesizes speech from the
    /// downloaded directory.
    func testDownloadsAndRunsTtsVoice() async throws {
        let downloader = AssetDownloader()
        let spec = ModelSpec.tts(language: "en_us", voice: "kokoro_af_heart")

        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        XCTAssertTrue(downloader.isModelPresent(root: tempRoot, spec: spec))

        let tts = try TextToSpeech(
            language: "en_us", g2pRoot: tempRoot.path, voice: "kokoro_af_heart")
        defer { tts.close() }

        let result = try tts.synthesize(text: "Hello from the download test.")
        XCTAssertGreaterThan(result.samples.count, 0, "synthesis produced no audio")
        XCTAssertGreaterThan(result.sampleRateHz, 0)
    }

    // MARK: - Intent

    /// Downloads the (large) intent-recognition embedding model and runs a trivial match. Gated by
    /// the same env var as the others; skip if you only want the lighter STT/TTS coverage.
    func testDownloadsAndRunsIntentModel() async throws {
        let downloader = AssetDownloader()
        let spec = ModelSpec.intent(variant: "q4")

        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        XCTAssertTrue(downloader.isModelPresent(root: tempRoot, spec: spec))

        let recognizer = try IntentRecognizer(modelPath: tempRoot.path, modelArch: .gemma300m)
        defer { recognizer.close() }

        try recognizer.registerIntent(canonicalPhrase: "turn on the lights")
        let ranked = try recognizer.getClosestIntents(
            utterance: "turn on the lights", toleranceThreshold: 0.0)
        XCTAssertFalse(ranked.isEmpty)
        XCTAssertEqual(ranked[0].canonicalPhrase, "turn on the lights")
    }
}
