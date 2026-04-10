import Foundation
import XCTest

@testable import MoonshineVoice

final class TextToSpeechTests: XCTestCase {

    /// Resolve the path to `core/moonshine-tts/data` by walking up from this source file.
    static func getTtsDataPath(file: StaticString = #filePath) throws -> String {
        // Walk up from the test source file to find the repo root
        var dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        for _ in 0..<10 {
            let candidate = dir.appendingPathComponent("core/moonshine-tts/data")
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir),
               isDir.boolValue {
                return candidate.path
            }
            dir = dir.deletingLastPathComponent()
        }
        // Also try relative to the working directory
        for rel in ["../core/moonshine-tts/data", "core/moonshine-tts/data"] {
            let url = URL(fileURLWithPath: rel).standardized
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
               isDir.boolValue {
                return url.path
            }
        }
        throw XCTSkip(
            "TTS data directory not found. Expected core/moonshine-tts/data in the repo."
        )
    }

    // MARK: - Creation Tests

    func testCreateSynthesizer() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        XCTAssertEqual(tts.language, "en_us")
    }

    func testCreateSynthesizerWithVoice() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(
            language: "en_us",
            g2pRoot: dataPath,
            voice: "kokoro_af_heart"
        )
        defer { tts.close() }

        XCTAssertEqual(tts.language, "en_us")
    }

    func testCreateSynthesizerInvalidLanguage() throws {
        let dataPath = try Self.getTtsDataPath()

        XCTAssertThrowsError(
            try TextToSpeech(language: "xx_invalid", g2pRoot: dataPath)
        ) { error in
            XCTAssertTrue(error is MoonshineError, "Should throw MoonshineError")
        }
    }

    // MARK: - Synthesis Tests

    func testSynthesizeBasic() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        let result = try tts.synthesize(text: "Hello world!")

        XCTAssertGreaterThan(result.samples.count, 0, "Should produce audio samples")
        XCTAssertGreaterThan(result.sampleRateHz, 0, "Sample rate should be positive")
        print("Synthesized \(result.samples.count) samples at \(result.sampleRateHz) Hz")
    }

    func testSynthesizeLongerText() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        let shortResult = try tts.synthesize(text: "Hi.")
        let longResult = try tts.synthesize(
            text: "This is a longer sentence that should produce more audio samples than a short one."
        )

        XCTAssertGreaterThan(
            longResult.samples.count, shortResult.samples.count,
            "Longer text should produce more samples"
        )
        XCTAssertEqual(
            longResult.sampleRateHz, shortResult.sampleRateHz,
            "Sample rate should be consistent"
        )
    }

    func testSynthesizeWithVoiceOption() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(
            language: "en_us",
            g2pRoot: dataPath,
            voice: "kokoro_af_heart"
        )
        defer { tts.close() }

        let result = try tts.synthesize(text: "Testing with a specific voice.")

        XCTAssertGreaterThan(result.samples.count, 0, "Should produce audio samples")
        XCTAssertGreaterThan(result.sampleRateHz, 0, "Sample rate should be positive")
    }

    func testSynthesizeWithSpeedOption() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        let normalResult = try tts.synthesize(text: "Testing speed control.")
        let fastResult = try tts.synthesize(
            text: "Testing speed control.",
            options: [TranscriberOption(name: "speed", value: "1.5")]
        )

        XCTAssertGreaterThan(normalResult.samples.count, 0)
        XCTAssertGreaterThan(fastResult.samples.count, 0)
        // Faster speed should produce fewer or equal samples
        XCTAssertLessThanOrEqual(
            fastResult.samples.count, normalResult.samples.count,
            "Faster speed should not produce more samples"
        )
    }

    func testSynthesizeSampleRange() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        let result = try tts.synthesize(text: "Hello world!")

        // Verify samples are in a reasonable range (PCM float, approximately -1..1)
        let maxAbs = result.samples.map { abs($0) }.max() ?? 0
        XCTAssertLessThanOrEqual(
            maxAbs, 2.0,
            "Sample values should be approximately in -1..1 range (max abs: \(maxAbs))"
        )
        XCTAssertGreaterThan(maxAbs, 0.01, "Samples should not be near-silent")
    }

    func testSynthesizeMultipleCalls() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        let result1 = try tts.synthesize(text: "First sentence.")
        let result2 = try tts.synthesize(text: "Second sentence.")
        let result3 = try tts.synthesize(text: "Third sentence.")

        XCTAssertGreaterThan(result1.samples.count, 0)
        XCTAssertGreaterThan(result2.samples.count, 0)
        XCTAssertGreaterThan(result3.samples.count, 0)
        XCTAssertEqual(result1.sampleRateHz, result2.sampleRateHz)
        XCTAssertEqual(result2.sampleRateHz, result3.sampleRateHz)
    }

    // MARK: - Say Tests

    func testSayDefaultDevice() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        // This will play audio on the default device; mainly tests that it
        // doesn't crash or throw.
        try tts.say("Hello from Swift TTS.")
    }

    func testSayMultipleCalls() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)
        defer { tts.close() }

        // Verify the engine caching works across repeated calls.
        try tts.say("First.")
        try tts.say("Second.")
        try tts.say("Third.")
    }

    // MARK: - Static Query Tests

    func testGetVoices() throws {
        let dataPath = try Self.getTtsDataPath()
        let json = try TextToSpeech.getVoices(
            languages: "en_us",
            options: [TranscriberOption(name: "g2p_root", value: dataPath)]
        )

        XCTAssertFalse(json.isEmpty, "Voices JSON should not be empty")
        XCTAssertTrue(json.contains("en_us"), "Should contain en_us language key")
        XCTAssertTrue(json.contains("kokoro_af_heart"), "Should list kokoro_af_heart voice")
        print("TTS voices: \(json.prefix(500))...")
    }

    func testGetDependencies() throws {
        let dataPath = try Self.getTtsDataPath()
        let json = try TextToSpeech.getDependencies(
            languages: "en_us",
            options: [TranscriberOption(name: "g2p_root", value: dataPath)]
        )

        XCTAssertFalse(json.isEmpty, "Dependencies JSON should not be empty")
        print("TTS dependencies: \(json.prefix(500))...")
    }

    // MARK: - Device Enumeration (macOS only)

    #if os(macOS)
    func testGetAudioOutputDevices() throws {
        let devices = TextToSpeech.getAudioOutputDevices()

        // Most Macs have at least one output device
        XCTAssertGreaterThan(devices.count, 0, "Should find at least one output device")

        for device in devices {
            XCTAssertFalse(device.name.isEmpty, "Device name should not be empty")
            print("Audio output: [\(device.id)] \(device.name)")
        }
    }
    #endif

    // MARK: - Resource Management

    func testCloseIdempotent() throws {
        let dataPath = try Self.getTtsDataPath()
        let tts = try TextToSpeech(language: "en_us", g2pRoot: dataPath)

        // Closing multiple times should not crash
        tts.close()
        tts.close()
        tts.close()
    }
}
