import Foundation
import XCTest

@testable import MoonshineVoice

final class IntentRecognizerTests: XCTestCase {

    func testCreateIntentRecognizer_invalidPath_throws() {
        XCTAssertThrowsError(
            try IntentRecognizer(
                modelPath: "/nonexistent/moonshine/intent/model",
                modelArch: .gemma300m
            )
        ) { error in
            guard case MoonshineError.custom = error else {
                XCTFail("Expected MoonshineError.custom, got \(error)")
                return
            }
        }
    }

    func testIntentRecognizer_closestIntents_whenEmbeddingModelPresent() throws {
        let base = try TranscriberTests.getTestAssetsPath()
        let modelDir = (base as NSString).appendingPathComponent("embeddinggemma-300m-ONNX")
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("Embedding model not in test-assets; skipping intent integration test")
        }

        let recognizer = try IntentRecognizer(modelPath: modelDir, modelArch: .gemma300m)
        defer { recognizer.close() }

        XCTAssertEqual(try recognizer.intentCount(), 0)
        try recognizer.registerIntent(canonicalPhrase: "turn on the lights")
        XCTAssertEqual(try recognizer.intentCount(), 1)

        let ranked = try recognizer.getClosestIntents(
            utterance: "turn on the lights",
            toleranceThreshold: 0.0
        )
        XCTAssertFalse(ranked.isEmpty)
        XCTAssertEqual(ranked[0].canonicalPhrase, "turn on the lights")

        XCTAssertFalse(try recognizer.unregisterIntent(canonicalPhrase: "unknown"))
        XCTAssertTrue(try recognizer.unregisterIntent(canonicalPhrase: "turn on the lights"))
        XCTAssertEqual(try recognizer.intentCount(), 0)

        try recognizer.clearIntents()
    }
}
