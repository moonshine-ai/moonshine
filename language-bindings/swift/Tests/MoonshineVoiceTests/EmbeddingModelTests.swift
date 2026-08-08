import Foundation
import XCTest

@testable import MoonshineVoice

final class EmbeddingModelTests: XCTestCase {

    func testCreateEmbeddingModel_invalidPath_throws() {
        XCTAssertThrowsError(
            try EmbeddingModel(
                modelPath: "/nonexistent/moonshine/embedding/model",
                modelArch: .gemma300m
            )
        ) { error in
            guard case MoonshineError.custom = error else {
                XCTFail("Expected MoonshineError.custom, got \(error)")
                return
            }
        }
    }

    func testEmbeddingModel_scoresPhrases_whenEmbeddingModelPresent() throws {
        let base = try TranscriberTests.getTestAssetsPath()
        let modelDir = (base as NSString).appendingPathComponent("embeddinggemma-300m-ONNX")
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("Embedding model not in test-assets; skipping embedding integration test")
        }

        let model = try EmbeddingModel(modelPath: modelDir, modelArch: .gemma300m)
        defer { model.close() }

        let lights = try model.calculateEmbedding("turn on the lights")
        XCTAssertFalse(lights.isEmpty)
        let lamps = try model.calculateEmbedding("switch on the lamps")
        let garage = try model.calculateEmbedding("close the garage door")

        XCTAssertGreaterThan(try model.distance(lights, lights), 0.99)
        XCTAssertGreaterThan(
            try model.distance(lights, lamps), try model.distance(lights, garage))
    }

    func testPhraseMatcher_picksClosestPhrase_whenEmbeddingModelPresent() throws {
        let base = try TranscriberTests.getTestAssetsPath()
        let modelDir = (base as NSString).appendingPathComponent("embeddinggemma-300m-ONNX")
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("Embedding model not in test-assets; skipping embedding integration test")
        }

        let model = try EmbeddingModel(modelPath: modelDir, modelArch: .gemma300m)
        defer { model.close() }
        let matcher = PhraseMatcher(model: model)

        let phrases = ["turn on the lights", "close the garage door"]
        XCTAssertEqual(
            matcher.match("switch on the lights", phrases: phrases, threshold: 0.55),
            "turn on the lights")
        XCTAssertNil(
            matcher.match("the stock market crashed", phrases: phrases, threshold: 0.9))
    }

    /// Without a model the matcher falls back to substring matching, which is
    /// what keeps dialogs working before `AgentFlow.load()`.
    func testPhraseMatcher_withoutModel_fallsBackToSubstrings() {
        let matcher = PhraseMatcher(model: nil)
        let groups = [(key: "yes", phrases: ["yes", "sure"]), (key: "no", phrases: ["no", "nope"])]
        XCTAssertEqual(matcher.match("Sure, go ahead", groups: groups, threshold: 0.55), "yes")
        XCTAssertEqual(matcher.match("nope", groups: groups, threshold: 0.55), "no")
        XCTAssertNil(matcher.match("maybe later", groups: groups, threshold: 0.55))
    }
}
