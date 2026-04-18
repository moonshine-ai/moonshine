import Foundation

/// One ranked intent match from ``IntentRecognizer.getClosestIntents(utterance:toleranceThreshold:)``.
public struct IntentMatch: Equatable, Sendable {
    public let canonicalPhrase: String
    public let similarity: Float

    public init(canonicalPhrase: String, similarity: Float) {
        self.canonicalPhrase = canonicalPhrase
        self.similarity = similarity
    }
}

/// Semantic intent recognizer (synchronous ranking via embedding similarity).
public final class IntentRecognizer: @unchecked Sendable {
    private let api = MoonshineAPI.shared
    private var handle: Int32

    /// Create an intent recognizer from an embedding model directory on disk.
    public init(
        modelPath: String,
        modelArch: EmbeddingModelArch = .gemma300m,
        modelVariant: String = "q4"
    ) throws {
        self.handle = try api.createIntentRecognizer(
            modelPath: modelPath,
            embeddingModelArch: modelArch.rawValue,
            modelVariant: modelVariant
        )
    }

    deinit {
        close()
    }

    public func close() {
        if handle >= 0 {
            api.freeIntentRecognizer(handle)
            handle = -1
        }
    }

    /// Register a canonical phrase to match against.
    /// - Parameters:
    ///   - canonicalPhrase: The phrase to register.
    ///   - embedding: Optional pre-computed embedding. Pass `nil` to auto-compute.
    ///   - priority: Higher-priority intents rank above lower-priority ones.
    public func registerIntent(
        canonicalPhrase: String,
        embedding: [Float]? = nil,
        priority: Int32 = 0
    ) throws {
        if let emb = embedding {
            var mutable = emb
            try mutable.withUnsafeMutableBufferPointer { buf in
                try api.registerIntentRecognizerIntent(
                    handle: handle,
                    canonicalPhrase: canonicalPhrase,
                    embedding: buf.baseAddress,
                    embeddingSize: UInt64(emb.count),
                    priority: priority)
            }
        } else {
            try api.registerIntentRecognizerIntent(
                handle: handle,
                canonicalPhrase: canonicalPhrase,
                priority: priority)
        }
    }

    /// Calculate the embedding vector for a sentence.
    /// - Parameter sentence: The input text to embed.
    /// - Returns: The embedding vector.
    public func calculateEmbedding(sentence: String) throws -> [Float] {
        try api.calculateIntentEmbedding(handle: handle, sentence: sentence)
    }

    /// - Returns: `true` if the phrase was removed, `false` if it was not registered.
    public func unregisterIntent(canonicalPhrase: String) throws -> Bool {
        try api.unregisterIntentRecognizerIntent(handle: handle, canonicalPhrase: canonicalPhrase)
    }

    /// Returns up to six matches at or above ``toleranceThreshold``, sorted by descending similarity.
    public func getClosestIntents(utterance: String, toleranceThreshold: Float) throws -> [IntentMatch] {
        let raw = try api.getClosestIntents(
            intentRecognizerHandle: handle,
            utterance: utterance,
            toleranceThreshold: toleranceThreshold
        )
        return raw.map { IntentMatch(canonicalPhrase: $0.canonicalPhrase, similarity: $0.similarity) }
    }

    public func intentCount() throws -> Int32 {
        try api.getIntentRecognizerIntentCount(handle: handle)
    }

    public func clearIntents() throws {
        try api.clearIntentRecognizerIntents(handle: handle)
    }
}
