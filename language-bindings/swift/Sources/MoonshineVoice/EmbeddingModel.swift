import Foundation

/// Turns text into embedding vectors and scores them against each other.
///
/// A low-level type, like ``Transcriber``. ``AgentFlow`` is the supported way
/// to match spoken phrases in an app; it owns a model and compares utterances
/// to phrases itself.
public final class EmbeddingModel: @unchecked Sendable {
    private let api = MoonshineAPI.shared
    private var handle: Int32

    /// Create an embedding model from a model directory on disk.
    public init(
        modelPath: String,
        modelArch: EmbeddingModelArch = .gemma300m,
        modelVariant: String = "q4"
    ) throws {
        self.handle = try api.createEmbeddingModel(
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
            api.freeEmbeddingModel(handle)
            handle = -1
        }
    }

    /// The embedding vector for `sentence`.
    public func calculateEmbedding(_ sentence: String) throws -> [Float] {
        return try api.calculateEmbedding(handle: handle, sentence: sentence)
    }

    /// Cosine similarity between two embeddings of equal length, in `-1...1`.
    public func distance(_ embeddingA: [Float], _ embeddingB: [Float]) throws -> Float {
        return try api.calculateEmbeddingDistance(
            handle: handle, embeddingA: embeddingA, embeddingB: embeddingB)
    }
}

/// Matches an utterance to one of several key→phrases groups by meaning.
///
/// Each phrase is embedded once and cached, the utterance is embedded once per
/// call, and the key of the best-scoring phrase at or above `threshold` wins.
/// Without an ``EmbeddingModel`` it falls back to case-insensitive substring
/// matching, which is what keeps dialogs working before ``AgentFlow/load()``.
final class PhraseMatcher: @unchecked Sendable {
    private let model: EmbeddingModel?
    private let lock = Mutex()
    private var cache: [String: [Float]] = [:]

    init(model: EmbeddingModel?) {
        self.model = model
    }

    /// The best-matching key, or nil when nothing clears `threshold`.
    func match(_ utterance: String, groups: [(key: String, phrases: [String])], threshold: Float)
        -> String?
    {
        guard !utterance.isEmpty, !groups.isEmpty else { return nil }
        guard let model else {
            let lower = utterance.lowercased()
            return groups.first { group in
                group.phrases.contains { phrase in
                    let needle = phrase.lowercased()
                    return !needle.isEmpty && (lower == needle || lower.contains(needle))
                }
            }?.key
        }

        guard let utteranceEmbedding = try? model.calculateEmbedding(utterance) else { return nil }
        var bestKey: String?
        var bestScore: Float = -1
        for group in groups {
            for phrase in group.phrases where !phrase.isEmpty {
                guard let phraseEmbedding = embedding(for: phrase, model: model),
                    let score = try? model.distance(utteranceEmbedding, phraseEmbedding)
                else { continue }
                if score > bestScore {
                    bestScore = score
                    bestKey = group.key
                }
            }
        }
        return bestScore >= threshold ? bestKey : nil
    }

    /// The best-matching phrase, treating each phrase as its own key.
    func match(_ utterance: String, phrases: [String], threshold: Float) -> String? {
        return match(
            utterance, groups: phrases.map { (key: $0, phrases: [$0]) }, threshold: threshold)
    }

    private func embedding(for phrase: String, model: EmbeddingModel) -> [Float]? {
        if let cached = lock.withLock({ cache[phrase] }) { return cached }
        guard let computed = try? model.calculateEmbedding(phrase) else { return nil }
        lock.withLock { cache[phrase] = computed }
        return computed
    }
}
