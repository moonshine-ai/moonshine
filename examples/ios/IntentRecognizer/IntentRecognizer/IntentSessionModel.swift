import Foundation
import MoonshineVoice
import SwiftUI

struct IntentPhraseRow: Identifiable, Equatable {
    let id: UUID
    var text: String

    init(id: UUID = UUID(), text: String) {
        self.id = id
        self.text = text
    }
}

/// Defaults mirror the Python `intent_recognizer` CLI (`--intents` comma-separated default).
private let defaultIntentPhrases: [String] = [
    "turn on the lights",
    "turn off the lights",
    "what is the weather",
    "set a timer",
    "play some music",
    "stop the music",
]

final class IntentSessionModel: ObservableObject {
    @Published var phrases: [IntentPhraseRow]
    /// Minimum cosine similarity for an intent to count (same as Python `tolerance_threshold` / `--threshold`). **Lower = more permissive**; **0** keeps any score ≥ 0, so the closest phrase almost always matches.
    @Published var threshold: Double = 0.8
    /// Minimum similarity value passed into the last completed-line scoring call (verifies the slider is applied).
    @Published var lastScoredMinimumSimilarity: Double?
    /// Best intent’s similarity on the last completed line (after the minimum filter).
    @Published var lastTopMatchSimilarity: Float?
    @Published var highlightedPhraseIDs: Set<UUID> = []
    @Published var statusMessage: String = ""
    /// Partial or final text for the current transcript line while the mic is on.
    @Published var liveTranscriptLine: String = ""
    @Published var isListening: Bool = false
    @Published var isEngineReady: Bool = false

    private var mic: MicTranscriber?
    private var intentRecognizer: IntentRecognizer?
    private let transcriptBridge = IntentTranscriptBridge()
    private var highlightResetTask: Task<Void, Never>?
    private var reapplyDebouncer: Task<Void, Never>?

    init() {
        phrases = defaultIntentPhrases.map { IntentPhraseRow(text: $0) }
        transcriptBridge.session = self
    }

    @MainActor
    func bootstrapIfNeeded() {
        guard !isEngineReady else { return }

        let root = modelsRootDirectory()
        let asrPath = root.appendingPathComponent("tiny-en", isDirectory: true).path
        let embPath = root.appendingPathComponent("embeddinggemma-300m", isDirectory: true).path

        guard FileManager.default.fileExists(atPath: asrPath),
              FileManager.default.fileExists(atPath: asrPath + "/tokenizer.bin")
        else {
            statusMessage =
                "Missing tiny-en in the app bundle. Rebuild after ensuring examples/ios/IntentRecognizer/moonshine-models is present (clone with Git LFS)."
            return
        }
        guard FileManager.default.fileExists(atPath: embPath),
              FileManager.default.fileExists(atPath: embPath + "/tokenizer.bin")
        else {
            statusMessage =
                "Missing embeddinggemma-300m in the app bundle. Rebuild after ensuring moonshine-models is checked out with Git LFS."
            return
        }

        do {
            intentRecognizer = try IntentRecognizer(
                modelPath: embPath,
                modelArch: .gemma300m,
                modelVariant: "q4"
            )
            try reapplyRegisteredIntents()

            mic = try MicTranscriber(modelPath: asrPath, modelArch: .tiny)
            mic?.addListener(transcriptBridge)

            isEngineReady = true
            statusMessage = "Ready. Tap Listen to use the microphone."
        } catch {
            statusMessage = "Failed to load models: \(error.localizedDescription)"
            intentRecognizer?.close()
            intentRecognizer = nil
            mic?.close()
            mic = nil
        }
    }

    @MainActor
    private func reapplyRegisteredIntents() throws {
        guard let intentRecognizer else { return }
        try intentRecognizer.clearIntents()
        for row in phrases {
            let t = row.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty {
                try intentRecognizer.registerIntent(canonicalPhrase: t)
            }
        }
    }

    /// Debounces embedding recompute while the user is typing.
    @MainActor
    func scheduleDebouncedIntentSync() {
        reapplyDebouncer?.cancel()
        reapplyDebouncer = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 450_000_000)
            phraseTextCommitted()
        }
    }

    @MainActor
    func phraseTextCommitted() {
        guard isEngineReady else { return }
        do {
            try reapplyRegisteredIntents()
            if !isListening, !statusMessage.contains("Missing") {
                statusMessage = "Intents updated."
            }
        } catch {
            statusMessage = "Could not update intents: \(error.localizedDescription)"
        }
    }

    @MainActor
    func addPhrase() {
        phrases.append(IntentPhraseRow(text: ""))
    }

    @MainActor
    func removePhrase(id: UUID) {
        phrases.removeAll { $0.id == id }
        phraseTextCommitted()
    }

    @MainActor
    func toggleListening() {
        guard isEngineReady, let mic else { return }
        do {
            if isListening {
                try mic.stop()
                isListening = false
                liveTranscriptLine = ""
                lastScoredMinimumSimilarity = nil
                lastTopMatchSimilarity = nil
                statusMessage = "Stopped."
            } else {
                liveTranscriptLine = ""
                statusMessage = ""
                lastScoredMinimumSimilarity = nil
                lastTopMatchSimilarity = nil
                try mic.start()
                isListening = true
            }
        } catch {
            statusMessage = "Microphone error: \(error.localizedDescription)"
            isListening = false
        }
    }

    @MainActor
    func handleCompletedTranscriptLine(_ raw: String) {
        let utterance = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty, let intentRecognizer else { return }

        let minimumSimilarity = Float(threshold)
        lastScoredMinimumSimilarity = threshold

        let matches: [IntentMatch]
        do {
            matches = try intentRecognizer.getClosestIntents(
                utterance: utterance,
                toleranceThreshold: minimumSimilarity
            )
        } catch {
            statusMessage = "Intent match error: \(error.localizedDescription)"
            return
        }

        guard let top = matches.first else {
            lastTopMatchSimilarity = nil
            return
        }

        // Belt-and-suspenders: only treat as a match if score meets the current minimum (native layer should already enforce this).
        if top.similarity + 1e-5 < minimumSimilarity {
            lastTopMatchSimilarity = top.similarity
            return
        }

        lastTopMatchSimilarity = top.similarity

        if let row = phrases.first(where: {
            $0.text.trimmingCharacters(in: .whitespacesAndNewlines) == top.canonicalPhrase
        }) {
            flashHighlight(phraseID: row.id)
        }
    }

    @MainActor
    func pauseMicIfNeededForBackground() {
        guard let mic, isListening else { return }
        try? mic.stop()
        isListening = false
        liveTranscriptLine = ""
        lastScoredMinimumSimilarity = nil
        lastTopMatchSimilarity = nil
        statusMessage = "Paused (app in background)."
    }

    @MainActor
    func handleTranscriptLineStarted(_ text: String) {
        liveTranscriptLine = text
        if statusMessage.contains("Intent match error") {
            statusMessage = ""
        }
    }

    @MainActor
    func handleTranscriptLineTextChanged(_ text: String) {
        liveTranscriptLine = text
    }

    @MainActor
    func handleTranscriptLineCompleted(_ text: String) {
        liveTranscriptLine = text
        handleCompletedTranscriptLine(text)
    }

    @MainActor
    private func flashHighlight(phraseID: UUID) {
        highlightResetTask?.cancel()
        highlightedPhraseIDs.insert(phraseID)
        highlightResetTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            highlightedPhraseIDs.remove(phraseID)
        }
    }

    private func modelsRootDirectory() -> URL {
        Bundle.main.bundleURL.appendingPathComponent("moonshine-models", isDirectory: true)
    }
}
