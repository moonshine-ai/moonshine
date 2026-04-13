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
    @Published var threshold: Double = 0.3
    @Published var highlightedPhraseIDs: Set<UUID> = []
    @Published var statusMessage: String = ""
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
            if isListening {
                statusMessage = "Listening…"
            } else if !statusMessage.contains("Missing") {
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
                statusMessage = "Stopped."
            } else {
                try mic.start()
                isListening = true
                statusMessage = "Listening…"
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

        let t = Float(threshold)
        let matches: [IntentMatch]
        do {
            matches = try intentRecognizer.getClosestIntents(utterance: utterance, toleranceThreshold: t)
        } catch {
            statusMessage = "Intent match error: \(error.localizedDescription)"
            return
        }

        guard let top = matches.first else { return }

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
        statusMessage = "Paused (app in background)."
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
