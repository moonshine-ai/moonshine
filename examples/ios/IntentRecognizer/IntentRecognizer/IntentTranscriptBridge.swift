import Foundation
import MoonshineVoice

/// Forwards streaming and completed transcript lines to ``IntentSessionModel`` on the main actor (audio callbacks are not main-thread).
final class IntentTranscriptBridge: TranscriptEventListener {
    weak var session: IntentSessionModel?

    func onLineStarted(_ event: LineStarted) {
        Task { @MainActor in
            session?.handleTranscriptLineStarted(event.line.text)
        }
    }

    func onLineTextChanged(_ event: LineTextChanged) {
        let text = event.line.text
        Task { @MainActor in
            session?.handleTranscriptLineTextChanged(text)
        }
    }

    func onLineCompleted(_ event: LineCompleted) {
        let text = event.line.text
        Task { @MainActor in
            session?.handleTranscriptLineCompleted(text)
        }
    }
}

