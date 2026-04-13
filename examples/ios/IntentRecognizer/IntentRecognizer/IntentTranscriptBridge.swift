import Foundation
import MoonshineVoice

/// Forwards completed transcript lines to ``IntentSessionModel`` on the main actor (audio callbacks are not main-thread).
final class IntentTranscriptBridge: TranscriptEventListener {
    weak var session: IntentSessionModel?

    func onLineCompleted(_ event: LineCompleted) {
        let text = event.line.text
        Task { @MainActor in
            session?.handleCompletedTranscriptLine(text)
        }
    }
}
