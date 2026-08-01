import AVFoundation
import MoonshineVoice
import SwiftUI

/// One line of the conversation, for the transcript view.
struct DialogLine: Identifiable, Equatable {
    enum Speaker {
        case you
        case assistant
    }

    let id = UUID()
    let speaker: Speaker
    let text: String
}

/// A voice-driven wifi setup, the whole thing.
///
/// Say "set up wifi" to start it, "cancel" to abandon it, or "start over" to run
/// it again. The last two are built in, so this function never mentions them.
///
/// Every `ask` speaks and then waits, so the conversation is straight-line code:
/// no state machine, and no callbacks to thread together.
@Sendable func wifiSetup(_ d: Dialog) async throws {
    let ssid = try await d.ask("What's the name of your wifi network?")
    guard try await d.confirm("I heard \(ssid). Is that right?") else {
        try await d.say("No problem, let's start over.")
        try d.restart()
    }

    let security = try await d.choose(
        "Is the network open, or does it use a password?",
        options: [
            "open": ["open", "no password", "none"],
            "password": ["password", "secured", "protected", "wpa"],
        ])
    if security == "password" {
        let password = try await d.ask("What's the password? Spell it out one letter at a time.")
        try await d.say("Got it, \(spellOut(password)).")
    }

    if try await d.confirm("Apply these changes?") {
        try await d.say("Done. Connecting to \(ssid).")
    } else {
        try await d.say("Okay, nothing changed.")
    }
}

/// Observable model that owns the agent and turns its callbacks into state
/// SwiftUI can render.
@MainActor
final class AgentSession: ObservableObject {
    @Published var lines: [DialogLine] = []
    @Published var status: String = "Preparing…"
    @Published var downloadFraction: Double? = nil
    @Published var isReady: Bool = false
    @Published var isListening: Bool = false

    private var agent: MoonshineVoice.AgentFlow? = nil

    static let prompt = "Say \u{201C}set up wifi\u{201D} to begin. "
        + "\u{201C}Cancel\u{201D} and \u{201C}start over\u{201D} work at any point."

    func prepare() async {
        guard agent == nil else { return }

        let flow = MoonshineVoice.AgentFlow()
            .onHeard { [weak self] text in
                Task { @MainActor in self?.lines.append(DialogLine(speaker: .you, text: text)) }
            }
            .onSaid { [weak self] text in
                Task { @MainActor in
                    self?.lines.append(DialogLine(speaker: .assistant, text: text))
                }
            }
            .onError { [weak self] error in
                Task { @MainActor in
                    self?.status = "Flow error: \(error.localizedDescription)"
                }
            }
            .onProgress { [weak self] fraction, file in
                let name = (file as NSString).lastPathComponent
                Task { @MainActor in
                    self?.downloadFraction = fraction
                    self?.status = "Downloading \(name) \(Int(fraction * 100))%…"
                }
            }

        flow.listenFor("set up wifi", wifiSetup)
        agent = flow

        status = "Downloading models (first run only)…"
        do {
            // One call downloads the speech, embedding and voice models and wires
            // the three engines together.
            try await flow.load()
            downloadFraction = nil
            isReady = true
            status = Self.prompt
        } catch {
            downloadFraction = nil
            status = "Failed to load: \(error.localizedDescription)"
        }
    }

    func startListening() {
        guard let agent, isReady, !isListening else { return }
        do {
            try agent.startListening()
            isListening = true
            status = Self.prompt
        } catch {
            status = "Couldn't open the microphone: \(error.localizedDescription)"
        }
    }

    func stopListening() {
        guard let agent, isListening else { return }
        try? agent.stopListening()
        isListening = false
        status = "Stopped."
    }

    /// Feeds typed text in as though it had been heard. The same path the
    /// microphone takes, and the only one that works on a simulator.
    func send(_ text: String) {
        guard let agent, isReady else { return }
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty else { return }
        Task { await agent.handleUtterance(utterance) }
    }

    func close() {
        agent?.close()
        agent = nil
    }
}

@main
struct AgentFlowApp: App {
    @StateObject private var session = AgentSession()

    var body: some Scene {
        WindowGroup {
            ContentView(session: session)
                .task { await session.prepare() }
        }
    }
}
