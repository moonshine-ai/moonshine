import Foundation
import MoonshineVoice

/// A voice-driven wifi setup, the whole thing. Say "set up wifi" to start it,
/// "cancel" to abandon it, or "start over" to run it again. The last two are
/// built in, so this file never mentions them.
///
/// Run with `--text` to type the answers instead of speaking them, which is
/// handy when you have no microphone (or no patience).
func wifiSetup(_ d: Dialog) async throws {
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

func main() async {
    let typed = CommandLine.arguments.contains("--text")

    let dialog = MoonshineVoice.DialogFlow()
        .microphone(!typed)
        .onHeard { text in print("you: \(text)") }
        .onSaid { text in print("assistant: \(text)") }
        .onError { error in fputs("Flow error: \(error)\n", stderr) }
        .onProgress { fraction, file in
            fputs("\r  \(file) \(Int(fraction * 100))%      ", stderr)
        }

    dialog.listenFor("set up wifi", wifiSetup)
    defer { dialog.close() }

    do {
        // One call downloads the speech, embedding, and voice models and wires the
        // three engines together.
        try await dialog.load()
        fputs("\n", stderr)
    } catch {
        fputs("\nError: failed to load: \(error)\n", stderr)
        exit(1)
    }

    if typed {
        print("Type an utterance (\"set up wifi\" to begin), or Ctrl+D to quit.")
        while let line = readLine() {
            let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            await dialog.handleUtterance(text)
        }
        return
    }

    do {
        try dialog.startListening()
    } catch {
        fputs("Error: couldn't open the microphone: \(error)\n", stderr)
        exit(1)
    }
    print("Listening. Say \"set up wifi\" to begin; press Ctrl+C to quit.")
    while true {
        try? await Task.sleep(nanoseconds: 1_000_000_000)
    }
}

await main()
