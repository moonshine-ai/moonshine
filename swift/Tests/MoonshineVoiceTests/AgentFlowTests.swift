import Foundation
import XCTest

@testable import MoonshineVoice

/// Drives ``AgentFlow`` from text rather than a microphone, so these run
/// without downloading any models: with no embedding model loaded the runner
/// falls back to substring matching, and ``AgentFlow/speakWith(_:)`` captures
/// the prompts instead of playing them.
@available(iOS 15.0, macOS 12.0, *)
final class AgentFlowTests: XCTestCase {

    /// Collects what the assistant said, from whichever thread says it.
    private final class Transcript: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []

        func append(_ line: String) {
            lock.lock()
            lines.append(line)
            lock.unlock()
        }

        var all: [String] {
            lock.lock()
            defer { lock.unlock() }
            return lines
        }
    }

    private func makeAgent(_ spoken: Transcript) -> AgentFlow {
        let agent = AgentFlow().microphone(false)
        agent.speakWith { text in spoken.append(text) }
        return agent
    }

    func testRunsAFlowToCompletion() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        agent.listenFor("set up wifi") { d in
            let ssid = try await d.ask("What's the name of your wifi network?")
            if try await d.confirm("I heard \(ssid). Is that right?") {
                try await d.say("Done. Connecting to \(ssid).")
            } else {
                try await d.say("No problem, let's start over.")
            }
        }

        await agent.handleUtterance("set up wifi")
        await agent.handleUtterance("home network")
        await agent.handleUtterance("yes")

        XCTAssertEqual(
            spoken.all,
            [
                "What's the name of your wifi network?",
                "I heard home network. Is that right?",
                "Done. Connecting to home network.",
            ])
        XCTAssertFalse(agent.isActive)
    }

    func testConfirmUnderstandsNo() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        agent.listenFor("set up wifi") { d in
            let ssid = try await d.ask("Which network?")
            if try await d.confirm("I heard \(ssid). Is that right?") {
                try await d.say("Connecting.")
            } else {
                try await d.say("No problem.")
            }
        }

        await agent.handleUtterance("set up wifi")
        await agent.handleUtterance("cafe wifi")
        await agent.handleUtterance("nope")

        XCTAssertEqual(spoken.all.last, "No problem.")
    }

    func testCancelIsBuiltIn() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        let reachedEnd = Transcript()
        agent.listenFor("set up wifi") { d in
            _ = try await d.ask("Which network?")
            reachedEnd.append("finished")
        }

        await agent.handleUtterance("set up wifi")
        await agent.handleUtterance("cancel")

        XCTAssertTrue(reachedEnd.all.isEmpty, "cancel should abandon the flow")
        XCTAssertFalse(agent.isActive)
    }

    func testStartOverIsBuiltIn() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        agent.listenFor("set up wifi") { d in
            let ssid = try await d.ask("Which network?")
            try await d.say("Using \(ssid).")
        }

        await agent.handleUtterance("set up wifi")
        await agent.handleUtterance("start over")
        await agent.handleUtterance("home network")

        XCTAssertEqual(
            spoken.all, ["Which network?", "Which network?", "Using home network."])
    }

    func testRepromptsThenGivesUp() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        agent.listenFor("set up wifi") { d in
            _ = try await d.confirm("Ready?")
            try await d.say("Never reached.")
        }

        await agent.handleUtterance("set up wifi")
        await agent.handleUtterance("bananas")
        await agent.handleUtterance("more bananas")

        XCTAssertEqual(spoken.all.last, "Sorry, I didn't get that. Let's start over.")
        XCTAssertFalse(spoken.all.contains("Never reached."))
    }

    func testChoosePicksAKey() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        let picked = Transcript()
        agent.listenFor("pick a band") { d in
            let band = try await d.choose(
                "Which band?", options: ["2.4 GHz": ["slow", "long range"], "5 GHz": ["fast"]])
            picked.append(band)
        }

        await agent.handleUtterance("pick a band")
        await agent.handleUtterance("the fast one")

        XCTAssertEqual(picked.all, ["5 GHz"])
    }

    func testUnknownUtterancesAreIgnored() async throws {
        let spoken = Transcript()
        let agent = makeAgent(spoken)
        agent.listenFor("set up wifi") { d in
            try await d.say("Starting.")
        }

        await agent.handleUtterance("what's the weather like")

        XCTAssertTrue(spoken.all.isEmpty)
        XCTAssertFalse(agent.isActive)
    }

    func testOnHeardAndOnSaidSeeBothSides() async throws {
        let spoken = Transcript()
        let heard = Transcript()
        let said = Transcript()
        let agent = makeAgent(spoken)
        agent.onHeard { heard.append($0) }
        agent.onSaid { said.append($0) }
        agent.listenFor("say hello") { d in
            try await d.say("Hello.")
        }

        await agent.handleUtterance("say hello")

        XCTAssertEqual(heard.all, ["say hello"])
        XCTAssertEqual(said.all, ["Hello."])
    }

    func testFlowErrorsReachOnError() async throws {
        struct Boom: Error {}
        let spoken = Transcript()
        let failures = Transcript()
        let agent = makeAgent(spoken)
        agent.onError { failures.append("\($0)") }
        agent.listenFor("break things") { _ in
            throw Boom()
        }

        await agent.handleUtterance("break things")

        XCTAssertEqual(failures.all.count, 1)
        XCTAssertFalse(agent.isActive)
    }

    func testSpellOutSeparatesCharacters() {
        XCTAssertEqual(spellOut("wifi"), "w i f i")
    }
}
