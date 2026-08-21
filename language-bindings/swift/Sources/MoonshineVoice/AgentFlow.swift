import Foundation

/// Voice agents: the one-call way to build a speech interface.
///
/// ```swift
/// let agent = AgentFlow()
///
/// agent.listenFor("set up wifi") { d in
///     let ssid = try await d.ask("What's the name of your wifi network?")
///     if try await d.confirm("I heard \(ssid). Is that right?") {
///         try await d.say("Done. Connecting to \(ssid).")
///     }
/// }
///
/// try await agent.load()
/// try await agent.startListening()
/// ```
///
/// ``load()`` downloads and wires everything a voice interface needs: a
/// streaming speech-to-text model, an embedding model for matching trigger
/// phrases, a text-to-speech voice, and a microphone. ``speech(_:)`` and
/// ``microphone(_:)`` each drop one of those when an application does not need
/// it. A flow is an ordinary async function, so it reads top to bottom and
/// `try` / `defer` work the way you expect.

/// Thrown into a flow when the user (or a global handler) cancels it.
public struct DialogCancelled: Error {
    public init() {}
}

/// Thrown into a flow when it should start again from the top.
public struct DialogRestart: Error {
    public init() {}
}

/// Thrown out of `ask` / `confirm` / `choose` after the retries run out.
public struct DialogNoMatch: Error {
    public let message: String
    public init(_ message: String = "No matching answer") {
        self.message = message
    }
}

/// How long to wait for an answer, and what to say when one doesn't arrive.
public struct AskOptions: Sendable {
    /// Give up waiting after this long and re-prompt.
    public var timeout: TimeInterval?
    /// Spoken when the answer wasn't understood. `{prompt}` is substituted.
    public var reprompt: String?
    /// How many times to re-prompt before giving up. Defaults to 2.
    public var maxRetries: Int?

    public init(
        timeout: TimeInterval? = nil, reprompt: String? = nil, maxRetries: Int? = nil
    ) {
        self.timeout = timeout
        self.reprompt = reprompt
        self.maxRetries = maxRetries
    }
}

/// What people say when they mean yes, matched by ``Dialog/confirm(_:yesPhrases:noPhrases:options:)``.
public let defaultYesPhrases = [
    "yes", "yeah", "yep", "correct", "that's right", "sure", "affirmative", "okay", "please do",
    "do it",
]
/// What people say when they mean no.
public let defaultNoPhrases = [
    "no", "nope", "incorrect", "that's wrong", "negative", "cancel", "don't do it", "stop",
]

/// The conversation, handed to a flow as its only argument. Every method speaks
/// and then waits, so a flow is just straight-line code.
@available(iOS 15.0, macOS 12.0, *)
public final class Dialog: @unchecked Sendable {
    /// The phrase that started this flow.
    public let triggerPhrase: String
    /// Scratch space for the flow's own use; the runner never touches it.
    public var state: [String: Any] = [:]

    private unowned let runner: AgentFlow

    init(runner: AgentFlow, triggerPhrase: String = "") {
        self.runner = runner
        self.triggerPhrase = triggerPhrase
    }

    /// Speaks `text` and waits for playback to finish.
    public func say(_ text: String) async throws {
        try await runner.speak(text)
    }

    /// Asks an open question and returns what the user said.
    public func ask(_ prompt: String, options: AskOptions = AskOptions()) async throws -> String {
        return try await runner.promptForAnswer(prompt: prompt, options: options) { text in
            text.isEmpty ? nil : text
        }
    }

    /// Asks a yes/no question.
    public func confirm(
        _ prompt: String,
        yesPhrases: [String] = defaultYesPhrases,
        noPhrases: [String] = defaultNoPhrases,
        options: AskOptions = AskOptions()
    ) async throws -> Bool {
        var settings = options
        settings.maxRetries = options.maxRetries ?? 1
        settings.reprompt =
            options.reprompt ?? "Sorry, I didn't catch that. Was that a yes or a no? {prompt}"
        return try await runner.promptForAnswer(prompt: prompt, options: settings) { [runner] text in
            switch runner.matchKey(
                text, groups: [("yes", yesPhrases), ("no", noPhrases)])
            {
            case "yes": return true
            case "no": return false
            default: return nil
            }
        }
    }

    /// Offers a set of choices and returns the key of the one picked. Each key
    /// maps to the phrases that select it; the key itself always counts.
    public func choose(
        _ prompt: String,
        options choices: [String: [String]],
        settings: AskOptions = AskOptions()
    ) async throws -> String {
        return try await runner.promptForAnswer(prompt: prompt, options: settings) { [runner] text in
            // The key itself always counts as one of its phrases.
            return runner.matchKey(
                text, groups: choices.map { (key: $0.key, phrases: [$0.key] + $0.value) })
        }
    }

    /// Abandons the flow.
    public func cancel() throws -> Never {
        throw DialogCancelled()
    }

    /// Runs the flow again from the beginning.
    public func restart() throws -> Never {
        throw DialogRestart()
    }
}

@available(iOS 15.0, macOS 12.0, *)
public typealias FlowFunction = @Sendable (Dialog) async throws -> Void

/// Receives speech that no flow, global or prompt claimed.
@available(iOS 15.0, macOS 12.0, *)
public typealias UnmatchedHandler = @Sendable (String) async -> Void

@available(iOS 15.0, macOS 12.0, *)
public final class AgentFlow: @unchecked Sendable {
    private static let defaultTriggerThreshold: Float = 0.7
    /// Prompt answers ("yes", "the blue one") are short and varied, so they
    /// match on a looser threshold than trigger phrases.
    private static let promptThreshold: Float = 0.55

    private var flowOrder: [String] = []
    private var flows: [String: FlowFunction] = [:]
    private var globalOrder: [String] = []
    private var globals: [String: FlowFunction] = [:]
    /// Globals that only mean anything while a flow is running. The built-in
    /// "cancel" and "start over" are in here: matching them when nothing is
    /// active would consume the line, do nothing with it, and leave a dictation
    /// interface silently missing a sentence.
    private var flowScopedGlobals: Set<String> = []

    private var languageCode = "en"
    private var arch: ModelArch = .mediumStreaming
    private var voiceId: String?
    private var wantsMicrophone = true
    private var wantsSpeech = true
    private var threshold = AgentFlow.defaultTriggerThreshold
    private var modelDirectory: URL?
    private var progressHandler: (@Sendable (Double, String) -> Void)?
    private var speakOverride: (@Sendable (String) async throws -> Void)?
    private var heardHandlers: [@Sendable (String) -> Void] = []
    private var saidHandlers: [@Sendable (String) -> Void] = []
    private var errorHandlers: [@Sendable (Error) -> Void] = []
    private var unmatchedHandlers: [UnmatchedHandler] = []

    private var tts: TextToSpeech?
    private var embedding: EmbeddingModel?
    private var matcher = PhraseMatcher(model: nil)
    private var mic: MicTranscriber?
    private var ownsTts = true
    private var ownsMic = true

    private let lock = Mutex()
    private var activeDialog: Dialog?
    private var activeTriggerPhrase: String?
    private var pending: CheckedContinuation<String, Error>?
    private var pendingTimeout: Task<Void, Never>?
    private var speaking = false
    /// Serializes utterance handling so one flow advances at a time.
    private var queueTask: Task<Void, Never>?
    /// Woken when the runner comes to rest, meaning the flow either finished or
    /// is parked waiting for the next thing the user says. Handing an utterance
    /// in resumes at that point rather than when the whole flow completes,
    /// which would deadlock: the flow is waiting for the utterance after this
    /// one.
    private var settleWaiters: [SettleSignal] = []

    public init() {
        // "cancel" and "start over" are what people actually say to a voice
        // interface, so they work without every application registering them.
        // Both only apply to a flow in progress, so they stay out of the way of
        // whatever else the microphone is being used for. Registering either
        // with `always(_:_:)` makes it live all the time, as any other global
        // is.
        addFlowScopedGlobal("cancel") { dialog in try dialog.cancel() }
        addFlowScopedGlobal("start over") { dialog in try dialog.restart() }
    }

    private func addFlowScopedGlobal(_ phrase: String, _ handler: @escaping FlowFunction) {
        always(phrase, handler)
        flowScopedGlobals.insert(phrase)
    }

    deinit {
        close()
    }

    // MARK: - Configuration

    /// Speech-to-text and synthesis language. Defaults to `"en"`.
    @discardableResult
    public func language(_ code: String) -> Self {
        languageCode = code
        return self
    }

    /// Overrides the streaming speech-to-text model.
    @discardableResult
    public func modelArch(_ arch: ModelArch) -> Self {
        self.arch = arch
        return self
    }

    /// Voice used for spoken prompts, e.g. `"kokoro_af_heart"`.
    @discardableResult
    public func voice(_ id: String) -> Self {
        voiceId = id
        return self
    }

    /// Loads every model from a directory you supply rather than the CDN.
    @discardableResult
    public func modelsFrom(_ directory: URL) -> Self {
        modelDirectory = directory
        return self
    }

    /// Set to false to drive the agent from text instead of a microphone.
    @discardableResult
    public func microphone(_ enabled: Bool) -> Self {
        wantsMicrophone = enabled
        return self
    }

    /// Whether ``load()`` should open a synthesizer. Defaults to true. Turn it
    /// off for a silent runner: prompts still reach ``onSaid(_:)`` and flows
    /// still advance, they just aren't spoken, and no voice is downloaded.
    @discardableResult
    public func speech(_ enabled: Bool = true) -> Self {
        wantsSpeech = enabled
        return self
    }

    /// Similarity a trigger phrase needs to match, 0 to 1. Defaults to 0.7.
    @discardableResult
    public func triggerThreshold(_ threshold: Float) -> Self {
        self.threshold = threshold
        return self
    }

    /// Combined download progress for every model, as a `0..1` fraction.
    @discardableResult
    public func onProgress(_ handler: @escaping @Sendable (Double, String) -> Void) -> Self {
        progressHandler = handler
        return self
    }

    /// Called with each thing the user says.
    @discardableResult
    public func onHeard(_ handler: @escaping @Sendable (String) -> Void) -> Self {
        heardHandlers.append(handler)
        return self
    }

    /// Called with each thing the assistant says.
    @discardableResult
    public func onSaid(_ handler: @escaping @Sendable (String) -> Void) -> Self {
        saidHandlers.append(handler)
        return self
    }

    /// Called when a flow throws something the runner doesn't handle itself.
    @discardableResult
    public func onError(_ handler: @escaping @Sendable (Error) -> Void) -> Self {
        errorHandlers.append(handler)
        return self
    }

    /// Replaces the built-in synthesizer, e.g. to route prompts somewhere else.
    @discardableResult
    public func speakWith(_ speak: @escaping @Sendable (String) async throws -> Void) -> Self {
        speakOverride = speak
        return self
    }

    /// Registers a flow to run when the user says something like `phrase`.
    @discardableResult
    public func listenFor(_ phrase: String, _ flow: @escaping FlowFunction) -> Self {
        if flows[phrase] == nil { flowOrder.append(phrase) }
        flows[phrase] = flow
        return self
    }

    /// Registers a handler that runs whenever `phrase` is heard, even in the
    /// middle of a flow and even with no flow running. The built-in `cancel`
    /// and `start over` work the same way but only while a flow is in progress;
    /// registering either here opts it into being live all the time.
    @discardableResult
    public func always(_ phrase: String, _ handler: @escaping FlowFunction) -> Self {
        if globals[phrase] == nil { globalOrder.append(phrase) }
        globals[phrase] = handler
        // Asking for a global by name means wanting it live, even if it is one
        // of the built-ins that is otherwise limited to a running flow.
        flowScopedGlobals.remove(phrase)
        return self
    }

    /// Registers a handler for speech that matched no global, no waiting
    /// prompt and no trigger. This is what a dictation interface hangs its
    /// text off: ``onHeard(_:)`` reports every line including commands and
    /// answers, while this one reports only the lines nothing else claimed.
    ///
    /// Nothing arrives here while a flow is running, because a flow's prompts
    /// take every line until it finishes.
    @discardableResult
    public func otherwise(_ handler: @escaping UnmatchedHandler) -> Self {
        unmatchedHandlers.append(handler)
        return self
    }

    @discardableResult
    public func useTextToSpeech(_ tts: TextToSpeech) -> Self {
        self.tts = tts
        ownsTts = false
        return self
    }

    @discardableResult
    public func useMicTranscriber(_ mic: MicTranscriber) -> Self {
        self.mic = mic
        ownsMic = false
        return self
    }

    // MARK: - Lifecycle

    /// Downloads and wires every model the agent needs.
    public func load() async throws {
        // A runner that has been silenced, or that speaks through a callback of
        // its own, has no use for a voice and should not spend a download on
        // one.
        if wantsSpeech, tts == nil, speakOverride == nil {
            let synthesizer = TextToSpeech().language(languageCode)
            if let voiceId { synthesizer.voice(voiceId) }
            if let modelDirectory { synthesizer.modelsFrom(modelDirectory) }
            if let progressHandler { synthesizer.onProgress(progressHandler) }
            try await synthesizer.load()
            tts = synthesizer
            ownsTts = true
        }

        if embedding == nil {
            let model = try await EmbeddingModel.load(
                cacheDirectory: modelDirectory,
                onProgress: fractionReporter(progressHandler))
            embedding = model
            matcher = PhraseMatcher(model: model)
        }

        if wantsMicrophone, mic == nil {
            let transcriber = MicTranscriber().language(languageCode).modelArch(arch)
            if let modelDirectory { transcriber.modelsFrom(modelDirectory) }
            if let progressHandler { transcriber.onProgress(progressHandler) }
            try await transcriber.load()
            mic = transcriber
            ownsMic = true
        }

        mic?.onLine { [weak self] line in
            guard let self else { return }
            // Don't let the assistant transcribe its own voice.
            guard !self.isSpeaking else { return }
            Task { await self.handleUtterance(line.text) }
        }
    }

    /// Opens the microphone and starts responding to trigger phrases.
    public func startListening() throws {
        guard let mic else {
            throw MoonshineError.custom(
                message:
                    "No microphone. Call load() first, or use handleUtterance() for text input.",
                code: -1)
        }
        try mic.start()
    }

    public func stopListening() throws {
        try mic?.stop()
    }

    /// Says something outside any flow, e.g. a welcome message.
    public func say(_ text: String) async throws {
        try await speak(text)
    }

    /// Speaks a reply that is still being written, a piece at a time.
    ///
    /// Each complete sentence starts playing while the rest is still arriving,
    /// so an LLM's answer can be forwarded token by token rather than waited
    /// for in full:
    ///
    /// ```swift
    /// try await flow.sayStream { push in
    ///     for try await token in llm.stream(question) {
    ///         push(token)
    ///     }
    /// }
    /// ```
    ///
    /// The microphone stays muted and self-capture stays suppressed for the
    /// whole passage, exactly as in ``say(_:)``, and this does not return until
    /// playback has finished. Falls back to collecting the text and speaking it
    /// in one go when there is no built-in synthesizer.
    public func sayStream(
        _ produce: (@escaping @Sendable (String) -> Void) async throws -> Void
    ) async throws {
        guard let tts else {
            let collected = Collected()
            try await produce { collected.append($0) }
            try await speak(collected.text)
            return
        }

        let collected = Collected()
        lock.withLock { speaking = true }
        mic?.mute(true)
        defer {
            // A no-op once the reply has drained; drops it if we left early.
            try? tts.cancelStream()
            // Leaving early also tore the reader down, and a cancellation is
            // held for whoever pulls next, so take it here rather than let the
            // next reply open with an interruption that was not its own.
            _ = try? tts.nextChunk()
            mic?.mute(false)
            lock.withLock { speaking = false }
            let text = collected.text
            if !text.isEmpty {
                for handler in saidHandlers { handler(text) }
            }
        }

        // Play chunks as they are produced while `produce` keeps pushing.
        let playback = Task.detached {
            for try await chunk in tts.chunks {
                try await tts.play(chunk)
            }
        }
        do {
            try await produce { piece in
                collected.append(piece)
                try? tts.pushText(piece)
            }
            try tts.endInput()
        } catch {
            playback.cancel()
            throw error
        }
        try await playback.value
    }

    /// Thread-safe accumulator for text pushed from a caller's producer.
    private final class Collected: @unchecked Sendable {
        private let lock = NSLock()
        private var pieces: [String] = []

        func append(_ piece: String) {
            lock.lock()
            pieces.append(piece)
            lock.unlock()
        }

        var text: String {
            lock.lock()
            defer { lock.unlock() }
            return pieces.joined().trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    /// Feeds in an utterance the agent didn't hear itself. Useful for text
    /// input and for tests. Returns once the flow has advanced as far as it can.
    public func handleUtterance(_ text: String) async {
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty else { return }
        for handler in heardHandlers { handler(utterance) }

        let task = lock.withLock { () -> Task<Void, Never> in
            let previous = queueTask
            let next = Task { [weak self] in
                await previous?.value
                await self?.dispatch(utterance)
            }
            queueTask = next
            return next
        }
        await task.value
    }

    /// True while a flow is running.
    public var isActive: Bool {
        return lock.withLock { activeDialog != nil }
    }

    /// The trigger phrase of the running flow, if any.
    public var activeTrigger: String? {
        return lock.withLock { activeTriggerPhrase }
    }

    /// Abandons the running flow. Returns false if there wasn't one.
    @discardableResult
    public func cancel() -> Bool {
        let wasActive = lock.withLock { () -> Bool in
            guard activeDialog != nil else { return false }
            activeDialog = nil
            activeTriggerPhrase = nil
            return true
        }
        guard wasActive else { return false }
        rejectPending(DialogCancelled())
        return true
    }

    public func close() {
        if ownsMic { mic?.close() }
        if ownsTts { tts?.close() }
        embedding?.close()
        mic = nil
        tts = nil
        embedding = nil
        matcher = PhraseMatcher(model: nil)
    }

    // MARK: - Internals used by Dialog

    private var isSpeaking: Bool {
        return lock.withLock { speaking }
    }

    /// Speaks a prompt, waits for an answer, and re-prompts until `interpret`
    /// accepts one or the retries run out.
    func promptForAnswer<T>(
        prompt: String,
        options: AskOptions,
        interpret: (String) -> T?
    ) async throws -> T {
        let maxRetries = options.maxRetries ?? 2
        let reprompt = options.reprompt ?? "Sorry, I didn't catch that. {prompt}"
        var attempt = 0
        while true {
            let line =
                attempt == 0
                ? prompt : reprompt.replacingOccurrences(of: "{prompt}", with: prompt)
            try await speak(line)

            let answer: String
            do {
                answer = try await waitForAnswer(timeout: options.timeout)
            } catch is DialogNoMatch where attempt < maxRetries {
                attempt += 1
                continue
            }

            if let value = interpret(answer.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return value
            }
            if attempt >= maxRetries {
                throw DialogNoMatch("Gave up understanding: \"\(answer)\"")
            }
            attempt += 1
        }
    }

    func speak(_ text: String) async throws {
        guard !text.isEmpty else { return }
        for handler in saidHandlers { handler(text) }
        lock.withLock { speaking = true }
        mic?.mute(true)
        defer {
            mic?.mute(false)
            lock.withLock { speaking = false }
        }
        if let speakOverride {
            try await speakOverride(text)
        } else if let tts {
            try await tts.say(text)
        } else {
            print("[AgentFlow] \(text)")
        }
    }

    // MARK: - Internals

    private func waitForAnswer(timeout: TimeInterval?) async throws -> String {
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<String, Error>) in
            lock.withLock {
                pending = continuation
                if let timeout {
                    pendingTimeout = Task { [weak self] in
                        try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                        guard !Task.isCancelled else { return }
                        self?.rejectPending(DialogNoMatch("Timed out waiting for an answer"))
                    }
                }
            }
            // The flow is now parked, so whoever handed us the last utterance
            // can stop waiting.
            notifySettled()
        }
    }

    /// Registers interest in the next time the runner finishes a flow or parks
    /// on a prompt. Registration is synchronous so the caller cannot miss a
    /// signal that lands before it gets around to awaiting.
    private func registerSettle() -> SettleSignal {
        let waiter = SettleSignal()
        lock.withLock { settleWaiters.append(waiter) }
        return waiter
    }

    private func notifySettled() {
        let waiters = lock.withLock { () -> [SettleSignal] in
            let pending = settleWaiters
            settleWaiters.removeAll()
            return pending
        }
        for waiter in waiters { waiter.signal() }
    }

    /// Detaches the parked continuation, so exactly one of resolve / reject
    /// ever resumes it.
    private func takePending() -> CheckedContinuation<String, Error>? {
        return lock.withLock { () -> CheckedContinuation<String, Error>? in
            guard let continuation = pending else { return nil }
            pending = nil
            pendingTimeout?.cancel()
            pendingTimeout = nil
            return continuation
        }
    }

    @discardableResult
    private func resolvePending(_ text: String) -> Bool {
        let continuation = takePending()
        guard let continuation else { return false }
        continuation.resume(returning: text)
        return true
    }

    private func rejectPending(_ error: Error) {
        takePending()?.resume(throwing: error)
    }

    private var hasPending: Bool {
        return lock.withLock { pending != nil }
    }

    private func dispatch(_ utterance: String) async {
        // Globals win over everything, so "cancel" works mid-question.
        let trigger = matchTrigger(utterance)
        if let trigger, globals[trigger] != nil {
            let settled = registerSettle()
            await invokeGlobal(trigger)
            // A global that cancelled or restarted the flow left it unwinding,
            // so wait for it to come to rest. One that just spoke did not.
            if !(isActive && !hasPending) {
                notifySettled()
            }
            await settled.wait()
            return
        }
        if hasPending {
            let settled = registerSettle()
            resolvePending(utterance)
            await settled.wait()
            return
        }
        // Busy between prompts: drop the line rather than interleave flows.
        if isActive { return }
        if let trigger, let flow = flows[trigger] {
            let settled = registerSettle()
            Task { await runFlow(trigger, flow: flow) }
            await settled.wait()
            return
        }
        // Nothing in the agent's domain wanted this line, so hand it to
        // whoever asked for the leftovers. Awaited, so that a handler doing
        // async work still sees utterances in the order they were spoken.
        for handler in unmatchedHandlers { await handler(utterance) }
    }

    private func matchTrigger(_ utterance: String) -> String? {
        let phrases = liveGlobals() + flowOrder
        guard !phrases.isEmpty else { return nil }
        return matcher.match(utterance, phrases: phrases, threshold: threshold)
    }

    /// The globals worth matching right now. Flow-scoped ones are offered only
    /// while a flow is running, so with nothing active their phrases reach
    /// ``otherwise(_:)`` like any other speech.
    private func liveGlobals() -> [String] {
        if flowScopedGlobals.isEmpty { return globalOrder }
        if isActive || hasPending { return globalOrder }
        return globalOrder.filter { !flowScopedGlobals.contains($0) }
    }

    /// The key of the group whose phrases best match `utterance`, used by
    /// ``Dialog/confirm(_:yesPhrases:noPhrases:options:)`` and
    /// ``Dialog/choose(_:options:settings:)``.
    func matchKey(_ utterance: String, groups: [(key: String, phrases: [String])]) -> String? {
        return matcher.match(utterance, groups: groups, threshold: AgentFlow.promptThreshold)
    }

    private func runFlow(_ triggerPhrase: String, flow: @escaping FlowFunction) async {
        defer {
            lock.withLock {
                activeDialog = nil
                activeTriggerPhrase = nil
            }
            notifySettled()
        }
        while true {
            let dialog = Dialog(runner: self, triggerPhrase: triggerPhrase)
            lock.withLock {
                activeDialog = dialog
                activeTriggerPhrase = triggerPhrase
            }
            do {
                try await flow(dialog)
                return
            } catch is DialogRestart {
                continue  // round again
            } catch is DialogCancelled {
                return
            } catch is DialogNoMatch {
                try? await speak("Sorry, I didn't get that. Let's start over.")
                return
            } catch {
                for handler in errorHandlers { handler(error) }
                return
            }
        }
    }

    private func invokeGlobal(_ triggerPhrase: String) async {
        guard let handler = globals[triggerPhrase] else { return }
        let existing = lock.withLock { activeDialog }
        let dialog = existing ?? Dialog(runner: self, triggerPhrase: triggerPhrase)
        do {
            try await handler(dialog)
        } catch {
            guard error is DialogCancelled || error is DialogRestart else {
                for callback in errorHandlers { callback(error) }
                return
            }
            // Hand the interruption to the flow, which is parked in an `await`.
            if hasPending {
                rejectPending(error)
            } else if error is DialogCancelled {
                lock.withLock {
                    activeDialog = nil
                    activeTriggerPhrase = nil
                }
            }
        }
    }
}

/// A one-shot "the runner came to rest" notification that can be signalled
/// before anyone waits on it.
private final class SettleSignal: @unchecked Sendable {
    private let lock = Mutex()
    private var signalled = false
    private var waiter: CheckedContinuation<Void, Never>?

    func signal() {
        let waiting = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            guard !signalled else { return nil }
            signalled = true
            let pending = waiter
            waiter = nil
            return pending
        }
        waiting?.resume()
    }

    func wait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let alreadySignalled = lock.withLock { () -> Bool in
                if signalled { return true }
                waiter = continuation
                return false
            }
            if alreadySignalled { continuation.resume() }
        }
    }
}

/// Renders a string as a space-separated spoken form for reading back.
public func spellOut(_ value: String) -> String {
    return value.map(String.init).joined(separator: " ")
}
