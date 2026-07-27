import Foundation

/// Voice dialogs: the one-call way to build a speech interface.
///
/// ```swift
/// let dialog = DialogFlow()
///
/// dialog.listenFor("set up wifi") { d in
///     let ssid = try await d.ask("What's the name of your wifi network?")
///     if try await d.confirm("I heard \(ssid). Is that right?") {
///         try await d.say("Done. Connecting to \(ssid).")
///     }
/// }
///
/// try await dialog.load()
/// try await dialog.startListening()
/// ```
///
/// ``load()`` downloads and wires everything a voice interface needs: a
/// streaming speech-to-text model, an intent model for matching trigger
/// phrases, a text-to-speech voice, and a microphone. A flow is an ordinary
/// async function, so it reads top to bottom and `try` / `defer` work the way
/// you expect.

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

    private unowned let runner: DialogFlow

    init(runner: DialogFlow, triggerPhrase: String = "") {
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
        return try await runner.promptForAnswer(prompt: prompt, options: settings) { text in
            if matchesAny(text, yesPhrases) { return true }
            if matchesAny(text, noPhrases) { return false }
            return nil
        }
    }

    /// Offers a set of choices and returns the key of the one picked. Each key
    /// maps to the phrases that select it; the key itself always counts.
    public func choose(
        _ prompt: String,
        options choices: [String: [String]],
        settings: AskOptions = AskOptions()
    ) async throws -> String {
        return try await runner.promptForAnswer(prompt: prompt, options: settings) { text in
            for (key, phrases) in choices {
                if matchesAny(text, [key] + phrases) { return key }
            }
            return nil
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

@available(iOS 15.0, macOS 12.0, *)
public final class DialogFlow: @unchecked Sendable {
    private static let defaultTriggerThreshold: Float = 0.7

    private var flowOrder: [String] = []
    private var flows: [String: FlowFunction] = [:]
    private var globalOrder: [String] = []
    private var globals: [String: FlowFunction] = [:]

    private var languageCode = "en"
    private var arch: ModelArch = .mediumStreaming
    private var voiceId: String?
    private var wantsMicrophone = true
    private var threshold = DialogFlow.defaultTriggerThreshold
    private var modelDirectory: URL?
    private var progressHandler: (@Sendable (Double, String) -> Void)?
    private var speakOverride: (@Sendable (String) async throws -> Void)?
    private var heardHandlers: [@Sendable (String) -> Void] = []
    private var saidHandlers: [@Sendable (String) -> Void] = []
    private var errorHandlers: [@Sendable (Error) -> Void] = []

    private var tts: TextToSpeech?
    private var intent: IntentRecognizer?
    private var mic: MicTranscriber?
    private var ownsTts = true
    private var ownsMic = true

    private let lock = Mutex()
    private var activeDialog: Dialog?
    private var activeTriggerPhrase: String?
    private var pending: CheckedContinuation<String, Error>?
    private var pendingTimeout: Task<Void, Never>?
    private var speaking = false
    private var triggersRegistered = false
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
        always("cancel") { dialog in try dialog.cancel() }
        always("start over") { dialog in try dialog.restart() }
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

    /// Set to false to drive the dialog from text instead of a microphone.
    @discardableResult
    public func microphone(_ enabled: Bool) -> Self {
        wantsMicrophone = enabled
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
        triggersRegistered = false
        return self
    }

    /// Registers a handler that runs whenever `phrase` is heard, even in the
    /// middle of a flow. This is how `cancel` and `start over` are implemented.
    @discardableResult
    public func always(_ phrase: String, _ handler: @escaping FlowFunction) -> Self {
        if globals[phrase] == nil { globalOrder.append(phrase) }
        globals[phrase] = handler
        triggersRegistered = false
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

    /// Downloads and wires every model the dialog needs.
    public func load() async throws {
        if tts == nil {
            let synthesizer = TextToSpeech().language(languageCode)
            if let voiceId { synthesizer.voice(voiceId) }
            if let modelDirectory { synthesizer.modelsFrom(modelDirectory) }
            if let progressHandler { synthesizer.onProgress(progressHandler) }
            try await synthesizer.load()
            tts = synthesizer
            ownsTts = true
        }

        if intent == nil {
            intent = try await IntentRecognizer.load(
                cacheDirectory: modelDirectory,
                onProgress: fractionReporter(progressHandler))
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

    /// Feeds in an utterance the dialog didn't hear itself. Useful for text
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
        intent?.close()
        mic = nil
        tts = nil
        intent = nil
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
            print("[DialogFlow] \(text)")
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
        }
    }

    private func matchTrigger(_ utterance: String) -> String? {
        let phrases = globalOrder + flowOrder
        guard !phrases.isEmpty else { return nil }

        if let intent {
            if !triggersRegistered {
                try? intent.clearIntents()
                for phrase in phrases {
                    try? intent.registerIntent(canonicalPhrase: phrase)
                }
                triggersRegistered = true
            }
            let matches = try? intent.getClosestIntents(
                utterance: utterance, toleranceThreshold: threshold)
            return matches?.first?.canonicalPhrase
        }
        let lower = utterance.lowercased()
        return phrases.first { lower.contains($0.lowercased()) }
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

private func matchesAny(_ utterance: String, _ phrases: [String]) -> Bool {
    let lower = utterance.lowercased()
    return phrases.contains { phrase in
        let needle = phrase.lowercased()
        return lower == needle || lower.contains(needle)
    }
}

/// Renders a string as a space-separated spoken form for reading back.
public func spellOut(_ value: String) -> String {
    return value.map(String.init).joined(separator: " ")
}
