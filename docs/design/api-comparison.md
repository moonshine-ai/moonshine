# Client API: before and after

This document accompanies [api-principles.md](api-principles.md). It shows the
client code required for three common tasks, in each of the three client
languages, before and after the redesign.

The three tasks are:

1. **Setting up wifi using a voice interface** — a multi-turn dialog.
2. **Transcribing live speech** — streaming microphone transcription.
3. **Cloning a voice** — capturing a reference clip and synthesising with it.

Everything shown under "before" is real code from this repository as of the
redesign. Everything under "after" is the target API.

## Conventions

The redesign applies the same five rules to all three bindings.

### 1. Construct, configure, load

Constructors are cheap and cannot fail. Configuration happens through chainable
setters. `load()` is the single slow, failure-prone call, and it is the only one
the caller has to think about scheduling.

```javascript
const agent = new AgentFlow().language('es');
await agent.load();
```

This replaces static factories that take an options object and return a bundle
of objects (`AgentFlow.load({...})` in JavaScript, `MicTranscriber.load(...)`
in Swift, `CatalogLoader.load(context, specs, builder, callback)` on Android).

### 2. Nothing is required

Every constructor works with no arguments. Language defaults to `en`, the
speech-to-text model defaults to the best streaming model available for that
language, the voice defaults to the engine default, and assets come from the
CDN. `modelArch`, `variant`, `g2pRoot`, `audioContext`, and `ModelSpec` are gone
from the common path; the first three survive as optional setters, and the last
two are internal.

### 3. Flow bodies are procedural

An agent flow reads top to bottom in every language. JavaScript uses `async` /
`await` (replacing the generator-and-`yield` protocol), Swift uses
`async throws`, and Java uses ordinary blocking calls that the app runs on its
own executor.

### 4. Named callbacks, not listener objects

`onText`, `onLine`, `onProgress`, and `onError` replace
`TranscriptEventListener` with its six optional methods and six event structs.
The full event interface remains available via `addListener` for applications
that need line ids, speaker spans, or word timings. On Android the named
callbacks are delivered on the main thread, so `runOnUiThread` disappears from
application code.

### 5. Offline is possible, just longer

`.modelsFrom(...)` opts out of the CDN. It takes the type that platform actually
uses to name a location: a directory `File` on Android, a directory `URL` in
Swift, and a base URL string in the browser (where there is no filesystem to
point at) — or, on the web, a per-file map when the names differ. This is the
only common operation the redesign makes more verbose, which is the trade the
principles document asks for.

### Naming

The same concepts now have the same names everywhere.

- `MicTranscriber` in all four bindings. JavaScript's
  `MicrophoneTranscriber` is renamed.
- `AgentFlow` in all four bindings. It did not exist in Swift or Java.
- `TextToSpeech.cloneFrom(...)` in all four bindings.
- `TextToSpeech.say(...)` speaks out loud and `synthesize(...)` returns the
  samples, in all four bindings. JavaScript's `say` used to be the one that
  returned samples, which meant the same call did different things depending on
  which binding you were reading.
- `load()` in all four bindings, never `loadFromCatalog`, `loadFromFiles`,
  `loadFromUrls`, or `loadFromMemory` on the common path.
- Progress is reported as a `0..1` fraction plus the file being fetched,
  rather than `(loaded, total, file)` or a `DownloadProgress` struct.

## Task 1: setting up wifi using a voice interface

### JavaScript, before

From `examples/web/agent-flow/index.html`. The flow is a generator whose
`yield`ed prompts are resumed with the user's answers. The application owns the
`AudioContext`, the text-to-speech instance, and the function that connects the
two, and it unpacks three objects from the returned bundle.

```javascript
function* wifiSetup(d) {
  const ssid = yield d.ask("What's the name of your wifi network?");
  if (!(yield d.confirm(`I heard ${ssid}. Is that right?`))) {
    yield d.say("No problem, let's start over.");
    return;
  }
  if (yield d.confirm('Apply these changes?')) {
    yield d.say(`Done. Connecting to ${ssid}.`);
  } else {
    yield d.say('Okay, nothing changed.');
  }
}

let tts = null;
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

const bundle = await AgentFlow.load({
  language: 'en',
  modelArch: ModelArch.MediumStreaming,
  microphone: true,
  audioContext,
  onProgress: (loaded, total, file) => {
    const pct = total ? ` ${Math.round((100 * loaded) / total)}%` : '';
    statusEl.textContent = `Downloading ${file}${pct}…`;
  },
  flows: { 'set up wifi': wifiSetup },
  globals: {
    cancel: (d) => d.cancel(),
    'start over': (d) => d.restart(),
  },
  micListeners: [{ onLineCompleted: (e) => log('user', e.line.text) }],
  speakFn: (text) => {
    log('assistant', text);
    try { tts?.say(text); } catch { /* synth is best-effort here */ }
  },
});

const runner = bundle.agent;
tts = bundle.tts;
const mic = bundle.mic ?? null;
if (mic) {
  await mic.start();
}
```

### JavaScript, after

```javascript
const agent = new AgentFlow();

agent.listenFor('set up wifi', async (d) => {
  const ssid = await d.ask("What's the name of your wifi network?");
  if (!(await d.confirm(`I heard ${ssid}. Is that right?`))) {
    await d.say("No problem, let's start over.");
    return;
  }
  if (await d.confirm('Apply these changes?')) {
    await d.say(`Done. Connecting to ${ssid}.`);
  } else {
    await d.say('Okay, nothing changed.');
  }
});

await agent.load();
await agent.startListening();
```

`cancel` and `start over` are built in, so applications no longer register them.
Speaking is internal, so there is no `speakFn` and no application-visible
text-to-speech handle. Applications that want a conversation log attach
`agent.onHeard(...)` and `agent.onSaid(...)`; applications that want a
progress bar attach `agent.onProgress(...)`.

### Swift, before

There is no `AgentFlow` in Swift. An application has to download two models,
build three engines, write a listener class to hop transcript events onto the
main actor, match intents itself, and implement its own turn-taking. This is the
shape of it, following the `examples/ios/IntentRecognizer` sample that this
redesign removed:

```swift
final class WifiSession {
    private var mic: MicTranscriber?
    private var intents: IntentRecognizer?
    private var tts: TextToSpeech?
    private var bridge: TranscriptBridge?

    func bootstrap() async throws {
        let sttSpec = ModelSpec.stt(language: "en", modelArch: .mediumStreaming)
        let intentSpec = ModelSpec.intent(variant: "q4")
        let directories = try await Moonshine.prepareModels([sttSpec, intentSpec])

        intents = try IntentRecognizer(
            modelPath: directories[intentSpec]!.path,
            modelArch: .gemma300m,
            modelVariant: "q4"
        )
        try intents?.registerIntent(canonicalPhrase: "set up wifi")
        try intents?.registerIntent(canonicalPhrase: "cancel")

        tts = try await TextToSpeech.load(language: "en")

        bridge = TranscriptBridge(session: self)
        mic = try MicTranscriber(
            modelPath: directories[sttSpec]!.path,
            modelArch: .mediumStreaming
        )
        mic?.addListener(bridge!)
        try mic?.start()
    }

    // Called from the bridge for every completed line. The application has to
    // track which question is outstanding, re-prompt on no match, mute the
    // microphone while the synthesiser is talking, and so on.
    func handleCompletedLine(_ text: String) {
        guard let matches = try? intents?.getClosestIntents(
            utterance: text, toleranceThreshold: 0.7
        ), let top = matches.first else { return }
        switch state {
        case .idle where top.canonicalPhrase == "set up wifi":
            state = .awaitingSsid
            tts?.say("What's the name of your wifi network?")
        case .awaitingSsid:
            ssid = text
            state = .confirmingSsid
            tts?.say("I heard \(text). Is that right?")
        // ... one case per turn, plus retries, timeouts and cancellation
        default:
            break
        }
    }
}

final class TranscriptBridge: TranscriptEventListener {
    weak var session: WifiSession?
    func onLineCompleted(_ event: LineCompleted) {
        Task { @MainActor in session?.handleCompletedLine(event.line.text) }
    }
}
```

### Swift, after

```swift
let agent = AgentFlow()

agent.listenFor("set up wifi") { d in
    let ssid = try await d.ask("What's the name of your wifi network?")
    guard try await d.confirm("I heard \(ssid). Is that right?") else {
        try await d.say("No problem, let's start over.")
        return
    }
    if try await d.confirm("Apply these changes?") {
        try await d.say("Done. Connecting to \(ssid).")
    } else {
        try await d.say("Okay, nothing changed.")
    }
}

try await agent.load()
try agent.startListening()
```

### Android, before

There is no `AgentFlow` in Java either. The closest existing code was the
`examples/android/IntentRecognizer` sample that this redesign removed, which
nests a `CatalogLoader.Builder` inside a `LoadCallback` inside
`CatalogLoader.load`, builds each engine by hand out of a
`Map<ModelSpec, File>`, and requires a separate `onMicPermissionGranted()`
handshake after the runtime permission dialog:

```java
ModelSpec sttSpec = ModelSpec.stt("en", JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, false);
ModelSpec intentSpec = ModelSpec.intent(null, "q4");

CatalogLoader.load(
    this,
    Arrays.asList(sttSpec, intentSpec),
    directories -> {
        IntentRecognizer recognizer = new IntentRecognizer(
            directories.get(intentSpec).getAbsolutePath(),
            JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
            "q4");
        recognizer.registerIntent("set up wifi");
        recognizer.registerIntent("cancel");

        MicTranscriber mic = new MicTranscriber();
        mic.addListener(transcriptListener);
        mic.loadFromFiles(
            directories.get(sttSpec).getAbsolutePath(),
            JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING);
        return new Engines(recognizer, mic);
    },
    new LoadCallback<Engines>() {
        @Override public void onSuccess(Engines engines) {
            this.engines = engines;
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED) {
                engines.mic.onMicPermissionGranted();
            }
        }
        @Override public void onError(Throwable error) { showError(error); }
    });

// Text-to-speech is a second, entirely separate load, and the turn-taking
// state machine is written by hand in onLineCompleted, exactly as in Swift.
```

### Android, after

```java
AgentFlow agent = new AgentFlow(this);

agent.listenFor("set up wifi", d -> {
    String ssid = d.ask("What's the name of your wifi network?");
    if (!d.confirm("I heard " + ssid + ". Is that right?")) {
        d.say("No problem, let's start over.");
        return;
    }
    if (d.confirm("Apply these changes?")) {
        d.say("Done. Connecting to " + ssid + ".");
    } else {
        d.say("Okay, nothing changed.");
    }
});

executor.execute(() -> {
    agent.load();
    agent.startListening();
});
```

The flow body blocks, which is what makes it readable; it runs on a worker
thread owned by the agent, never on the main thread. The application only has
to decide where `load()` runs, which is the "procedural code wrapped in an
asynchronous block" the principles ask for. In Kotlin the wrapper is
`lifecycleScope.launch(Dispatchers.IO) { agent.load(); agent.startListening() }`.

## Task 2: transcribing live speech

### JavaScript, before

From `examples/web/stt/index.html`:

```javascript
const listener = {
  onLineTextChanged: (e) => render(e.line.text),
  onLineCompleted: (e) => { lines.set(e.line.id, e.line.text); render(); },
  onError: (e) => { statusEl.textContent = 'Error: ' + e.error.message; },
};

const mic = await MicrophoneTranscriber.load({
  language: 'en',
  modelArch: ModelArch.MediumStreaming,
  listeners: [listener],
  onProgress: (loaded, total, file) => {
    const pct = total ? ` ${Math.round((100 * loaded) / total)}%` : '';
    statusEl.textContent = `Downloading ${file}${pct}…`;
  },
});
await mic.start();
```

### JavaScript, after

```javascript
const mic = new MicTranscriber()
  .onText((text) => renderLive(text))
  .onLine((line) => appendFinal(line.text))
  .onProgress((fraction, file) => {
    statusEl.textContent = `Downloading ${file} ${Math.round(100 * fraction)}%…`;
  });

await mic.load();
await mic.start();
```

### Swift, before

From `examples/macos/MicTranscription`, which needs a listener class because
`TranscriptEventListener` is a protocol:

```swift
class TestListener: TranscriptEventListener {
    func onLineTextChanged(_ event: LineTextChanged) {
        print(event.line.text, terminator: "\r")
    }
    func onLineCompleted(_ event: LineCompleted) {
        print(event.line.text)
    }
}

let micTranscriber = try await MicTranscriber.load(
    language: "en",
    modelArch: .mediumStreaming
) { progress in
    let pct = progress.bytesTotal > 0
        ? Int(progress.bytesDownloaded * 100 / progress.bytesTotal) : 0
    fputs("\r  \(progress.relativePath) [\(progress.fileIndex)/\(progress.totalFiles)] \(pct)%", stderr)
}
let listener = TestListener()
micTranscriber.addListener(listener)
try micTranscriber.start()
```

### Swift, after

Either the closure form:

```swift
let mic = MicTranscriber()
    .onText { print($0, terminator: "\r") }
    .onLine { print($0.text) }

try await mic.load()
try mic.start()
```

or, where an `AsyncSequence` reads better:

```swift
let mic = MicTranscriber()
try await mic.load()
try mic.start()

for try await line in mic.transcript {
    print(line.text)
}
```

### Android, before

From the removed `examples/android/IntentRecognizer`. The listener is a
`Consumer` wrapping
a visitor, every callback hops to the main thread by hand, model loading is a
nested builder-inside-callback, and the microphone needs a permission handshake
that is separate from the Android runtime permission grant:

```kotlin
private val transcriptListener = java.util.function.Consumer<TranscriptEvent> { event ->
    event.accept(
        object : TranscriptEventListener() {
            override fun onLineStarted(e: TranscriptEvent.LineStarted) {
                runOnUiThread { binding.liveTranscript.text = e.line.text.orEmpty() }
            }
            override fun onLineTextChanged(e: TranscriptEvent.LineTextChanged) {
                runOnUiThread { binding.liveTranscript.text = e.line.text.orEmpty() }
            }
            override fun onLineCompleted(e: TranscriptEvent.LineCompleted) {
                runOnUiThread { handleCompletedTranscriptLine(e.line.text.orEmpty()) }
            }
        },
    )
}

val sttSpec = ModelSpec.stt("en", JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, false)
CatalogLoader.load(
    this,
    listOf(sttSpec),
    CatalogLoader.Builder<MicTranscriber> { directories ->
        val m = MicTranscriber()
        m.addListener(transcriptListener)
        m.loadFromFiles(
            directories[sttSpec]!!.absolutePath,
            JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
        )
        m
    },
    object : LoadCallback<MicTranscriber> {
        override fun onSuccess(engine: MicTranscriber) {
            mic = engine
            if (ContextCompat.checkSelfPermission(
                    this@MainActivity, Manifest.permission.RECORD_AUDIO,
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                engine.onMicPermissionGranted()
            }
        }
        override fun onError(error: Throwable) { showError(error) }
    },
)

// ...and later, once the user has granted the permission and tapped Listen:
mic.onMicPermissionGranted()
mic.start()
```

### Android, after

```kotlin
val mic = MicTranscriber(this)
    .onText { binding.liveTranscript.text = it }
    .onLine { handleCompletedTranscriptLine(it.text) }

lifecycleScope.launch(Dispatchers.IO) {
    mic.load()
    mic.start()
}
```

`onText` and `onLine` are delivered on the main thread. `start()` requests the
`RECORD_AUDIO` permission itself when it is given an `Activity`, and blocks
until the user answers; given a plain `Context` it throws
`MissingPermissionException` if the permission has not already been granted.
`onMicPermissionGranted()` is gone.

## Task 3: cloning a voice

### JavaScript, before

From `examples/web/tts/index.html`. The application opens the microphone, runs a
transcription-suppressed `Transcriber` purely for its voice-activity detector,
implements `findSpeechWindow` to locate the start of speech, slices four seconds
of PCM itself, loads a *second* speech-to-text model to transcribe that slice,
and only then constructs the synthesiser. Roughly 120 lines in total; the core
of it is:

```javascript
const vad = await Transcriber.load({
  files: new Map(), // no model files needed in skip_transcription mode
  modelArch: ModelArch.Tiny,
  options: { skip_transcription: 'true' },
  module,
});
const stream = vad.createStream();
stream.start();

node.onaudioprocess = (event) => {
  const rs = resampleTo16k(
    new Float32Array(event.inputBuffer.getChannelData(0)), inputRate);
  chunks.push(rs);
  total += rs.length;
  const seconds = total / TARGET_SR;
  stream.addAudio(rs, TARGET_SR);
  const transcript = stream.transcribe();
  const start = findSpeechWindow(transcript.lines, seconds);
  if (start != null) {
    settle({ start });
  } else if (seconds >= CLONE_MAX_RECORD_SEC) {
    settle({ start: Math.max(0, seconds - CLONE_WINDOW_SEC) });
  }
};

// ...then assemble the recording, slice it, and transcribe the slice:
const clip = recording.slice(from, from + CLONE_WINDOW_SEC * TARGET_SR);
const stt = await ensureClipTranscriber();
const cloneTranscript = stt
  .transcribe(clip, { sampleRate: TARGET_SR })
  .lines.map((l) => l.text)
  .join(' ')
  .trim();

const tts = await TextToSpeech.load({
  language: LANGUAGE,
  clone: { audio: clip, sampleRate: TARGET_SR, transcript: cloneTranscript },
});
```

### JavaScript, after

One-shot, from a URL, a `File`, a `Blob`, an `AudioBuffer`, or a
`Float32Array`:

```javascript
const tts = new TextToSpeech();
await tts.load();
await tts.cloneFrom('some-speech.wav');
await tts.say('Hello world!');
```

Streaming, for capturing a reference clip from a live microphone, with the
readiness flag the principles ask for:

```javascript
const tts = new TextToSpeech();
await tts.load();

const clone = tts.startCloning();
clone.onReady(() => { statusEl.textContent = 'Got it — you can stop talking.'; });
await clone.fromMicrophone();   // resolves once there is enough speech

await tts.cloneFrom(clone);
await tts.say('Hello world!');
```

An application driving its own audio pipeline pushes samples in instead of
calling `fromMicrophone()`, and polls or listens for readiness:

```javascript
const clone = tts.startCloning();
for await (const chunk of myAudioSource) {
  clone.addAudio(chunk, 48000);
  if (clone.isReady) break;
}
await tts.cloneFrom(clone);
```

### Swift, before

There is no example; the only demonstration is a unit test, and it starts from
PCM the caller has already trimmed and transcribed:

```swift
let tts = try TextToSpeech(
    language: "en_us",
    g2pRoot: dataPath,
    clonePCM: pcm,
    cloneSampleRate: 24000,
    cloneTranscript: "This is a reference clip."
)
let result = try tts.synthesize(text: "Cloning a custom voice.")
```

### Swift, after

```swift
let tts = TextToSpeech()
try await tts.load()
try await tts.cloneFrom(url)
try await tts.say("Hello world!")
```

```swift
let clone = tts.startCloning()
clone.onReady { statusText = "Got it — you can stop talking." }
try await clone.fromMicrophone()
try await tts.cloneFrom(clone)
try await tts.say("Hello world!")
```

### Android, before

Again only a test, again starting from pre-trimmed PCM and a hand-written
transcript:

```java
TextToSpeech tts = TextToSpeech.fromZipVoiceClone(
    "en_us", pcm, 24000, "This is a reference clip.", root, null);
TtsSynthesisResult result = tts.synthesize("Cloning a custom voice.");
```

### Android, after

This is the example from the principles document, and it now compiles:

```java
TextToSpeech tts = new TextToSpeech(context);
tts.load();
tts.cloneFrom("some-speech.wav");
tts.say("Hello world!");
```

```java
VoiceClone clone = tts.startCloning();
clone.onReady(() -> status.setText("Got it — you can stop talking."));
clone.fromMicrophone();
tts.cloneFrom(clone);
tts.say("Hello world!");
```

## What makes the "after" column possible

Three pieces of shared machinery, so that none of this is reimplemented per
language.

**Speech-clip extraction moves into the C++ core.** `findSpeechWindow` and the
transcription-suppressed transcriber trick in the web demo become
`moonshine_extract_speech_clip`, which runs the built-in Silero voice-activity
detector over arbitrary PCM and returns the first `max_duration_seconds` of
detected speech, resampled to 16 kHz mono. It also reports whether it has seen a
full clip's worth of speech yet, which is what backs `VoiceClone.isReady` in all
three bindings. The Silero model is compiled into the library, so this needs no
downloads.

**Clone-clip transcription is already in the core.** The C API accepts
`zipvoice_asr_transcriber_handle`, so `cloneFrom` can load a small speech-to-text
model internally and hand its handle to the synthesiser rather than making the
application transcribe the reference clip. The `TextToSpeech` load path pulls
that model down alongside the voice assets when cloning is requested.

**`AgentFlow` is ported from Python and JavaScript to Swift and Java.** The
routing, retry, re-prompt, and cancellation logic in
`python/src/moonshine_voice/agent_flow.py` and `wasm/src/agent-flow.ts` is the
reference. Only the suspension mechanism differs per language: a promise in
JavaScript, a continuation in Swift, and a blocking queue in Java.

## Compatibility

This is a breaking change to all three client bindings. The low-level types
(`Transcriber`, `Stream`, `GraphemeToPhonemizer`, `AssetDownloader`) keep their
existing behaviour and remain public for applications that need them, but the
entry points named above change shape, and `MicrophoneTranscriber` is renamed in
JavaScript.

`IntentRecognizer` is the exception: it becomes internal in all four bindings,
renamed `EmbeddingModel` and trimmed to embedding a sentence and scoring two
embeddings against each other. It was only ever a way to reach the embedding
model, and `AgentFlow` now owns one on the application's behalf, loading it on
first use and comparing utterances to phrases itself. Keeping it public would
have meant two ways to do trigger matching, one of which requires the caller to
assemble the pieces. The intent-recognition sample apps go with it, and the
Raspberry Pi `my-dalek` demo is ported to `AgentFlow` globals.

Python's `AgentFlow` moves to the same construct-configure-load pattern, so all
four bindings read the same. It predates this work, and used to be constructed
with a dozen keyword arguments and handed a `TextToSpeech` and a
`MicTranscriber` the application had built itself; it now opens all three models
on `load()` and its microphone on `start_listening()`. The registration and
routing methods are renamed to match the other bindings — `register_flow` to
`listen_for`, `register_global` to `always`, `process_utterance` to
`handle_utterance`, and `cancel_active` to `cancel` — and the constructor
arguments become chainable setters. This is a breaking change for the Python
binding too.

The flows themselves are untouched: they stay generator functions driven by
`yield`, which is the idiomatic Python equivalent of the promise, continuation
and blocking-queue mechanisms used elsewhere, and the whole engine is built
around it.
