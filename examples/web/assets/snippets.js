/**
 * Code samples for the language tabs on the demo pages.
 *
 * These are hand-copied from the real example programs each entry names in its
 * `path`, and nothing yet checks that they still match. Phase 4 of
 * docs/design/api-regularization.md replaces this file with a generated one
 * built by walking `examples/` for `snippet:` region markers, at which point a
 * sample that drifts from its example fails the build. The shape below is what
 * the generator will emit, so the pages should not need touching again.
 *
 * Keeping application-specific identifiers (`rewriter.partial`,
 * `transcriptText`) is deliberate: a reader learns more from the real call than
 * from an invented placeholder.
 *
 * `lines` is the region of `path` the sample was taken from, and becomes the
 * `#L12-L34` anchor on the caption link. Line numbers drift as the examples are
 * edited, which is why a test re-reads each range and fails if the calls the
 * snippet shows are no longer inside it. Widen or move the range rather than
 * dropping it; a link to the wrong lines is worse than a link to the file.
 */

/** Install lines, one per language tab. */
export const INSTALL = {
  javascript: 'npm i @moonshine-ai/moonshine-wasm',
  python: 'pip install moonshine-voice',
  swift: 'https://github.com/moonshine-ai/moonshine-swift/',
  java: 'ai.moonshine:moonshine-voice:0.1.1',
};

/** Says where the install line goes, for the tabs where that is not obvious. */
export const INSTALL_HINT = {
  swift: 'Xcode ▸ Add Package Dependencies…',
  java: 'build.gradle.kts dependencies',
};

/** Attaches the install line and hint that go with each tab's language. */
export function withInstall(snippets) {
  return snippets.map((snippet) => ({
    ...snippet,
    install: INSTALL[snippet.id],
    installHint: INSTALL_HINT[snippet.id],
  }));
}

/** Live microphone transcription. */
export const MIC_TRANSCRIPTION = [
  {
    id: 'javascript',
    label: 'JavaScript',
    file: 'live-transcription.js',
    path: 'examples/web/stt/index.html',
    lines: [334, 356],
    code: `import { MicTranscriber, ModelArch } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';

const mic = new MicTranscriber()
  .modelArch(ModelArch.MediumStreaming)
  .onText((text) => showInProgress(text))
  .onLine((line) => appendLine(line.text, line.lastTranscriptionLatencyMs));

await mic.load();   // downloads the model once, then it is cached
await mic.start();  // opens the microphone and starts transcribing`,
  },
  {
    id: 'python',
    label: 'Python',
    file: 'mic_transcription.py',
    path: 'examples/python/mic_transcription.py',
    lines: [41, 63],
    code: `from moonshine_voice import MicTranscriber

mic = (MicTranscriber()
       .on_text(rewriter.partial)
       .on_line(rewriter.final))

mic.load()  # downloads the model once, then it is cached

with mic:
    mic.start()  # opens the microphone and starts transcribing
    while True:
        time.sleep(0.1)`,
  },
  {
    id: 'swift',
    label: 'Swift',
    file: 'TranscriberApp.swift',
    path: 'examples/ios/Transcriber/Transcriber/TranscriberApp.swift',
    lines: [41, 63],
    code: `import MoonshineVoice

let mic = MicTranscriber()
    .onText { [weak self] text in
        Task { @MainActor in self?.liveText = text }
    }
    .onLine { [weak self] line in
        Task { @MainActor in self?.lines.append(line.text) }
    }

try await mic.load()  // downloads the model once, then it is cached
try mic.start()       // opens the microphone and starts transcribing`,
  },
  {
    id: 'java',
    label: 'Android',
    file: 'MainActivity.java',
    path: 'examples/android/Transcriber/app/src/main/java/ai/moonshine/androidtranscriber/MainActivity.java',
    lines: [36, 56],
    code: `import ai.moonshine.voice.MicTranscriber;

mic = new MicTranscriber(this)
        .onText(text -> transcriptText.setText(finishedLines + text))
        .onLine(line -> {
            finishedLines.append(line.text).append('\\n');
            transcriptText.setText(finishedLines.toString());
        });

worker.execute(() -> {
    mic.load();   // downloads the model once, then it is cached
    mic.start();  // opens the microphone and starts transcribing
});`,
  },
];

/**
 * Speech synthesis, and cloning a voice from a recording.
 *
 * Every tab clones, and each links to an example app that does. The mobile two
 * record the reference clip from the microphone rather than reading a file, so
 * their `cloneFrom` sits a few methods away from the `load` the anchor points
 * at; `startCloning` and `fromMicrophone` are the calls to look for there.
 */
export const TEXT_TO_SPEECH = [
  {
    id: 'javascript',
    label: 'JavaScript',
    file: 'speak.js',
    path: 'examples/web/tts/index.html',
    lines: [465, 509],
    code: `import { TextToSpeech } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';

const tts = new TextToSpeech().language('en_us').voice('kokoro_af_heart');
await tts.load();
await tts.say('Hello from Moonshine.');

const cloneTts = new TextToSpeech().language('en_us').cloning();
await cloneTts.load();
await cloneTts.cloneFrom(recording);
await cloneTts.say('Now I sound like you.');`,
  },
  {
    id: 'python',
    label: 'Python',
    file: 'text_to_speech.py',
    path: 'examples/python/text_to_speech.py',
    lines: [64, 82],
    code: `from moonshine_voice import TextToSpeech

tts = TextToSpeech().language('en_us').voice('kokoro_af_heart')
tts.load()
tts.say('Hello from Moonshine.')

clone_tts = TextToSpeech().language('en_us').cloning()
clone_tts.load()
clone_tts.clone_from(recording)
clone_tts.say('Now I sound like you.')`,
  },
  {
    id: 'swift',
    label: 'Swift',
    file: 'TextToSpeechApp.swift',
    path: 'examples/ios/TextToSpeech/TextToSpeech/TextToSpeechApp.swift',
    lines: [187, 204],
    code: `import MoonshineVoice

let tts = MoonshineVoice.TextToSpeech()
    .language("en_us")
    .voice("kokoro_af_heart")

try await tts.load()
try await tts.say("Hello from Moonshine.")

let cloneTts = MoonshineVoice.TextToSpeech()
    .language("en_us")
    .cloning()
try await cloneTts.load()
try await cloneTts.cloneFrom(recording)
try await cloneTts.say("Now I sound like you.")`,
  },
  {
    id: 'java',
    label: 'Android',
    file: 'MainActivity.java',
    path: 'examples/android/TextToSpeech/app/src/main/java/ai/moonshine/examples/texttospeech/MainActivity.java',
    lines: [281, 299],
    code: `import ai.moonshine.voice.TextToSpeech;

TextToSpeech tts = new TextToSpeech(this)
        .language("en_us")
        .voice("kokoro_af_heart");

worker.execute(() -> {
    tts.load();
    tts.say("Hello from Moonshine.");

    TextToSpeech cloneTts = new TextToSpeech(this)
        .language("en_us")
        .cloning();
    cloneTts.load();
    cloneTts.cloneFrom(recording);
    cloneTts.say("Now I sound like you.");
});`,
  },
];

/**
 * The wifi setup flow the voice agent page runs.
 *
 * Unlike the other two samples these are not lifted verbatim: each is the
 * page's own flow written against that language's API, trimmed the same way the
 * JavaScript listing is trimmed from `examples/python/agent_flow.py` and
 * `examples/macos/AgentFlow`. That is what lets the page light up the line the
 * conversation is parked on in whichever language is showing.
 *
 * `steps` maps each awaited call to its line, counting from zero. Editing a
 * snippet means renumbering its steps, and a test checks every one of them
 * lands on a line that mentions the call it claims to be.
 */
export const AGENT_FLOW = [
  {
    id: 'javascript',
    label: 'JavaScript',
    path: 'examples/web/agent-flow/index.html',
    lines: [402, 421],
    steps: { askSsid: 3, confirmSsid: 5, startOver: 6, confirmApply: 9, done: 10, unchanged: 12 },
    code: `import { AgentFlow } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';

agent.listenFor('set up wifi', async (d) => {
  const ssid = await d.ask("What's the name of your wifi network?");

  if (!(await d.confirm(\`I heard \${ssid}. Is that right?\`))) {
    return d.say("No problem, let's start over.");
  }

  if (await d.confirm('Apply these changes?')) {
    await d.say(\`Done. Connecting to \${ssid}.\`);
  } else {
    await d.say('Okay, nothing changed.');
  }
});`,
  },
  {
    id: 'python',
    label: 'Python',
    path: 'examples/python/agent_flow.py',
    lines: [30, 51],
    steps: { askSsid: 3, confirmSsid: 5, startOver: 6, confirmApply: 9, done: 10, unchanged: 12 },
    code: `from moonshine_voice import AgentFlow

def setup_wifi(d):
    ssid = yield d.ask("What's the name of your wifi network?")

    if not (yield d.confirm(f"I heard, {ssid}. Is that right?")):
        yield d.say("No problem, let's start over.")
        return

    if (yield d.confirm("Apply these changes?")):
        yield d.say(f"Done. Connecting to {ssid}.")
    else:
        yield d.say("Okay, nothing changed.")


agent.listen_for("set up wifi", setup_wifi)`,
  },
  {
    id: 'swift',
    label: 'Swift',
    path: 'examples/ios/AgentFlow/AgentFlow/AgentFlowApp.swift',
    lines: [24, 47],
    steps: { askSsid: 3, confirmSsid: 5, startOver: 6, confirmApply: 10, done: 11, unchanged: 13 },
    code: `import MoonshineVoice

func wifiSetup(_ d: Dialog) async throws {
    let ssid = try await d.ask("What's the name of your wifi network?")

    guard try await d.confirm("I heard \\(ssid). Is that right?") else {
        try await d.say("No problem, let's start over.")
        try d.restart()
    }

    if try await d.confirm("Apply these changes?") {
        try await d.say("Done. Connecting to \\(ssid).")
    } else {
        try await d.say("Okay, nothing changed.")
    }
}

agent.listenFor("set up wifi", wifiSetup)`,
  },
  {
    id: 'java',
    label: 'Android',
    path: 'examples/android/AgentFlow/app/src/main/java/ai/moonshine/examples/agentflow/MainActivity.java',
    lines: [41, 62],
    steps: { askSsid: 3, confirmSsid: 4, startOver: 5, confirmApply: 9, done: 10, unchanged: 12 },
    code: `import ai.moonshine.voice.AgentFlow;

private void wifiSetup(AgentFlow.Dialog d) {
    String ssid = d.ask("What's the name of your wifi network?");
    if (!d.confirm("I heard " + ssid + ". Is that right?")) {
        d.say("No problem, let's start over.");
        d.restart();
    }

    if (d.confirm("Apply these changes?")) {
        d.say("Done. Connecting to " + ssid + ".");
    } else {
        d.say("Okay, nothing changed.");
    }
}

agent.listenFor("set up wifi", this::wifiSetup);`,
  },
];
