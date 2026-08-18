# Quickstart

=== "Javascript"

    In node run `npm install @moonshine-ai/moonshine-wasm`, or for the web import directly from the CDN.

    <!-- doc-test: parse-only -->
    ```js
    import { MicTranscriber, ModelArch } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';
     
    const mic = new MicTranscriber()
      .modelArch(ModelArch.MediumStreaming)
      .onText((text) => showInProgress(text))
      .onLine((line) => appendLine(line.text, line.lastTranscriptionLatencyMs));
     
    await mic.load();
    await mic.start();
    ```

    You can see live examples running at [moonshine.ai](https://moonshine.ai), or download the [speech to text](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-stt.tar.gz), [text to speech](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-tts.tar.gz), [voice agent](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-agent-flow.tar.gz), [dictation](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-dictation.tar.gz), or [meeting note taker](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-meeting-notes.tar.gz) projects. To serve them run `node serve.mjs` and navigate to [http://localhost:8080/](http://localhost:8080/).

=== "Python"

    <!-- doc-test: parse-only -->
    ```bash
    pip install moonshine-voice
    moonshine-voice mic --language en
    ```

    Listens to the microphone and prints updates to the transcript as they come in.

    <!-- doc-test: parse-only -->
    ```bash
    moonshine-voice agent
    ```

    Runs a spoken wifi-setup conversation: it listens for a trigger phrase, asks questions, and confirms the answers. Matching is semantic, so natural language variations are recognized. For more, check out [our "Getting Started" Colab notebook](https://bit.ly/moonshine-colab) and [video](https://www.youtube.com/watch?v=WH-AGvHmtoM).

    <!-- doc-test: parse-only -->
    ```bash
    moonshine-voice tts --language en_us --text "Hello world"
    ```

    Synthesizes and speaks the text.

=== "iOS"

    First [add `https://github.com/moonshine-ai/moonshine-swift/` as a package dependency to your project in Xcode](using/adding-the-library.md).

    ```swift
    import MoonshineVoice
     
    let mic = MicTranscriber()
        .onText { [weak self] text in
            Task { @MainActor in self?.liveText = text }
        }
        .onLine { [weak self] line in
            Task { @MainActor in self?.lines.append(line.text) }
        }
     
    try await mic.load()
    try mic.start()
    ```

    Download [github.com/moonshine-ai/moonshine/releases/latest/download/ios-Transcriber.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/ios-Transcriber.tar.gz), extract it, and then open the `Transcriber/Transcriber.xcodeproj` project in Xcode.

=== "Android"

    Add `ai.moonshine:moonshine-voice:0.1.5` to your project's `build.gradle.kts` (or equivalent).

    ```java
    import ai.moonshine.voice.MicTranscriber;
     
    mic = new MicTranscriber(this)
            .onText(text -> transcriptText.setText(finishedLines + text))
            .onLine(line -> {
                finishedLines.append(line.text).append('\n');
                transcriptText.setText(finishedLines.toString());
            });
     
    worker.execute(() -> {
        mic.load();
        mic.start();
    });
    ```

    Download [github.com/moonshine-ai/moonshine/releases/latest/download/android-Transcriber.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/android-Transcriber.tar.gz), extract it, and then open the `Transcriber` folder in Android Studio.

=== "Linux"

    Moonshine Voice ships prebuilt shared libraries for both x86_64 and arm64 Linux. The quickest way to try it is with the [portable C++ example](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/README.md), which downloads the library, an English speech to text model, and a sample recording, then builds and runs a transcriber:

    <!-- doc-test: skip -->
    ```bash
    curl -O -L https://github.com/moonshine-ai/moonshine/releases/latest/download/cpp-examples.tar.gz
    tar xzf cpp-examples.tar.gz
    cd c++
    ./download-library.sh
    g++ transcriber.cpp -Imoonshine-voice/include -Lmoonshine-voice/lib -lmoonshine -Wl,-rpath,'$ORIGIN/moonshine-voice/lib' -o transcriber
    ./transcriber
    ```

=== "MacOS"

    First [add `https://github.com/moonshine-ai/moonshine-swift/` as a package dependency to your project in Xcode](using/adding-the-library.md).

    ```swift
    import MoonshineVoice
     
    let mic = MicTranscriber()
        .onText { [weak self] text in
            Task { @MainActor in self?.liveText = text }
        }
        .onLine { [weak self] line in
            Task { @MainActor in self?.lines.append(line.text) }
        }
     
    try await mic.load()
    try mic.start()
    ```

    This code is identical to the iOS version.

    Download [github.com/moonshine-ai/moonshine/releases/latest/download/macos-MicTranscription.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/macos-MicTranscription.tar.gz), extract it, and then open the `MicTranscription/MicTranscription.xcodeproj` project in Xcode.

=== "Windows"

    Download [github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz), extract it, and then open the `cli-transcriber\cli-transcriber.vcxproj` project in Visual Studio.

    It's a self-contained archive that includes the library and model, so Ctrl+Shift+B or F7 will build the executable.

=== "Raspberry Pi"

    You'll need a USB microphone plugged in to get audio input, but the Python pip package has been optimized for the Pi, so you can run:

    <!-- doc-test: skip -->
    ```bash
     sudo pip install --break-system-packages moonshine-voice
     moonshine-voice mic --language en
    ```

    I've recorded [a screencast on YouTube](https://www.youtube.com/watch?v=NNcqx1wFxl0) to help you get started, and you can also download [github.com/moonshine-ai/moonshine/releases/latest/download/raspberry-pi-my-dalek.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/raspberry-pi-my-dalek.tar.gz) for some fun, Pi-specific examples. [The README](https://github.com/moonshine-ai/moonshine/blob/main/examples/raspberry-pi/my-dalek/README.md) has information about using a virtual environment for the Python install if you don't want to use `--break-system-packages`.

    You can look at [github.com/moonshine-ai/pi-help-bot](https://github.com/moonshine-ai/pi-help-bot) for a more advanced example.

## Coding agents

If you are using Cursor, Claude Code, Codex, or another Agent Skills client, copy [`.agents/skills/moonshine-voice/`](https://github.com/moonshine-ai/moonshine/tree/main/.agents/skills/moonshine-voice) into your project's `.agents/skills/` folder, or run `npx skills add moonshine-ai/moonshine --skill moonshine-voice`. That skill is the happy path for integrating this library; it is not a substitute for the docs above.
