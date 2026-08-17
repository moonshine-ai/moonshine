# Examples

Try the live demos in your browser at [moonshine.ai](https://moonshine.ai/).

The [`examples`](https://github.com/moonshine-ai/moonshine/tree/main/examples) folder has code samples organized by platform. We use the usual tooling per stack (Android Studio and Gradle, Xcode and Swift on Apple platforms, Visual Studio on Windows). [GitHub Releases](https://github.com/moonshine-ai/moonshine/releases/latest) currently ship the downloadable assets below (example trees are mostly named **`{platform}-{Project}.tar.gz`**; Windows and C++ also include prebuilt native library bundles).

To add the library to your own app rather than starting from a sample, see [Adding the Library](using/adding-the-library.md).

=== "Javascript"

    Self-contained archives: extract, run `node serve.mjs`, then open the demo path in a browser.

    - **[stt](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-stt.tar.gz)** — Real-time speech recognition in the browser via WebAssembly; nothing is uploaded.
    - **[tts](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-tts.tar.gz)** — On-device speech synthesis and zero-shot voice cloning in the browser.
    - **[agent-flow](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-agent-flow.tar.gz)** — A multi-step spoken conversation (wifi setup) triggered by meaning rather than exact wording.
    - **[dictation](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-dictation.tar.gz)** — Dictate a document in the browser with on-device recognition.
    - **[meeting-notes](https://github.com/moonshine-ai/moonshine/releases/latest/download/web-meeting-notes.tar.gz)** — Transcribe a meeting and your microphone separately, entirely on-device.

=== "Python"

    Scripts live under [`examples/python/`](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/). Install with `pip install moonshine-voice` first.

    - **[basic_transcription.py](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/basic_transcription.py)** — Transcribe a WAV file offline or with streaming events.
    - **[mic_transcription.py](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/mic_transcription.py)** — Live microphone transcription with in-place partial updates in the terminal.
    - **[text_to_speech.py](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/text_to_speech.py)** — Speak text aloud, optionally in a cloned voice.
    - **[agent_flow.py](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/agent_flow.py)** — Generator-based conversational agent that walks through wifi setup by voice or keyboard.
    - **[ollama_voice.py](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/ollama-voice/ollama_voice.py)** — Pipe finalized mic transcripts into a local [Ollama](https://ollama.com/) LLM and stream the reply.
    - **[lora-training](https://github.com/moonshine-ai/moonshine/blob/main/examples/python/lora-training/)** — Train a LoRA domain adapter (`pip install 'moonshine-voice[lora]'`). Recipe in [Domain Customization](models/domain-customization.md#retraining).

=== "iOS"

    Open the extracted Xcode project after downloading. Samples pull **`MoonshineVoice`** from the Swift package.

    - **[Transcriber](https://github.com/moonshine-ai/moonshine/releases/latest/download/ios-Transcriber.tar.gz)** — Live microphone transcription UI with partial and final results.
    - **[TextToSpeech](https://github.com/moonshine-ai/moonshine/releases/latest/download/ios-TextToSpeech.tar.gz)** — Speak text with on-device TTS and optional voice cloning.
    - **[AgentFlow](https://github.com/moonshine-ai/moonshine/releases/latest/download/ios-AgentFlow.tar.gz)** — Spoken multi-step wifi-setup conversation using `AgentFlow`.

=== "Android"

    Open the extracted folder in Android Studio. Samples depend on **`ai.moonshine:moonshine-voice:0.1.3`** from Maven Central.

    - **[Transcriber](https://github.com/moonshine-ai/moonshine/releases/latest/download/android-Transcriber.tar.gz)** — Live microphone transcription UI with partial and final results.
    - **[TextToSpeech](https://github.com/moonshine-ai/moonshine/releases/latest/download/android-TextToSpeech.tar.gz)** — Speak text with on-device TTS and optional voice cloning.
    - **[AgentFlow](https://github.com/moonshine-ai/moonshine/releases/latest/download/android-AgentFlow.tar.gz)** — Spoken multi-step wifi-setup conversation using `AgentFlow`.

=== "Linux"

    Portable C++ samples under [`examples/c++/`](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/README.md). The release archive includes sources and a `download-library.sh` helper for the prebuilt library, English model, and sample audio.

    - **[cpp-examples.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/cpp-examples.tar.gz)** — Full example tree plus the download helper for library and models.
    - **[transcriber.cpp](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/transcriber.cpp)** — Minimal CLI that streams a WAV through the C++ API and prints the transcript.
    - **[text-to-speech.cpp](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/text-to-speech.cpp)** — Minimal CLI that synthesizes speech from text and writes a WAV.

=== "MacOS"

    Open the extracted Xcode project (or build with SwiftPM where noted). Samples pull **`MoonshineVoice`** from the Swift package.

    - **[BasicTranscription](https://github.com/moonshine-ai/moonshine/releases/latest/download/macos-BasicTranscription.tar.gz)** — CLI that transcribes a WAV file, offline or with streaming events.
    - **[MicTranscription](https://github.com/moonshine-ai/moonshine/releases/latest/download/macos-MicTranscription.tar.gz)** — Live microphone transcription from the command line.
    - **[TextToSpeech](https://github.com/moonshine-ai/moonshine/releases/latest/download/macos-TextToSpeech.tar.gz)** — Speak text with on-device TTS and optional voice cloning.
    - **[AgentFlow](https://github.com/moonshine-ai/moonshine/releases/latest/download/macos-AgentFlow.tar.gz)** — Spoken wifi-setup dialog flow (`AgentFlow`), with optional `--text` mode.

=== "Windows"

    Self-contained Visual Studio project with the library, model, and sample WAV bundled.

    - **[cli-transcriber](https://github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz)** — C++ CLI that transcribes a WAV or the live microphone; open in Visual Studio and build with F7.

=== "Raspberry Pi"

    USB microphone recommended. Python samples use the same `moonshine-voice` pip package optimized for the Pi.

    - **[my-dalek](https://github.com/moonshine-ai/moonshine/releases/latest/download/raspberry-pi-my-dalek.tar.gz)** — Playful voice-command interface for driving a robot-style script from the Pi.
    - **[Pi Help Bot](https://github.com/moonshine-ai/pi-help-bot/archive/refs/heads/main.zip)** — A more advanced Raspberry Pi voice assistant ([source](https://github.com/moonshine-ai/pi-help-bot)).

None of the samples bundle model weights. Every engine downloads what it needs on first use — the speech model for `Transcriber`, the voice and G2P assets for `TextToSpeech`, all three plus the embedding model for `AgentFlow` — from `https://download.moonshine.ai/`, reporting progress through the `onProgress` callback the examples wire up to a label. Downloads are cached (under `filesDir` on Android, `Caches/MoonshineModels` on Apple platforms), so later launches run offline. Switching to a different voice triggers the same on-demand download for whatever that voice needs.

If you want a fully offline build with no first-run download, fetch the assets ahead of time and point the engine at them with `modelsFrom(path)`; see [`docs/design/api-comparison.md`](https://github.com/moonshine-ai/moonshine/blob/main/docs/design/api-comparison.md) for the tradeoff.
