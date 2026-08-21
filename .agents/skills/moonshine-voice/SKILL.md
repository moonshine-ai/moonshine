---
name: moonshine-voice
description: >-
  Integrate on-device Moonshine Voice (speech-to-text, text-to-speech, voice
  cloning, AgentFlow conversational agents). Use when the user asks to add
  Moonshine, MicTranscriber, AgentFlow, on-device STT/TTS, a voice agent,
  dictation, meeting notes, or keyterms/domain customization — not Whisper,
  cloud STT, or OpenAI Realtime.
---

# Moonshine Voice

On-device voice toolkit. No API keys, no cloud. Do not substitute Whisper, OpenAI Realtime, or a hosted STT/TTS API when the user asked for Moonshine.

Never trust model memory for this SDK. Verify against live docs before writing code:

- Docs: https://moonshine-voice.readthedocs.io
- Transcription: https://moonshine-voice.readthedocs.io/en/latest/using/transcription/
- Conversational agent: https://moonshine-voice.readthedocs.io/en/latest/using/conversational-agent/
- Text to speech: https://moonshine-voice.readthedocs.io/en/latest/using/text-to-speech/
- Domain customization: https://moonshine-voice.readthedocs.io/en/latest/models/domain-customization/
- Debugging: https://moonshine-voice.readthedocs.io/en/latest/using/debugging/
- Adding the library: https://moonshine-voice.readthedocs.io/en/latest/using/adding-the-library/

Python snippets below are the lingua franca. Other bindings use the same shape with native names (`on_text` → `onText`, `listen_for` → `listenFor`, `start_listening` → `startListening`).

## Install

| Platform | Package |
| --- | --- |
| Python | `pip install moonshine-voice` then `import moonshine_voice` |
| JavaScript | `npm install @moonshine-ai/moonshine-wasm` (or the jsDelivr CDN) |
| Swift (iOS/macOS) | SPM: `https://github.com/moonshine-ai/moonshine-swift/` then `import MoonshineVoice` |
| Android | Maven `ai.moonshine:moonshine-voice` (pass an Android `Context` to constructors) |
| Linux/Windows C++ | Prebuilt libs from GitHub Releases; `#include "moonshine-cpp.h"` |

## Canonical shape

Construct → chainable setters → `load()` → `start()` / `start_listening()`.

Constructors are cheap and cannot fail. Nothing is downloaded or opened until `load()`. `load()` is the slow, fallible call (first use may download models into a local cache; later launches reuse it and run offline). Call setters before `load()`.

Pick the high-level type:

| Need | Type |
| --- | --- |
| Live microphone speech-to-text | `MicTranscriber` |
| Feed PCM/WAV yourself | `Transcriber` |
| Spoken conversational flows | `AgentFlow` |
| Playback or voice cloning | `TextToSpeech` |

`AgentFlow` loads STT, embeddings, TTS, and a mic internally. Do not assemble those objects yourself unless the user asked for that.

## Transcription

`on_text` / `onText` is the in-progress hypothesis (it will change). `on_line` / `onLine` is the finished segment. Do not treat partial text as final.

```python
from moonshine_voice import MicTranscriber

mic = (
    MicTranscriber()
    .language("en")
    .on_text(lambda text: print(text, end="\r", flush=True))
    .on_line(lambda line: print(line.text))
)
mic.load()
mic.start()
```

Use `Transcriber` only when feeding audio yourself. For line ids, speaker spans, or word timings, `add_listener()` still exists; the named callbacks cover the common path.

## Conversational agent

Python flow bodies use `yield` so the runner can wait for speech. JavaScript/Swift/Java flow bodies are ordinary `async` / blocking functions (`await d.ask(...)`), not generators.

```python
from moonshine_voice import AgentFlow, Dialog

def report_ip(d: Dialog):
    yield d.say("Your address is 1 9 2 dot 1 6 8 dot 1 dot 1")

agent = AgentFlow().language("en").listen_for("What is my IP address?", report_ip)
agent.start_listening()
```

`start_listening()` downloads on first use. Call `load()` first if you need to schedule that yourself. Matching is semantic. `otherwise()` handles speech that matched no trigger. Fetch the agent doc for `ask` / `confirm` / spelled input.

## Text to speech and cloning

```python
from moonshine_voice import TextToSpeech

tts = TextToSpeech().language("en-us")
tts.load()
tts.say("Hello world")
tts.wait()
```

Call `cloning()` before `load()`, then `clone_from()` (file or PCM) or `start_cloning()` (microphone). Catalog `voice()` and `cloning()` are mutually exclusive.

When the text comes from a language model, stream it instead of waiting for the whole reply:

```python
with tts.say_stream() as speech:
    for token in llm.stream(prompt):
        tts.push_text(token)
    tts.end_input()
    speech.wait()
```

The streaming calls live on the synthesizer itself; there is no stream object to open or close, and a synthesizer speaks one reply at a time, so `say()` and `synthesize()` refuse while one is in flight. `AgentFlow` has the same `say_stream()`. Use `for chunk in tts.stream(pieces)` when you want the `TtsChunk` objects rather than playback. Text is buffered until a sentence completes, because a synthesizer fed one word at a time produces a list, not a sentence — call `flush()` to force what is buffered, `end_input()` when done, and `cancel_stream()` to drop a reply the user talked over. The same names (`pushText`, `flush`, `endInput`, `cancelStream`, `onChunk`) exist in TypeScript, Swift and Java.

## Domain customization

`set_keyterms([...])` biases toward jargon; `set_context(passage)` extracts terms from a document. Streaming architectures only — Tiny/Base raise. Takes effect on the next transcription; does not rewrite text already emitted. Keep the list curated; thousands of terms hurt accuracy. See the domain-customization doc for `keyterm_boost`. Teaching conventions or a new acoustic environment is `pip install 'moonshine-voice[finetune]'` then `python -m moonshine_voice.lora` (or `moonshine-voice finetune`) in a training environment — do not add PyTorch or Transformers to an inference app. ATCOSIM is phraseology, not VHF; real radio is `--dataset uwb_atcc` or `--train-manifest`.

## Anti-patterns

- Do not use `DialogFlow` or the old Intent API. The type is `AgentFlow`.
- Do not load models in the constructor or via a static `MicTranscriber.load(...)`.
- Do not supply `.onnx` models. The library accepts OnnxRuntime flatbuffers (`.ort`) only.
- Do not replace a Moonshine request with Whisper, OpenAI Realtime, or cloud STT/TTS.
- Do not use `keyterms` / `context` on Tiny or Base.
- Do not treat `on_text` as a finished line.
- Do not copy Python `yield` flow bodies into JavaScript/Swift/Java.
- Do not add torch/transformers to an inference install; training is `moonshine-voice[finetune]` (same extra as `[lora]`).

## Debugging

If transcription looks wrong, set `save_input_wav_path` to dump the audio the transcriber actually received. Set `log_api_calls=true` to print the underlying call timeline. See the debugging doc.
