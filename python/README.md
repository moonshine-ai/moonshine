# Moonshine Voice Python Package

A fast, accurate, on-device AI library for building interactive voice applications. [Join our Discord to get help and support](https://discord.gg/27qp9zSRXF).

## Installation

```bash
pip install moonshine-voice
```

## Quick Start

<!-- doc-test: parse-only -->
```bash
# Listens to the microphone, logging to the console when there are 
# speech updates.
moonshine-voice mic
```

Installing the package adds a `moonshine-voice` command (with a shorter
`moonshine` alias) that groups the built-in tools as subcommands: `mic`,
`transcribe`, `tts`, `dialog`, `download`, and `g2p`. Run
`moonshine-voice --help`, or `moonshine-voice <command> --help` for a specific
tool. Each subcommand is equivalent to `python -m moonshine_voice.<module>`, so
either invocation style works.

## Example

```python
"""Transcribes live audio from the default microphone"""
import time
from moonshine_voice import MicTranscriber

# MicTranscriber handles connecting to the microphone, capturing the audio
# data, detecting voice activity, breaking the speech up into segments,
# transcribing it, and calling you back as the results firm up over time.
mic = (
    MicTranscriber()
    .on_text(lambda text: print(f"\r{text}", end="", flush=True))
    .on_line(lambda line: print(f"\r{line.text}"))
)

# Downloads the model files and caches them, so the first call is the slow one.
mic.load()
mic.start()
print("Listening to the microphone, press Ctrl+C to stop...")

while True:
    time.sleep(0.1)
```

`on_text` gives you the line as it is being spoken, rewritten as the model
changes its mind, and `on_line` gives you each line once it is final.
Configuration is chainable and everything has a working default:
`language()`, `model_arch()`, `device()`, `update_interval()`, and
`on_progress()` for a download bar.

When you need more than the text, such as line ids, speaker spans, or word
timings, attach a listener object instead. Both styles can be used together.

```python
from moonshine_voice import MicTranscriber, TranscriptEventListener

class TestListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")

mic = MicTranscriber()
mic.add_listener(TestListener())
mic.load()
mic.start()
```

## Other Sources

If you have a different source you're capturing audio from you can supply it directly to a transcriber.

```python
"""Transcribes live audio from an arbitrary audio source."""
from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
    load_wav_file,
    get_assets_path,
)
import os
from typing import Iterator, Tuple


def audio_chunk_generator(
    wav_file_path: str, chunk_duration: float = 0.1
) -> Iterator[Tuple[list, int]]:
    """
    Example function that loads a WAV file and yields audio chunks.

    This demonstrates how you can integrate your own proprietary
    audio data capture sources. Replace this function with your own
    implementation that yields (audio_chunk, sample_rate) tuples.

    Args:
        wav_file_path: Path to the WAV file to load
        chunk_duration: Duration of each chunk in seconds

    Yields:
        Tuple of (audio_chunk, sample_rate) where:
        - audio_chunk: List of float audio samples
        - sample_rate: Sample rate in Hz
    """
    audio_data, sample_rate = load_wav_file(wav_file_path)
    chunk_size = int(chunk_duration * sample_rate)

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i: i + chunk_size]
        yield (chunk, sample_rate)


model_path, model_arch = get_model_for_language("en")

transcriber = Transcriber(
    model_path=model_path, model_arch=model_arch)

stream = transcriber.create_stream(update_interval=0.5)
stream.start()


class TestListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"{event.line.start_time:.2f}s: Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(
            f"{event.line.start_time:.2f}s: Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"{event.line.start_time:.2f}s: Line completed: {event.line.text}")


listener = TestListener()
stream.add_listener(listener)

# Feed audio chunks from the generator into the stream.
wav_file_path = os.path.join(get_assets_path(), "two_cities.wav")
for chunk, sample_rate in audio_chunk_generator(wav_file_path):
    stream.add_audio(chunk, sample_rate)

stream.stop()
stream.close()
```

## Voice Commands

Voice commands go through `DialogFlow`. Register the phrases you want to listen
for and it handles the rest: it downloads the speech recognition, speech
synthesis and phrase-matching models, opens the microphone, matches what the
user said semantically rather than by exact wording, and runs your handler.

```python
from moonshine_voice import DialogFlow

def lights_on(d):
    print("\n💡 LIGHTS ON!")

def lights_off(d):
    print("\n🌑 LIGHTS OFF!")

runner = (
    DialogFlow()
    .always("turn on the lights", lights_on)
    .always("turn off the lights", lights_off)
)
```

`load()` downloads and opens everything it needs, and `start_listening()` opens
the microphone. There's nothing else to construct:

```python
runner.load()
runner.start_listening()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nStopping...", file=sys.stderr)
finally:
    runner.close()
```

Configuration is chainable, and everything has a working default — `language()`,
`voice()`, `trigger_threshold()`, `on_heard()` / `on_said()` / `on_error()` for
observing the conversation, and `use_mic_transcriber()` / `use_text_to_speech()`
when you'd rather supply your own. To drive a runner from text instead of audio,
turn the microphone off with `microphone(False)` and feed it `handle_utterance()`.

For multi-turn conversations — asking a question, confirming an answer,
spelling out a password — register a flow with `listen_for()` instead of a
global. See `examples/python/dialog_flow.py`, or run `moonshine-voice dialog`.

## Speaking

`TextToSpeech` follows the same shape: configure it, `load()` it, then use it.

```python
from moonshine_voice import TextToSpeech

tts = TextToSpeech().language("en_us").voice("kokoro_af_heart")
tts.load()
tts.say("Hello from Moonshine.")
tts.wait()
```

`say()` returns immediately and queues the utterance, pre-synthesizing the next
one while the current one plays, so consecutive calls run together without a
gap. `wait()` blocks until the queue drains, `stop()` cancels it, and
`is_talking()` polls. If you would rather have the samples than hear them, use
`synthesize()`. The chainable setters are `language()`, `voice()`,
`output_device()`, `volume()`, `models_from()`, and `on_progress()` for a
download bar.

To speak in someone else's voice, `clone_from()` takes a few seconds of them
talking, either as a `.wav` file or as a `(pcm, sample_rate)` pair.

```python
tts = TextToSpeech().language("en_us").cloning()
tts.load()
tts.clone_from("some-speech.wav")
tts.say("Now I sound like you.")
```

`cloning()` fetches the cloning engine during `load()` so the first clone is
quick. To capture the reference clip from the microphone instead, `start_cloning()`
hands back a `VoiceClone` that listens until it has heard enough usable speech.

```python
clone = tts.start_cloning()
clone.on_ready(lambda: print("Got it, you can stop talking."))
clone.from_microphone()
tts.clone_from(clone)
```

## Multiple Languages

The framework currently supports English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Arabic, and Ukrainian. We are working on wider language support, and you can see which are supported in your version by calling `supported_languages()`. To use a language, request it using `get_model_for_language()` passing in the two-letter language code. For example `get_model_for_language("es")` will download the Spanish models and pass the information you need to create `Transcriber` objects using them.

## Documentation

For more information, see the [main Moonshine Voice documentation](https://github.com/moonshine-ai/moonshine).

## License

The code and English-language models are released under the MIT License - see the main project repository for details. The models used for other languages are released under the [Moonshine Community License](https://www.moonshine.ai/license).
