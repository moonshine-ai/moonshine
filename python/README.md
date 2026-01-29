# Moonshine Voice Python Package

A fast, accurate, on-device AI library for building interactive voice applications. [Join our Discord to get help and support](https://discord.gg/27qp9zSRXF).

## Installation

```bash
pip install moonshine-voice
```

## Quick Start

```
# Listens to the microphone, logging to the console when there are 
# speech updates.
python -m moonshine_voice.mic_transcriber
```

## Example

```python
"""Transcribes live audio from the default microphone"""
import time
from moonshine_voice import (
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
)

# This will download the model files and cache them.
model_path, model_arch = get_model_for_language("en")

# MicTranscriber handles connecting to the microphone, capturing
# the audio data, detecting voice activity, breaking the speech
# up into segments, transcribing the speech, and sending events
# as the results are updated over time.
mic_transcriber = MicTranscriber(
    model_path=model_path, model_arch=model_arch)

# We use an event-driven interface to respond in real time
# as speech is detected.
class TestListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")

listener = TestListener()
mic_transcriber.add_listener(listener)
mic_transcriber.start()
print("Listening to the microphone, press Ctrl+C to stop...")

while True:
    time.sleep(0.1)
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

We also provide voice command recognition using the `IntentRecognizer` module. It captures transcribed audio from a `MicTranscriber` and invokes callback functions that match your programmed intents. Since it relies on an embedding model, you can use a helper function to get started:

```python
from moonshine_voice import (
    MicTranscriber,
    IntentRecognizer,
    ModelArch,
    EmbeddingModelArch,
    get_embedding_model,
    get_model_for_language
)

# Download and load the embedding model for intent recognition
embedding_model_path, embedding_model_arch = get_embedding_model()
```

Next, create a recognizer and register your intent callbacks:

```python
intent_recognizer = IntentRecognizer(
    model_path=embedding_model_path,
    model_arch=embedding_model_arch
)

def on_lights_on(trigger: str, utterance: str, similarity: float):
    """Handler for turning lights on."""
    print(f"\n💡 LIGHTS ON! (matched '{trigger}' with {similarity:.0%} confidence)")

def on_lights_off(trigger: str, utterance: str, similarity: float):
    """Handler for turning lights off."""
    print(f"\n🌑 LIGHTS OFF! (matched '{trigger}' with {similarity:.0%} confidence)")

intent_recognizer.register_intent("turn on the lights", on_lights_on)
intent_recognizer.register_intent("turn off the lights", on_lights_off)
```

Finally, create a `MicTranscriber`, connect it to your `IntentRecognizer`, and start the audio stream:

```python
# Get the transcription model and initialize a MicTranscriber
model_path, model_arch = get_model_for_language("en")
mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

# The intent recognizer will process completed transcript lines and invoke trigger handlers
mic_transcriber.add_listener(intent_recognizer)

mic_transcriber.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nStopping...", file=sys.stderr)
finally:
    intent_recognizer.close()
    mic_transcriber.stop()
    mic_transcriber.close()
```

## Multiple Languages

The framework currently supports English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Arabic, and Ukrainian. We are working on wider language support, and you can see which are supported in your version by calling `supported_languages()`. To use a language, request it using `get_model_for_language()` passing in the two-letter language code. For example `get_model_for_language("es")` will download the Spanish models and pass the information you need to create `Transcriber` objects using them.

## Documentation

For more information, see the [main Moonshine Voice documentation](https://github.com/moonshine-ai/moonshine).

## License

The code and English-language models are released under the MIT License - see the main project repository for details. The models used for other languages are released under the [Moonshine Community License](https://www.moonshine.ai/license).
