# Moonshine Voice Python Package

Python bindings for the Moonshine Voice library - a fast, accurate, on-device AI library for building interactive voice applications.

## Installation

```bash
pip install moonshine-voice
```

## Requirements

- Python 3.8 or higher
- The Moonshine C library (libmoonshine.so on Linux, libmoonshine.dylib on macOS, moonshine.dll on Windows)

## Quick Start

```python
from moonshine_voice import Transcriber, ModelArch

# Initialize transcriber with model files
transcriber = Transcriber(
    model_path="/path/to/models",
    model_arch=ModelArch.BASE
)

# Transcribe audio data
audio_data = [...]  # List of float samples (-1.0 to 1.0) at 16kHz
transcript = transcriber.transcribe(audio_data, sample_rate=16000)

# Print results
for line in transcript.lines:
    print(f"[{line.start_time:.2f}s] {line.text}")

# Clean up
transcriber.close()
```

## Streaming Transcription

For real-time transcription from a microphone or live audio source:

```python
from moonshine_voice import Transcriber, ModelArch

transcriber = Transcriber(
    model_path="/path/to/models",
    model_arch=ModelArch.BASE_STREAMING
)

# Create a stream
stream = transcriber.create_stream()
stream.start()

# Add audio chunks as they arrive
while True:
    audio_chunk = get_audio_chunk()  # Your audio capture function
    stream.add_audio(audio_chunk, sample_rate=16000)
    
    # Get updated transcript
    transcript = stream.transcribe()
    for line in transcript.lines:
        if line.is_updated:
            print(f"Updated: {line.text}")

stream.stop()
stream.close()
transcriber.close()
```

## Model Architectures

- `ModelArch.TINY`: Smallest model, fastest inference
- `ModelArch.BASE`: Balanced model with better accuracy
- `ModelArch.TINY_STREAMING`: Tiny model optimized for streaming
- `ModelArch.BASE_STREAMING`: Base model optimized for streaming

## Error Handling

```python
from moonshine_voice import (
    Transcriber,
    MoonshineError,
    MoonshineInvalidHandleError,
)

try:
    transcriber = Transcriber("/path/to/models")
except MoonshineError as e:
    print(f"Error: {e}")
```

## License

MIT License - see the main project repository for details.

## Documentation

For more information, see the [main Moonshine Voice documentation](https://github.com/usefulsensors/moonshine-v2).

