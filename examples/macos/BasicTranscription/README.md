# Basic Transcription Example

A Swift command-line application that demonstrates how to use Moonshine Voice for transcription, mirroring the functionality of the Python `basic_transcription.py` script.

## Features

- Transcribe audio files offline without streaming
- Transcribe audio files with streaming (event-based)
- Support for multiple languages and model architectures
- Command-line argument parsing

## Building

```bash
cd examples/macos/BasicTranscription
swift build
```

## Running

```bash
# Use default test file (test-assets/two_cities.wav)
swift run BasicTranscription

# Specify input files
swift run BasicTranscription path/to/audio1.wav path/to/audio2.wav

# Specify language
swift run BasicTranscription --language en path/to/audio.wav

# Specify model architecture
swift run BasicTranscription --model-arch tiny path/to/audio.wav

# Show help
swift run BasicTranscription --help
```

## Command-Line Options

- `--language, -l LANGUAGE`: Language to use for transcription (default: `en`)
- `--model-arch, -m ARCH`: Model architecture: `tiny`, `base`, `tiny-streaming`, `base-streaming`
- `--help, -h`: Show help message

## Supported Languages

- `en` / `english` - English
- `ja` / `japanese` - Japanese
- `es` / `spanish` - Spanish
- `ar` / `arabic` - Arabic
- `ko` / `korean` - Korean
- `vi` / `vietnamese` - Vietnamese
- `uk` / `ukrainian` - Ukrainian
- `zh` / `chinese` - Chinese

## Example Output

The application will:
1. Transcribe each input file offline (non-streaming) and print the results
2. Transcribe each input file with streaming, showing events as they occur:
   - Line started events
   - Line text changed events
   - Line completed events

## Notes

- The application expects model files to be in `test-assets/{model-name}/` directory
- WAV files must be PCM format (16-bit, 24-bit, or 32-bit)
- Multi-channel audio is automatically mixed down to mono

