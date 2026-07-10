# CLI Transcriber

A command-line application for Windows that transcribes speech using the Moonshine C++ API. You'll need **Visual Studio 2022** (or later) with C++ development tools installed.

The [GitHub release archive](https://github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz) is self-contained: it includes the Moonshine library, headers, the English medium streaming model, and a sample WAV file for testing.

## Setup

Download and extract [windows-cli-transcriber.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/windows-cli-transcriber.tar.gz).

If you're working from a git checkout instead, run `download-lib.bat` to fetch the library bundle separately, then download models with `pip install moonshine-voice` and `moonshine-voice download --stt --language en`.

## Building

1. Open `cli-transcriber.sln` in Visual Studio
2. Select **Release** and **x64** (recommended), or **Debug** and **x64** if you need to step through your own code
3. Build the solution (Build > Build Solution or press F7)

The prebuilt Moonshine libraries are Release-only (`/MD`). The Debug configuration uses the same runtime library so it can link against them; you still get unoptimized code and debug symbols for the example app itself.

Alternatively, you can build from the command line:

```batch
msbuild cli-transcriber.sln /p:Configuration=Release /p:Platform=x64
```

## Running

### Transcribe the bundled sample file

```batch
x64\Release\cli-transcriber.exe --wav-path beckett.wav
```

This uses the bundled `models/medium-streaming-en` model (architecture 5) and should print a transcript of the Beckett quote.

### Live microphone transcription

```batch
x64\Release\cli-transcriber.exe
```

This loads the bundled model, listens on the default microphone, and displays transcriptions in real time. Press Ctrl+C to stop.

### Custom model path

```batch
x64\Release\cli-transcriber.exe --model-path "path\to\model" --model-arch 5
```

## Notes

- The application uses Windows Audio Session API (WASAPI) to capture audio from the default microphone
- Audio is automatically resampled to 16kHz mono if needed
- Make sure you have microphone permissions enabled in Windows settings when using live capture

## Adding Moonshine to your own Project

To use Moonshine Voice in an application:

 - Make sure `moonshine-voice-windows-x86_64` is downloaded and accessible.
 - Add `moonshine-voice-windows-x86_64\include` to the include paths.
 - Add `moonshine-voice-windows-x86_64\lib` to the linker paths.
 - Add all of the libraries in `moonshine-voice-windows-x86_64\lib` (bin-tokenizer.lib, moonshine-utils.lib, moonshine.lib, onnxruntime.lib, and ort-utils.lib) to be linked.
 - Ensure that `onnxruntime.dll` from `moonshine-voice-windows-x86_64\lib` is copied to the same folder as your executable. This example project does that using a custom build step.
