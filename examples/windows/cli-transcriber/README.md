# CLI Transcriber

A command-line application for Windows that listens to the microphone and transcribes speech in real-time using the Moonshine C++ API. You'll need **Visual Studio 2022** (or later) with C++ development tools installed, and Python 3.8+ with the `moonshine-voice` pip package is recommended for downloading models.

## Setup

1. **Download the Moonshine Voice Library**. Download [github.com/moonshine-ai/moonshine/releases/latest/download/moonshine-voice-windows-x86_64.tar.gz](https://github.com/moonshine-ai/moonshine/releases/latest/download/moonshine-voice-windows-x86_64.tar.gz) and extract it into this folder, or you can run `download-lib.bat`.

2. **Download the Models**. Run `pip install moonshine-voice` if you haven't already, and then run `python -m moonshine_voice.download --language en` to download the Moonshine English-language speech to text models. Make a note of the log output from that command, you'll use it to provide a model path when you run the transcriber.

## Building

1. Open `cli-transcriber.sln` in Visual Studio
2. Select the desired configuration (Debug or Release) and platform (x64)
3. Build the solution (Build > Build Solution or press F7)

Alternatively, you can build from the command line:
```batch
msbuild cli-transcriber.sln /p:Configuration=Release /p:Platform=x64
```

## Running

After building, run the executable from the command line:

```batch
x64\Release\cli-transcriber.exe --model-path <path from the download command> --model-arch <number from the download command>
```

If you don't have the path and architecture handy, rerun the `python -m moonshine_voice.download --language en` command. It shouldn't download again, but will print out the information you need.

Here's what the command should look like, using the downloaded location on my machine:

```batch
 .\x64\Release\cli-transcriber.exe --model-path "C:\Users\windo\AppData\Local\moonshine_voice\moonshine_voice\Cache\download.moonshine.ai/model/base-en/quantized/base-en" --model-arch 1
```

This example:

1. Loads the Moonshine transcriber with the specified model
2. Start listening to the default microphone
3. Display transcriptions in real-time as you speak

Press Ctrl+C to stop

## Notes

- The application uses Windows Audio Session API (WASAPI) to capture audio from the default microphone
- Audio is automatically resampled to 16kHz mono if needed
- The transcriber uses streaming mode for real-time transcription
- Make sure you have microphone permissions enabled in Windows settings

## Adding Moonshine to your own Project

To use Moonshine Voice in an application:

 - Make sure the `moonshine-voice-x86_64` is downloaded and accessible.
 - Add `moonshine-voice-x86_64\include` to the include paths.
 - Add `moonshine-voice-x86_64\lib` to the linker paths.
 - Add all of the libraries in `moonshine-voice-x86_64\lib` (bin-tokenizer.lib, moonshine-utils.lib, moonshine.lib, onnxruntime.lib, and ort-utils.lib) to be linked.
 - Ensure that `onnxruntime.dll` from `moonshine-voice-x86_64\lib` is copied to the same folder as your executable. This example project does that using a custom build step.