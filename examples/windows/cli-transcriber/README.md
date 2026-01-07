# CLI Transcriber

A command-line application for Windows that listens to the microphone and transcribes speech in real-time using the Moonshine C++ API.

## Prerequisites

1. **Build the Moonshine library and all dependencies**: Before building this project, you need to build the core Moonshine library and all its dependencies (bin-tokenizer, ort-utils, ten_vad, moonshine-utils):
   ```batch
   cd ..\..\..\core
   if not exist build mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```
   
   This will build all the required static libraries:
   - `moonshine.lib` (in `core/build/Release/`)
   - `bin-tokenizer.lib` (in `core/bin-tokenizer/build/Release/`)
   - `ort-utils.lib` (in `core/ort-utils/build/Release/`)
   - `ten_vad.lib` (in `core/third-party/ten_vad/build/Release/`)
   - `moonshine-utils.lib` (in `core/moonshine-utils/build/Release/`)
   
   The ONNX Runtime library (`onnxruntime.lib` and `onnxruntime.dll`) should already be present in `core/third-party/onnxruntime/lib/windows/x86_64/`.

2. **Visual Studio 2022** (or later) with C++ development tools installed

3. **Windows SDK** (included with Visual Studio)

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
x64\Release\cli-transcriber.exe [options]
```

### Options

- `-m, --model-path PATH`: Path to the model directory (default: `../../../test-assets/tiny-en`)
- `-a, --model-arch ARCH`: Model architecture: 0=TINY, 1=BASE, 2=TINY_STREAMING, 3=BASE_STREAMING (default: 0)
- `-h, --help`: Show help message

### Example

```batch
cli-transcriber.exe -m ..\..\..\test-assets\tiny-en -a 0
```

The application will:
1. Load the Moonshine transcriber with the specified model
2. Start listening to the default microphone
3. Display transcriptions in real-time as you speak
4. Press Ctrl+C to stop

## Notes

- The application uses Windows Audio Session API (WASAPI) to capture audio from the default microphone
- Audio is automatically resampled to 16kHz mono if needed
- The transcriber uses streaming mode for real-time transcription
- Make sure you have microphone permissions enabled in Windows settings

