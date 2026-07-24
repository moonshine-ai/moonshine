# Moonshine Voice C++ Example

This is a minimal, platform independent example of using the C++ interface to the Moonshine Voice Library.

To use this you'll first need to download a prebuilt version of the library, or build it yourself using cmake in core/ if you're on a platform without a prebuilt version.

## Download

The easiest way to grab everything you need is the `download-library.sh` helper in this folder. It detects your platform, downloads the matching prebuilt library archive from our [GitHub releases](https://github.com/moonshine-ai/moonshine/releases), and always extracts it into a folder named `moonshine-voice` here (regardless of operating system or architecture, so the build commands below never change). It also fetches the Medium Streaming English model into `medium-streaming-en/` and a sample recording (`two_cities.wav`) so the transcriber example runs out of the box:

```bash
cd examples/c++
./download-library.sh
```

On Linux the extracted `moonshine-voice/lib` folder is self-contained: it holds `libmoonshine.so` alongside the `libonnxruntime.so.1` it depends on, and the library is built with an `$ORIGIN` rpath so it finds the ONNX Runtime next to itself with no `LD_LIBRARY_PATH` needed.

If you would rather download by hand, look for `moonshine-voice-<platform>.tar.gz` on the releases page and extract it into a `moonshine-voice` folder in this directory. For example, on MacOS:

```bash
cd examples/c++
curl -O -L https://github.com/moonshine-ai/moonshine/releases/download/v0.0.73/moonshine-voice-macos-arm64.tar.gz
mkdir -p moonshine-voice
tar xzf moonshine-voice-macos-arm64.tar.gz -C moonshine-voice --strip-components=1
```

## Build

The archive contains the Moonshine library inside the `lib` folder — a shared `libmoonshine.so` on Linux, a static `libmoonshine.a` on MacOS, or `moonshine.lib` on Windows. This is what you'll link against. There are also two headers, one for the low-level C API, and another for the higher-level C++ framework that's built on top of it. On Linux the `lib` folder also contains `libonnxruntime.so.1`, the ONNX Runtime shared library that `libmoonshine.so` loads at runtime.

Since this is a generic C++ example, I'll show the simplest possible build command lines on some common platforms. Because `download-library.sh` always extracts into a folder named `moonshine-voice`, the exact same command works on both x86_64 and 64-bit ARM within each operating system.

### Linux

```bash
g++ transcriber.cpp \
  -Imoonshine-voice/include \
  -Lmoonshine-voice/lib \
  -lmoonshine \
  -Wl,-rpath,'$ORIGIN/moonshine-voice/lib' \
  -o transcriber
```

The `-Wl,-rpath,'$ORIGIN/...'` flag records a runtime library search path relative to the compiled `transcriber` binary, so it finds `libmoonshine.so` (and, through that library's own `$ORIGIN` rpath, `libonnxruntime.so.1`) without any `LD_LIBRARY_PATH`. If you prefer, drop the rpath flag and instead `export LD_LIBRARY_PATH=$(pwd)/moonshine-voice/lib` before running.

You can build `text-to-speech.cpp` the same way (swap the source and `-o` output names). Note that it additionally needs the TTS voice and G2P data — pass its location with `--asset-root` (in a checkout of this repo that is `../../core/moonshine-tts/data`).

### MacOS

```bash
g++ transcriber.cpp \
  -Imoonshine-voice/include \
  -Lmoonshine-voice/lib \
  -lmoonshine \
  -o transcriber \
  -framework CoreFoundation \
  -framework Foundation
g++ text-to-speech.cpp \
  -Imoonshine-voice/include \
  -Lmoonshine-voice/lib \
  -lmoonshine \
  -o text-to-speech \
  -framework CoreFoundation \
  -framework Foundation
```

## Run

You should now have an executable called `transcriber` in this folder. Run it with:

```bash
./transcriber
```

By default it transcribes the `two_cities.wav` sample using the Medium Streaming English model that `download-library.sh` placed in the `medium-streaming-en/` folder, so you should see transcription results printed as it goes. You can point it at different models and inputs using `--model-path`, `--model-arch`, and `--wav-path`.