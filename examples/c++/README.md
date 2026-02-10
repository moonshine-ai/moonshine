# Moonshine Voice C++ Example

This is a minimal, platform independent example of using the C++ interface to the Moonshine Voice Library.

To use this you'll first need to download a prebuilt version of the library, or build it yourself using cmake in core/ if you're on a platform without a prebuilt version.

## Download

The library files are available as part of our [releases on GitHub](https://github.com/moonshine-ai/moonshine/releases). Look for the most recent version, and you should see a file called moonshine-voice-_.tgz, where _ is your platform. Download and extract that archive, placing the resulting folder in this `examples/c++` directory.

## Build

The archive contains a .a or .lib archive (depending on your platform) inside the `lib` folder. This is the static library you'll need to link against. There are also two headers, one for the low-level C API, and another for the higher-level C++ framework that's built on top of it.

Since this is a generic C++ example, I'll show the simplest possible build command lines on some common platforms. You should replace `moonshine-voice-*` with the name of the library you downloaded.

### Linux

```bash
g++ transcriber.cpp \
  -Imoonshine-voice-linux-x86_64/include \
  -Lmoonshine-voice-linux-x86_64/lib \
  -lmoonshine \
  -o transcriber
export LD_LIBRARY_PATH=`pwd`/moonshine-voice-linux-x86_64/lib
```

### MacOS

```bash
g++ transcriber.cpp \
  -Imoonshine-voice-macos/include \
  -Lmoonshine-voice-macos/lib \
  -lmoonshine \
  -o transcriber \
  -framework CoreFoundation \
  -framework Foundation
```

## Run

You should now have an executable called `transcriber` in this folder. To test it, run `./transcriber` and you should see some transcription results. You can try different models and inputs using `--model-path`, `--model-arch`, and `--wav-path`.