![Moonshine Voice Logo](images/logo.png)

# Moonshine Voice

## Voice Interfaces for Everyone

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building voice applications. 

 - Everything runs on-device, so it's fast, private, and there are no server API charges. 
 - Our models are trained from scratch, and offer higher accuracy than Whisper Large V3 at the top end, down to 26MB models for constrained deployments.
 - It's cross-platform, running on [Python](#python), [iOS](#ios), [Android](#android), [MacOS](#macos), [Linux](#linux) and [Windows](#windows).
 - Batteries are included. Its high-level APIs offer complete solutions for common tasks like transcription, so you don't need to be an ML expert to use them.
 - It supports multiple languages, including English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, and Arabic.
 - The framework and models are optimized for streaming applications, offering low latency responses by doing most of the work while the user is still talking.

 ## Quickstart

 ### Python

 ```bash
 pip install moonshine-voice
 python -m moonshine_voice.mic_transcriber --language en
 ```

Listens to the microphone and prints updates to the transcript as they come in.

### iOS

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) (or checkout this repository) and open `examples/ios/Transcriber/Transcriber.xcodeproj` in Xcode.

### Android

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) (or checkout this repository) and open `examples/android/Transcriber/` in Android Studio.

### Linux

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) (or checkout this repository) and then run:

```bash
cd core
mkdir build
cmake ..
cmake --build .
./moonshine-cpp-test
```

### MacOS

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) (or checkout this repository) and open `examples/macos/MicTranscription/MicTranscription.xcodeproj` in Xcode.

### Windows

TK

## Using the Library

TK

## Adding the Library to your own App

### iOS

TK

### Android

TK

### Linux

TK

### MacOS

TK

### Windows

TK

## Acknowledgements

We're grateful to:

 - The TEN team for open sourcing [their voice-activity detection model](https://github.com/TEN-framework/ten-vad).
 - The ONNX Runtime community for building [a fast, cross-platform inference engine](https://github.com/microsoft/onnxruntime).
 - [Viktor Kirilov](https://github.com/onqtam) for [his fantastic DocTest C++ testing framework](https://github.com/doctest/doctest).
 - [Nemanja Trifunovic](https://github.com/nemtrif) for [his very helpful UTF8 CPP library](https://github.com/nemtrif/utfcpp).

 ## License

 This code, apart from the contributions in `core/third-party` is copyright [Moonshine AI](https://moonsh), and licensed under the MIT License, see LICENSE in this repository.

 The English-language models are also released under the MIT License. Models for other languages are released under the [Moonshine Community License](https://moonshine.ai), which is a non-commercial license.

 The code in `core/third-party` is licensed according to the terms of the open source projects it originates from, with details in a LICENSE file in each subfolder.