![Moonshine Voice Logo](images/logo.png)

# Moonshine Voice

## Voice Interfaces for Everyone

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building voice applications. 

 - Everything runs on-device, so it's fast, private, and there are never any server API charges. 
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

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/ios/Transcriber/Transcriber.xcodeproj` in Xcode.

### Android

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/android/Transcriber/` in Android Studio.

### Linux

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and then run:

```bash
cd core
mkdir build
cmake ..
cmake --build .
./moonshine-cpp-test
```

### MacOS

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/macos/MicTranscription/MicTranscription.xcodeproj` in Xcode.

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

## Downloading Models

The easiest way to obtain model files is using the Python module. After [installing it](#python) run the downloader like this:

```bash
python -m moonshine_voice.download --language en
```

You can use either the two-letter code or the English name for the `language` argument. If you want to see which languages are supported by your current version supply a bogus language as the argument:

```bash
python -m moonshine_voice.download --language foo
```

You can also optionally request a specific model architecture using the `model-arch` flag, chosen from the numbers in [moonshine-c-api.h](/core/moonshine-c-api.h). If no architecture is set, the script will load the highest-quality model available.

The download script will log the location of the downloaded model files and the model architecture, for example:

```bash
encoder_model.ort: 100%|███████████████████████████████████████████████████████| 29.9M/29.9M [00:00<00:00, 34.5MB/s]
decoder_model_merged.ort: 100%|██████████████████████████████████████████████████| 104M/104M [00:02<00:00, 52.6MB/s]
tokenizer.bin: 100%|█████████████████████████████████████████████████████████████| 244k/244k [00:00<00:00, 1.44MB/s]
Model download url: https://download.moonshine.ai/model/base-en/quantized/base-en
Model components: ['encoder_model.ort', 'decoder_model_merged.ort', 'tokenizer.bin']
Model arch: 1
Downloaded model path: /Users/petewarden/Library/Caches/moonshine_voice/download.moonshine.ai/model/base-en/quantized/base-en
```

The last two lines tell you which model architecture is being used, and where the model files are on disk. By default it uses your user cache directory, which is `~/Library/Caches/moonshine_voice` on MacOS, but you can use a different location by setting the `MOONSHINE_VOICE_CACHE` environment variable before running the script.

## Benchmarking

The core library includes a benchmarking tool that simulates processing live audio by loading a .wav audio file and feeding it in chunks to the model. To run it:

```
cd core
md build
cd build
cmake ..
cmake --build . --config Release
./benchmark
```

This will report the absolute time taken to process the audio, and what percentage of the audio file's duration that is. This percentage is helpful because it approximates how much of a compute load the model will be on your hardware. For example if it shows 20% then that means the speech processing will take a fifth of the compute time when running in your application, leaving 80% for the rest of your code.

By default it uses the Tiny English model that's embedded in the framework, but you can pass in the `--model-path` and `--model-arch` parameters to choose [one that you've downloaded](#downloading-models).

You can also choose how often the transcript should be updated using the `--transcription-interval` argument. This defaults to 0.5 seconds, but the right value will depend on how fast your application needs updates. Longer intervals reduce the compute required a bit, at the cost of slower updates.

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