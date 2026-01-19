![Moonshine Voice Logo](images/logo.png)

- [Quickstart](#quickstart)
- [When should you choose Moonshine over Whisper?](#when-should-you-choose-moonshine-over-whisper)
- [Examples](#examples)
- [Using the Library](#using-the-library)
- [Adding the Library to your own App](#adding-the-library-to-your-own-app)
  - [iOS](#ios)
  - [Android](#android)
  - [Linux](#linux)
  - [MacOS](#macos)
  - [Windows](#windows)
- [Downloading Models](#downloading-models)
- [Benchmarking](#benchmarking)
- [Acknowledgements](#acknowledgements)
- [License](#license)

**Voice Interfaces for Everyone**

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building voice applications. 

 - Everything runs on-device, so it's fast, private, and you don't need an account, credit card, or API keys. 
 - All models are based on our [cutting](https://arxiv.org/abs/2410.15608) [edge](https://arxiv.org/abs/2509.02523) research and trained from scratch, so we can offer higher accuracy than Whisper Large V3 at the top end, down to tiny 26MB models for constrained deployments.
 - It's cross-platform, running on [Python](#python), [iOS](#ios), [Android](#android), [MacOS](#macos), [Linux](#linux), [Windows](#windows), Raspberry Pis, [IoT devices](https://www.linkedin.com/posts/petewarden_most-of-the-recent-news-about-ai-seems-to-activity-7384664255242932224-v6Mr/), and wearables.
 - Batteries are included. Its high-level APIs offer complete solutions for common tasks like transcription, so you don't need to be an ML expert to use them.
 - It supports multiple languages, including English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, and Arabic.
 - The framework and models are optimized for streaming applications, offering low latency responses by doing most of the work while the user is still talking.

 ## Quickstart

[Join our community on Discord to get live support](https://discord.gg/27qp9zSRXF). 

 **Python**

 ```bash
 pip install moonshine-voice
 python -m moonshine_voice.mic_transcriber --language en
 ```

Listens to the microphone and prints updates to the transcript as they come in.

**iOS**

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/ios/Transcriber/Transcriber.xcodeproj` in Xcode.

**Android**

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/android/Transcriber/` in Android Studio.

**Linux**

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and then run:

```bash
cd core
mkdir build
cmake ..
cmake --build .
./moonshine-cpp-test
```

**MacOS**

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository and open `examples/macos/MicTranscription/MicTranscription.xcodeproj` in Xcode.

**Windows**

[Download](https://github.com/moonshine-ai/moonshine-v2/archive/refs/heads/main.zip) or `git clone` this repository. 

[Install Moonshine in Python](#python) for model downloading.

In the terminal:

```batch
pip install moonshine-voice
cd examples\windows\cli-transcriber
.\download-lib.bat
msbuild cli-transcriber.sln /p:Configuration=Release /p:Platform=x64
python -m moonshine_voice.download --language en
x64\Release\cli-transcriber.exe --model-path <path from the download command> --model-arch <number from the download command>
```

**Raspberry Pi**

You'll need a USB microphone plugged in to get audio input, but the Python pip package has been optimized for the Pi, so you can run:

```bash
 pip install moonshine-voice
 python -m moonshine_voice.mic_transcriber --language en
 ```

## When should you choose Moonshine over Whisper?

TL;DR - When you're working with live speech.

[OpenAI's release of their Whisper family of models]() was a massive step forward for open-source speech to text. They offered a range of sizes, allowing developers to trade off compute and storage space against accuracy to fit their applications. Their biggest models, like Large v3, also gave accuracy scores that were higher than anything available outside of large tech companies like Google or Apple. At Moonshine we were early and enthusiastic adopters of Whisper, and we still remain big fans of the models and the great frameworks like [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) and others that have been built around them.

However, as we built applications that needed a live voice interface we found we needed features that weren't available through Whisper:

 - **Whisper always operates on a 30-second input window**. This isn't an issue when you're processing audio in large batches, you can usually just look ahead in the file and find a 30-second-ish chunk of speech to apply it to. Voice interfaces can't look ahead to create larger chunks from their input stream, and phrases are seldom longer than five to ten seconds. This means there's a lot of wasted computation encoding zero padding in the encoder and decoder, which means longer latency in returning results. Since one of the most important requirements for any interface is responsiveness, usually defined as latency below 200ms, this hurts the user experience even on platforms that have compute to spare, and makes it unusable on more constrained devices.
- **Whisper doesn't cache anything**. Another common requirement for voice interfaces is that they display feedback as the user is talking, so that they know the app is listening and understanding them. This means calling the speech to text model repeatedly over time as a sentence is spoken. Most of the audio input is the same, with only a short addition to the end. Even though a lot of the input is constant, Whisper starts from scratch every time, doing a lot of redundant work on audio that it has seen before. Like the fixed input window, this unnecessary latency impairs the user experience.
- **Whisper supports a lot of languages poorly**. Whisper's multilingual support is an incredible feat of engineering, and demonstrated a single model could handle many languages, and even offer translations. This chart from OpenAI ([raw data in Appendix D-2.4](https://cdn.openai.com/papers/whisper.pdf)) shows the drop-off in Word Error Rate (WER) with the very largest 1.5 billion parameter model. 
![Language Chart](images/lang-chart.png)

82 languages are listed, but only 33 have sub-20% WER (what we consider usable). For the Base model size commonly used on edge devices, only 5 languages are under 20% WER. Asian languages like Korean and Japanese stand out as the native tongue of large markets with a lot of tech innovation, but Whisper doesn't offer good enough accuracy to use in most applications The proprietary in-house versions of Whisper that are available through OpenAI's cloud API seem to offer better accuracy, but aren't available as open models.

- **Fragmented edge support**. A fantastic ecosystem has grown up around Whisper, there are a lot of mature frameworks you can use to deploy the models. However these often tend to be focused on desktop-class machines and operating systems. There are projects you can use across edge platforms like iOS, Android, or Raspberry Pi OS, but they tend to have different interfaces, capabilities, and levels of optimization. This made building applications that need to run on a variety of devices unnecessarily difficult.

All these limitations drove us to create our own family of models that better meet the needs of live voice interfaces. It took us some time since the combined size of the open speech datasets available is tiny compared to the amount of web-derived text data, but after extensive data-gathering work, we were able to release [the first generation of Moonshine models](https://arxiv.org/abs/2410.15608). These removed the fixed-input window limitation along with some other architectural improvements, and gave significantly lower latency than Whisper in live speech applications, often running 5x faster or more.

However we kept encountering applications that needed even lower latencies on even more constrained platforms. We also wanted to offer higher accuracy than the Base-equivalent that was the top end of the initial models. That led us to this second generation of Moonshine models, which offer:

 - **Flexible input windows**. You can supply any length of audio (though we recommend staying below around 30 seconds) and the model will only spend compute on that input, no zero-padding required. This gives us a significant latency boost.
 - **Caching for streaming**. Our models now support incremental addition of audio over time, and they cache the input encoding and part of the decoder's state so that we're able to skip even more of the compute, driving latency down dramatically.
 - **Language-specific models**. We have gathered data and trained models for multiple languages, including Arabic, Japanese, Korean, Spanish, Ukrainian, Vietnamese, and Chinese. As we discuss in our [Flavors of Moonshine paper](https://arxiv.org/abs/2509.02523), we've found that we can get much higher accuracy for the same size and compute if we restrict a model to focus on just one language, compared to training one model across many.
 - **Cross-platform library support**. We're building applications ourselves, and needed to be able to deploy these models across Linux, MacOS, Windows, iOS, and Android, as well as use them from languages like Python, Swift, Java, and C++. To support this we architected a portable C++ core library that handles all of the processing, uses OnnxRuntime for good performance across systems, and then built native interfaces for all the required high-level languages. This allows developers to learn one API, and then deploy it almost anywhere they want to run.
 - **Better accuracy than Whisper V3 Large**. On [HuggingFace's OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), our newest streaming model for English, Medium Streaming, achieves a lower word-error rate than the most-accurate Whisper model from OpenAI. This is despite Moonshine's version using 200 million parameters, versus Large v3's 1.5 billion, making it much easier to deploy on the edge.

Hopefully this gives you a good idea of how Moonshine compares to Whisper. We've built the framework and models we wished we'd had when we first started building applications with voice interfaces, and if you're working with live voice inputs, we think you might want to [give Moonshine a try](#quickstart).

## Examples



## Using the Library

The Moonshine API is designed to take care of the details around capturing and transcribing live speech, giving application developers a high-level API focused on actionable events. This API is consistent across all the high-level languages the 

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

The easiest way to get the model files is using the Python module. After [installing it](#python) run the downloader like this:

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

 - Lambda Labs and Stephen Balaban for supporting our model training through [their foundational model grants](https://lambda.ai/research).
 - The TEN team for open sourcing [their voice-activity detection model](https://github.com/TEN-framework/ten-vad).
 - The ONNX Runtime community for building [a fast, cross-platform inference engine](https://github.com/microsoft/onnxruntime).
 - [Viktor Kirilov](https://github.com/onqtam) for [his fantastic DocTest C++ testing framework](https://github.com/doctest/doctest).
 - [Nemanja Trifunovic](https://github.com/nemtrif) for [his very helpful UTF8 CPP library](https://github.com/nemtrif/utfcpp).

 ## License

 This code, apart from the contributions in `core/third-party` is copyright [Moonshine AI](https://moonsh), and licensed under the MIT License, see LICENSE in this repository.

 The English-language models are also released under the MIT License. Models for other languages are released under the [Moonshine Community License](https://moonshine.ai), which is a non-commercial license.

 The code in `core/third-party` is licensed according to the terms of the open source projects it originates from, with details in a LICENSE file in each subfolder.