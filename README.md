![Moonshine Voice Logo](images/logo.png)

# Moonshine Voice

**Voice Interfaces for Everyone**

- [Quickstart](#quickstart)
- [When should you choose Moonshine over Whisper?](#when-should-you-choose-moonshine-over-whisper)
- [Examples](#examples)
- [Using the Library](#using-the-library)
- [Adding the Library to your own App](#adding-the-library-to-your-own-app)
- [Downloading Models](#downloading-models)
- [Benchmarking](#benchmarking)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building voice applications. 

 - Everything runs on-device, so it's fast, private, and you don't need an account, credit card, or API keys. 
 - All models are based on our [cutting](https://arxiv.org/abs/2410.15608) [edge](https://arxiv.org/abs/2509.02523) research and trained from scratch, so we can offer higher accuracy than Whisper Large V3 at the top end, down to tiny 26MB models for constrained deployments.
 - It's cross-platform, running on [Python](#python), [iOS](#ios), [Android](#android), [MacOS](#macos), [Linux](#linux), [Windows](#windows), Raspberry Pis, [IoT devices](https://www.linkedin.com/posts/petewarden_most-of-the-recent-news-about-ai-seems-to-activity-7384664255242932224-v6Mr/), and wearables.
 - Batteries are included. Its high-level APIs offer complete solutions for common tasks like transcription, so you don't need to be an ML expert to use them.
 - It supports multiple languages, including English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, and Arabic.
 - The framework and models are optimized for streaming applications, offering low latency responses by doing most of the work while the user is still talking.

 ## Quickstart

[Join our community on Discord to get live support](https://discord.gg/27qp9zSRXF). 

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

### Raspberry Pi

You'll need a USB microphone plugged in to get audio input, but the Python pip package has been optimized for the Pi, so you can run:

```bash
 pip install moonshine-voice
 python -m moonshine_voice.mic_transcriber --language en
 ```

## When should you choose Moonshine over Whisper?

TL;DR - When you're working with live speech.

| Model | WER | MacBook Pro Latency | Linux x86 Latency | iPad | Pixel 9 | Raspberry Pi 5 | Parameters |
| ----- | --- | ------------------- | ----------------- | ---- | ------- | -------------- | ---------- |
| Moonshine Medium Streaming | % | 0ms | 0ms | 0ms | 0ms | 0ms | 200m |
| Whisper Large v3 | % | 0ms | 0ms | 0ms | 0ms | 0ms | 1500m |
| Moonshine Small Streaming | % | 0ms | 0ms | 0ms | 0ms | 0ms | ? |
| Whisper Small | % | 0ms | 0ms | 0ms | 0ms | 0ms | 244m |
| Moonshine Tiny Streaming | % | 0ms | 0ms | 0ms | 0ms | 0ms | 26m |
| Whisper Tiny | % | 0ms | 0ms | 0ms | 0ms | 0ms | 39m |

*See [benchmarks](#benchmarks) for how these were measured.*

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

Hopefully this gives you a good idea of how Moonshine compares to Whisper. If you're working with GPUs in the cloud on data in bulk where throughput is most important then Whisper (or Nvidia alternatives like Parakeet) offer advantages like batch processing, but we believe we can't be beat for live speech. We've built the framework and models we wished we'd had when we first started building applications with voice interfaces, so if you're working with live voice inputs, you might want to [give Moonshine a try](#quickstart).

## Examples

TK

## Using the Library

The Moonshine API is designed to take care of the details around capturing and transcribing live speech, giving application developers a high-level API focused on actionable events. I'll use Python to illustrate how it works, but the API is consistent across all the supported languages.

### Concepts

A [**Transcriber**](python/src/moonshine_voice/transcriber.py) takes in audio input and turns any speech into text. This is the first object you'll need to create to use Moonshine, and you'll give it a path to [the models you've downloaded](#downloading-models).

A **MicTranscriber** is a helper class based on the general transcriber that takes care of connecting to a microphone using your platform's built-in support (for example sounddevice in Python) and then feeding the audio in as it's captured.

A **Stream** is a handler for audio input. The reason streams exist is because you may want to process multiple audio inputs at once, and a transcriber can support those through multiple streams, without duplicating the model resources. If you only have one input, the transcriber class includes the same methods (start/stop/add_audio) as a stream, and you can use that interface instead and forget about streams.

A **TranscriptLine** is a data structure holding information about one line in the transcript. When someone is speaking, the library waits for short pauses (where punctuation might go in written language) and starts a new line. These aren't exactly sentences, since a speech pause isn't a sure sign of the end of a sentence, but this does break the spoken audio into segments that can be considered phrases. A line includes state such as whether the line has just started, is still being spoken, or is complete, along with its start time and duration.

A **Transcript** is a list of lines in time order holding information about what text has already been recognized, along with other state like when it was captured.

A **TranscriptLineEvent** contains information about changes to the transcript. Events include a new line being started, the text in a line being updated, and a line being completed. The event object includes the transcript line it's referring to as a member, holding the latest state of that line.

A **TranscriptListener** is a protocol that allows app-defined functions to be called when transcript events happen. This is the main way that most applications interact with the results of the transcription. When live speech is happening, applications usually need to respond or display results as new speech is recognized, and this approach allows you to handle those changes in a similar way to events from traditional user interfaces like touch screen gestures or mouse clicks on buttons.

### Getting Started

We have [examples](#examples) for most platforms so as a first step I recommend checking out what we have for the systems you're targeting.

Next, you'll need to [add the library to your project](#adding-the-library-to-your-own-app). We aim to provide pre-built binaries for all major platforms using their native package managers. On Python this means a pip install, for Android it's a Maven package, and for MacOS and iOS we provide a Swift package through SPM.

The transcriber needs access to the files for the model you're using, so after [downloading them](#downloading-models) you'll need to place them somewhere the application can find them, and make a note of the path. This usually means adding them as resources in your IDE if you're planning to distribute the app, or you can use hard-wired paths if you're just experimenting. The download script gives you the location of the models and their architecture type on your drive after it completes.

Now you can try creating a transcriber. Here's what that looks like in Python:

```python
transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
```

If the model isn't found, or if there's any other error, this will throw an exception with information about the problem. You can also check the console for logs from the core library, these are printed to `stderr` or your system's equivalent.

Now we'll create a listener that contains the app logic that you want triggered when the transcript updates, and attach it to your transcriber:

```python
class TestListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")

transcriber.add_listener(listener)
```

The transcriber needs some audio data to work with. If you want to try it with the microphone you can update your transcriber creation line to use a MicTranscriber instead, but if you want to start with a .wav file for testing purposes here's how you feed that in:

```python
    audio_data, sample_rate = load_wav_file(wav_path)

    transcriber.start()

    # Loop through the audio data in chunks to simulate live streaming
    # from a microphone or other source.
    chunk_duration = 0.1
    chunk_size = int(chunk_duration * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i: i + chunk_size]
        transcriber.add_audio(chunk, sample_rate)

    transcriber.stop()
```

The important things to notice here are:

 - We create an array of mono audio data from a wav file, using the convenience `load_wav_file()` function that's part of the Moonshine library.
 - We start the transcriber to activate its processing code.
 - The loop adds audio in chunks. These chunks can be any length and any sample rate, the library takes care of all the housekeeping.
 - As audio is added, the event listener you added will be called, giving information about the latest speech.

In a real application you'd be calling `add_audio()` from an audio handler that's receiving it from your source. Since the library can handle arbitrary durations and sample rates, just make sure it's mono and otherwise feed it in as-is.

The transcriber analyses the speech at a default interval of every 500ms of input. You can change this with the `update_interval` argument to the transcriber constructor. For streaming models most of the work is done as the audio is being added, and it's automatically done at the end of a phrase, so changing this won't usually affect the workload or latency massively.

The key takeaway is that you usually don't need to worry about the transcript data structure itself, the event system tells you when something important happens. You can manually trigger a transcript update by calling `update_transcription()` which returns a transcript object with all of the information about the current session if you do need to examine the state.

By calling `start()` and `stop()` on a transcriber (or stream) we're beginning and ending a session. Each session has one transcript document associated with it, and it is started fresh on every `start()` call, so you should make copies of any data you need from the transcript object before that.

The transcriber class also offers a simpler `transcribe_without_streaming()` method, for when you have an array of data from the past that you just want to analyse, such as a file or recording.

### Debugging

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

## Roadmap

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