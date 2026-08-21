# Downloading Models

- [Automatic Downloading](#automatic-downloading)
- [Speech to Text Models](#speech-to-text-models)
- [Embedding Models](#embedding-models)
- [Text to Speech Models](#text-to-speech-models)

## Automatic Downloading

High-level helpers such as `MicTranscriber`, `TextToSpeech`, and `AgentFlow` fetch whatever they need on first `load()`, so most apps never call a download API by hand. When you want to stage files into a directory you control — or warm a cache before going offline — the bindings also expose an **opt-in** downloader built on the same dependency catalog as the Python CLI.

This is strictly opt-in for the low-level path: apps that bundle their models and load them with the usual `from files` / `from assets` / `from memory` paths behave exactly as before and never touch the network. The downloader resolves the file list from the native catalog (so you never hardcode filenames), writes each file atomically (through a `.part` file), resumes interrupted transfers with HTTP `Range`, checks free space before large writes, and reports per-file progress.

=== "Javascript"

    In the browser, `MicTranscriber.load()`, `TextToSpeech.load()`, and `AgentFlow.load()` download missing assets through an internal `AssetDownloader` into the Cache Storage bucket `moonshine-models-v1`. Later visits reuse that cache and run offline.

    ```js
    import { MicTranscriber } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';

    const mic = new MicTranscriber()
      .onProgress((fraction, file) => {
        console.log(`${(fraction * 100).toFixed(0)}% ${file}`);
      })
      .onText((text) => showInProgress(text))
      .onLine((line) => appendLine(line.text));

    await mic.load();
    await mic.start();
    ```

    Append `?fresh=1` to a demo URL during development to clear the model cache and force a redownload. Pass `modelsFrom` / a custom `baseUrl` when you host the assets yourself instead of using the CDN.

=== "Python"

    `MicTranscriber`, `TextToSpeech`, and `AgentFlow` download into the user cache on first `load()` (override the location with `MOONSHINE_VOICE_CACHE`). Pass `on_progress()` to drive a UI; attaching a handler also silences the default terminal progress bars.

    ```python
    from moonshine_voice import MicTranscriber

    mic = (
        MicTranscriber()
        .language("en")
        .on_progress(lambda fraction, file: print(f"{fraction:.0%} {file}"))
        .on_text(lambda text: show_in_progress(text))
        .on_line(lambda line: append_line(line.text))
    )
    mic.load()
    mic.start()
    ```

    Use `models_from(path)` when you already have a directory on disk (for example after running the [CLI download](#speech-to-text-models) commands below). Bundled or pre-placed models never hit the network.

=== "iOS"

    `MicTranscriber.load()` and `TextToSpeech.load()` download automatically. For explicit control, `AssetDownloader.ensureModelPresent` fetches whatever is missing under a directory you choose and returns that directory, ready to hand to `Transcriber`, `MicTranscriber`, or `TextToSpeech`. Call it off the main actor (it is `async`).

    ```swift
    import MoonshineVoice

    let modelDir = URL.cachesDirectory.appending(path: "moonshine/tiny-en")

    // Speech-to-text: download the default English model (add includeSpelling: true for the
    // alphanumeric spelling model, or pass modelArch: to pick a specific architecture).
    let downloader = AssetDownloader(allowsCellularAccess: false)  // Wi-Fi only
    try await downloader.ensureModelPresent(root: modelDir, spec: .stt(language: "en")) { progress in
        print("\(progress.relativePath): \(progress.bytesDownloaded)/\(progress.bytesTotal)")
    }
    let transcriber = try Transcriber(modelPath: modelDir.path, modelArch: .tiny)

    // Text embeddings (the embeddinggemma-300m model is large — a few hundred MB even at q4):
    try await downloader.ensureModelPresent(root: embeddingDir, spec: .embedding(variant: "q4"))

    // Text to speech (files land under the directory you pass as g2p_root):
    try await downloader.ensureModelPresent(root: ttsRoot, spec: .tts(language: "en_us", voice: "kokoro_af_heart"))
    ```

    `isModelPresent(root:spec:)` is a cheap synchronous check you can use to skip the download UI entirely when everything is already on disk. Download failures throw `AssetDownloadError` (HTTP status, insufficient space, cancellation, …) so you can distinguish "couldn't fetch" from a later load failure. No extra `Info.plist` entries or permissions are required for HTTPS downloads.

    Opt-in network tests: `MOONSHINE_DOWNLOAD_TESTS=1 swift test --filter AssetDownloaderNetworkTests`.

=== "Android"

    High-level `load()` paths download automatically. The Android `AssetDownloader` mirrors the Swift API and performs blocking network I/O, so run it off the main thread. For reliable background downloads that survive process death, honor a network constraint, and retry with backoff, use `MoonshineDownloadWorker` (WorkManager).

    ```java
    File modelDir = new File(context.getFilesDir(), "moonshine/tiny-en");

    // Direct (call from a background thread / coroutine):
    File root = new AssetDownloader().ensureModelPresent(
            modelDir, ModelSpec.stt("en"),
            (path, index, total, done, size) -> Log.i("dl", path + " " + done + "/" + size));
    Transcriber transcriber = new Transcriber();
    transcriber.loadFromFiles(root.getAbsolutePath(), JNI.MOONSHINE_MODEL_ARCH_TINY);

    // Or via WorkManager, downloading only over unmetered (e.g. Wi-Fi) connections:
    OneTimeWorkRequest request =
            MoonshineDownloadWorker.buildRequest(modelDir, ModelSpec.stt("en"), /*requireUnmetered=*/ true);
    WorkManager.getInstance(context).enqueue(request);
    ```

    `ModelSpec` also has `tts(language, voice)`, `embedding(modelName, variant)`, and `g2p(language)` factories, matching the Swift specs. The library manifest declares the `INTERNET` and `ACCESS_NETWORK_STATE` permissions, which merge into your app automatically; observe `MoonshineDownloadWorker` progress via `WorkInfo.getProgress()` using the worker's `PROGRESS_*` data keys. Using the downloader pulls in OkHttp and WorkManager transitively; apps that bundle their models still ship these but never invoke the network path.

    Opt-in network tests: `./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=ai.moonshine.voice.AssetDownloaderTest -Pandroid.testInstrumentationRunnerArguments.moonshineDownloadTests=1` (needs a connected device/emulator).

=== "Linux"

    Prefer the Python helpers: `MicTranscriber.load()`, `TextToSpeech.load()`, or `AgentFlow.load()` download into the user cache on first use. To warm the cache ahead of time (for example before taking a device offline), use the [CLI commands](#speech-to-text-models) below.

    C++ apps that construct a `Transcriber` from a path should run `moonshine-voice download` (or `./download-library.sh` in [`examples/c++`](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/README.md), which also fetches a sample English model) and pass that directory to the engine.

=== "MacOS"

    `MicTranscriber.load()` and `TextToSpeech.load()` download automatically. For explicit control, `AssetDownloader.ensureModelPresent` fetches whatever is missing under a directory you choose and returns that directory, ready to hand to `Transcriber`, `MicTranscriber`, or `TextToSpeech`. Call it off the main actor (it is `async`).

    ```swift
    import MoonshineVoice

    let modelDir = URL.cachesDirectory.appending(path: "moonshine/tiny-en")

    // Speech-to-text: download the default English model (add includeSpelling: true for the
    // alphanumeric spelling model, or pass modelArch: to pick a specific architecture).
    let downloader = AssetDownloader(allowsCellularAccess: false)  // Wi-Fi only
    try await downloader.ensureModelPresent(root: modelDir, spec: .stt(language: "en")) { progress in
        print("\(progress.relativePath): \(progress.bytesDownloaded)/\(progress.bytesTotal)")
    }
    let transcriber = try Transcriber(modelPath: modelDir.path, modelArch: .tiny)

    // Text embeddings (the embeddinggemma-300m model is large — a few hundred MB even at q4):
    try await downloader.ensureModelPresent(root: embeddingDir, spec: .embedding(variant: "q4"))

    // Text to speech (files land under the directory you pass as g2p_root):
    try await downloader.ensureModelPresent(root: ttsRoot, spec: .tts(language: "en_us", voice: "kokoro_af_heart"))
    ```

    `isModelPresent(root:spec:)` is a cheap synchronous check you can use to skip the download UI entirely when everything is already on disk. Download failures throw `AssetDownloadError` (HTTP status, insufficient space, cancellation, …) so you can distinguish "couldn't fetch" from a later load failure. No extra `Info.plist` entries or permissions are required for HTTPS downloads.

    Opt-in network tests: `MOONSHINE_DOWNLOAD_TESTS=1 swift test --filter AssetDownloaderNetworkTests`.

=== "Windows"

    Prefer the Python helpers: `MicTranscriber.load()`, `TextToSpeech.load()`, or `AgentFlow.load()` download into the user cache on first use. To warm the cache ahead of time, use the [CLI commands](#speech-to-text-models) below.

    C++ apps (including [`examples/windows/cli-transcriber`](https://github.com/moonshine-ai/moonshine/blob/main/examples/windows/cli-transcriber)) should download models with `moonshine-voice download` or use the models bundled in the release archive, then point the engine at that directory.

=== "Raspberry Pi"

    Same as Python: `pip install moonshine-voice`, then let `MicTranscriber.load()` / `TextToSpeech.load()` / `AgentFlow.load()` fetch into the cache on first use. Warm the cache with `moonshine-voice download` before going offline if needed.

### Where files go

When you use `AssetDownloader` (or the Swift/Android `ensureModelPresent` helpers), you pick the destination directory, so choose one appropriate for your platform's cache/storage policy (for example `URL.cachesDirectory` on Apple platforms, or `context.getFilesDir()` / `context.getCacheDir()` on Android). Files are laid out under that directory exactly as the loaders expect: STT and embedding files use their bare filenames, while TTS/G2P assets keep their canonical relative paths (e.g. `en_us/dict.tsv`) so the engine can find them from the root alone.

In the browser, downloaded models live in Cache Storage (`moonshine-models-v1`) rather than a filesystem path. On Python, the default cache is under your user cache directory (`~/Library/Caches/moonshine_voice` on MacOS); override it with `MOONSHINE_VOICE_CACHE`.

`scripts/test-model-downloads.sh` runs the native catalog samples and then drives the opt-in Swift and Android network download tests automatically (skipping a platform when its toolchain or a device is unavailable). The mocked, offline `AssetDownloaderTests` still run as part of the default `swift test` and cover manifest parsing, resume, and error handling without a network.

## Speech to Text Models

The easiest way to get the model files required for transcription is by using the Python download module. After [installing it](../quickstart.md) run the downloader like this:

```bash
moonshine-voice download --stt --language en
```

You can use either the two-letter code or the English name for the `language` argument. If you want to see which languages are supported by your current version they're [listed below](../models/available-models.md), or you can supply a bogus language as the argument to this command:

<!-- doc-test: expect-error -->
```bash
moonshine-voice download --stt --language foo
```

You can also optionally request a specific model architecture using the `model-arch` flag, chosen from the numbers in [moonshine-c-api.h](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-c-api.h). If no architecture is set, the script will load the highest-quality model available.

The download script will log the location of the downloaded model files and the model architecture, for example:

```text
adapter.ort: 100%|██████████████████████████████████████████████████████████████| 3.48M/3.48M [00:00<00:00, 18.6MB/s]
cross_kv.ort: 100%|█████████████████████████████████████████████████████████████| 11.0M/11.0M [00:00<00:00, 39.4MB/s]
decoder_kv.ort: 100%|███████████████████████████████████████████████████████████████| 139M/139M [00:01<00:00, 100MB/s]
encoder.ort: 100%|██████████████████████████████████████████████████████████████| 89.8M/89.8M [00:01<00:00, 83.1MB/s]
frontend.weights.ort: 100%|█████████████████████████████████████████████████████| 11.3M/11.3M [00:00<00:00, 65.7MB/s]
frontend.model.ort: 100%|█████████████████████████████████████████████████████████| 28.0k/28.0k [00:00<00:00, 1.88MB/s]
streaming_config.json: 100%|██████████████████████████████████████████████████████| 513/513 [00:00<00:00, 1.88MB/s]
tokenizer.bin: 100%|█████████████████████████████████████████████████████████████| 244k/244k [00:00<00:00, 3.06MB/s]
spelling_cnn.ort: 100%|█████████████████████████████████████████████████████████| 1.59M/1.59M [00:00<00:00, 10.2MB/s]
spelling_cnn_meta.json: 100%|█████████████████████████████████████████████████████| 622/622 [00:00<00:00, 1.72MB/s]
Model download url: https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21
Model components: ['adapter.ort', 'cross_kv.ort', 'decoder_kv.ort', 'encoder.ort', 'frontend.model.ort', 'frontend.weights.ort', 'streaming_config.json', 'tokenizer.bin']
Model arch: 5
Downloaded model path: /Users/petewarden/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21
```

Since no architecture was requested here, this downloaded Medium Streaming (architecture 5), the highest-quality English model. The two `spelling_cnn` files at the end are the alphanumeric spelling model, which the downloader fetches alongside the main model when one is published for the language.

The last two lines tell you which model architecture is being used, and where the model files are on disk. By default it uses your user cache directory, which is `~/Library/Caches/moonshine_voice` on MacOS, but you can use a different location by setting the `MOONSHINE_VOICE_CACHE` environment variable before running the script.

## Embedding Models

The download module also helps you obtain the assets needed to match spoken phrases, primarily a sentence embedding model. `AgentFlow` fetches this for you on first use, so you only need this command to warm the cache ahead of time — before shipping a device that will be offline, for example.

```bash
moonshine-voice download --embedding
```

```text
model_q4.ort: 100%|████████████████████████████████████████████████| 189M/189M [00:02<00:00, 90.1MB/s]
tokenizer.bin: 100%|███████████████████████████████████████████████| 2.46M/2.46M [00:00<00:00, 13.9MB/s]
Embedding model path: /Users/petewarden/Library/Caches/moonshine_voice/download.moonshine.ai/model/embeddinggemma-300m
/Users/petewarden/Library/Caches/moonshine_voice/download.moonshine.ai/model/embeddinggemma-300m
```

## Text to Speech Models

A large variety of models, dictionaries and other files are needed for TTS, and these vary widely by language. You can use the download module to pull down exactly what you need for a particular language, and optionally a voice:

```bash
moonshine-voice download --tts --root /tmp/tts-files/
```

```text
dict_filtered_heteronyms.tsv: 100%|██████████████████████████████| 2.77M/2.77M [00:00<00:00, 15.5MB/s]
g2p-config.json: 100%|██████████████████████████████████████████████| 60.0/60.0 [00:00<00:00, 160kB/s]
model.ort: 100%|█████████████████████████████████████████████████| 21.1M/21.1M [00:00<00:00, 37.7MB/s]
onnx-config.json: 100%|████████████████████████████████████████████| 4.53k/4.53k [00:00<00:00, 11.7MB/s]
model.ort: 100%|█████████████████████████████████████████████████| 88.3M/88.3M [00:01<00:00, 85.6MB/s]
config.json: 100%|███████████████████████████████████████████████| 2.30k/2.30k [00:00<00:00, 6.88MB/s]
af_heart.kokorovoice: 100%|████████████████████████████████████████| 510k/510k [00:00<00:00, 3.82MB/s]
TTS assets root (use as g2p_root): /private/tmp/tts-files
/private/tmp/tts-files
```

The downloaded models are placed in child folders underneath the root folder, and by default the text to speech module expects the files to have the same relative paths so it can find them automatically given only the parent's path. If you do need to move them to different locations, you can supply new paths for each file using the `options()` setter on `TextToSpeech` (before `load()`), with the usual relative path as the key, and the actual path to the file as the value.

If you have an application that may be stored in an arbitrary location after installation, you can also pass in a `tts_root` value as an option to set the path to the actual root folder of the TTS data at runtime.
