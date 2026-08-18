# Adding the Library to your own App

We distribute the library through the most widely-used package managers for each platform. Here's how you can use these to add the framework to an existing project on different systems.

=== "Javascript"

    The WebAssembly package is published on npm as [`@moonshine-ai/moonshine-wasm`](https://www.npmjs.com/package/@moonshine-ai/moonshine-wasm).

    In Node or a bundler project:

    ```bash
    npm install @moonshine-ai/moonshine-wasm
    ```

    ```js
    import { MicTranscriber, ModelArch } from '@moonshine-ai/moonshine-wasm';
    ```

    In the browser you can import the same package from a CDN with no install step:

    ```js
    import { MicTranscriber, ModelArch } from 'https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm/dist/index.js';
    ```

    For a full working tree, see the [web examples](../examples.md).

=== "Python"

    The Python package is [hosted on PyPi](https://pypi.org/project/moonshine-voice/), so all you should need to do to install it is `pip install moonshine-voice`, and then `import moonshine_voice` in your project.

    #### Command-line tools

    Installing the pip package adds a `moonshine-voice` command (with a shorter `moonshine` alias) that groups the built-in tools as subcommands. These are designed for one-off use cases, if you need multiple Moonshine calls for the same task then loading the models once from Python will be more efficient.

    <!-- doc-test: parse-only -->
    ```bash
    moonshine-voice --help
    ```

    | Command | Description |
    | --- | --- |
    | `moonshine-voice mic` | Transcribe live microphone input to the terminal. |
    | `moonshine-voice transcribe` | Transcribe a WAV file (optionally with speaker IDs / word timestamps). |
    | `moonshine-voice tts` | Synthesize speech from text to a WAV file or audio device. |
    | `moonshine-voice agent` | Run a spoken agent flow (wifi setup) from the microphone. |
    | `moonshine-voice download` | Download STT, TTS, G2P, or embedding model assets. |
    | `moonshine-voice g2p` | Convert text to phonemes (IPA). |

    Run `moonshine-voice <command> --help` for the options each one accepts. Every subcommand is equivalent to running the underlying module directly, so `moonshine-voice mic --language en` and `python -m moonshine_voice.mic_transcriber --language en` do exactly the same thing.

=== "iOS"

    For iOS we use the Swift Package Manager, with [an auto-updated GitHub repository](https://github.com/moonshine-ai/moonshine-swift/) holding each version. To use this right-click on the file view sidebar in Xcode and choose "Add Package Dependencies..." from the menu. A dialog should open up, paste `https://github.com/moonshine-ai/moonshine-swift/` into the top search box and you should see `moonshine-swift`. Select it and choose "Add Package", and it should be added to your project. You should now be able to `import MoonshineVoice` and use the library. You will need to add any model files you use to your app bundle and ensure they're copied during the deployment phase, so they can be accessed on-device.

    For reference purposes you can find an Xcode project with these changes applied in [`examples/ios/Transcriber`](https://github.com/moonshine-ai/moonshine/blob/main/examples/ios/Transcriber).

=== "Android"

    On Android we publish [the package to Maven](https://mvnrepository.com/artifact/ai.moonshine/moonshine-voice). To include it in your project using Android Studio and Gradle, first add the version number you want to the `gradle/libs.versions.toml` file by inserting a line in the `[versions]` section, for example `moonshineVoice = "0.1.5"`. Then in the `[libraries]` part, add a reference to the package: `moonshine-voice = { group = "ai.moonshine", name = "moonshine-voice", version.ref = "moonshineVoice" }`.

    Finally, in your `app/build.gradle.kts` add the library to the `dependencies` list: `implementation(libs.moonshine.voice)`. The [`examples/android/Transcriber`](https://github.com/moonshine-ai/moonshine/blob/main/examples/android/Transcriber/) and [`examples/android/TextToSpeech`](https://github.com/moonshine-ai/moonshine/blob/main/examples/android/TextToSpeech/) samples use the same coordinates (`moonshineVoice = "0.1.5"` in their catalogs).

=== "Linux"

    Prebuilt shared libraries for x86_64 and arm64 Linux ship on [GitHub Releases](https://github.com/moonshine-ai/moonshine/releases/latest). The easiest way to pull one in is the helper in [`examples/c++`](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/README.md):

    ```bash
    cd examples/c++
    ./download-library.sh
    ```

    That extracts a `moonshine-voice/` folder with headers under `include/` and `libmoonshine.so` (plus its ONNX Runtime dependency) under `lib/`. Add those paths to your compile and link lines, and include the C++ binding:

    ```bash
    g++ your_app.cpp \
      -Imoonshine-voice/include \
      -Lmoonshine-voice/lib \
      -lmoonshine \
      -Wl,-rpath,'$ORIGIN/moonshine-voice/lib' \
      -o your_app
    ```

    ```cpp
    #include "moonshine-cpp.h"
    ```

    See [`examples/c++/transcriber.cpp`](https://github.com/moonshine-ai/moonshine/blob/main/examples/c++/transcriber.cpp) for a minimal end-to-end program.

=== "MacOS"

    For MacOS we use the Swift Package Manager, with [an auto-updated GitHub repository](https://github.com/moonshine-ai/moonshine-swift/) holding each version. To use this right-click on the file view sidebar in Xcode and choose "Add Package Dependencies..." from the menu. A dialog should open up, paste `https://github.com/moonshine-ai/moonshine-swift/` into the top search box and you should see `moonshine-swift`. Select it and choose "Add Package", and it should be added to your project. You should now be able to `import MoonshineVoice` and use the library. You will need to add any model files you use to your app bundle and ensure they're copied during the deployment phase, so they can be accessed on-device.

    For reference purposes you can find an Xcode project with these changes applied in [`examples/macos/BasicTranscription`](https://github.com/moonshine-ai/moonshine/blob/main/examples/macos/BasicTranscription/).

=== "Windows"

    We couldn't find a single package manager that is used by most Windows developers, so instead we've made the raw library and headers available as a download. The script in [`examples/windows/cli-transcriber/download-lib.bat`](https://github.com/moonshine-ai/moonshine/blob/main/examples/windows/cli-transcriber/download-lib.bat) will fetch these for you. You'll see an `include` folder that you should add to the include search paths in your project settings, and a `lib` directory that you should add to the library search paths. Then add all of the library files in the `lib` folder to your project's linker dependencies.

    The recommended interface to use on Windows is the C++ language binding. This is a header-only library that offers a higher-level API than the underlying C version. You can `#include "moonshine-cpp.h"` to access Moonshine from your C++ code. If you want to see an example of all these changes together, take a look at [`examples/windows/cli-transcriber`](https://github.com/moonshine-ai/moonshine/blob/main/examples/windows/cli-transcriber).

=== "Raspberry Pi"

    The same Python package used on desktop works on Raspberry Pi OS. With a USB microphone plugged in:

    ```bash
    pip install moonshine-voice
    ```

    If pip warns about system packages, either use a virtual environment or `pip install --break-system-packages moonshine-voice`. Then `import moonshine_voice` in your project, or use the `moonshine-voice` CLI the same way as on other platforms.

    For Pi-specific samples, see the [Raspberry Pi examples](../examples.md).
