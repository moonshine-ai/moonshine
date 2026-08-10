# Building from Source

If you want to debug into the library internals, or add instrumentation to help understand its operation, or add improvements or customizations, all of the source is available for you to build it for yourself.

> [!TIP]
> Large model and TTS binaries are **not** stored in git. Before running
> [`scripts/test-core.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-core.sh) or offline TTS work, fetch them
> from the CDN (or the [Hugging Face mirror](https://huggingface.co/moonshine-ai/moonshine-voice-assets)):
>
> ```bash
> scripts/fetch-voice-assets.sh all
> ```
>
> A few compile-time embeds and ONNX Runtime prebuilts still use Git LFS. If you
> clone without LFS set up, you may see errors like
> `'version' does not name a type` when compiling embedded sources such as
> `community1_cpp_annote_embedded.cpp` (LFS pointers left as text). Install
> git-lfs and run `git lfs install` before cloning, or `git lfs pull` in an
> existing clone.

- [Cmake](#cmake)
- [Language Bindings](#language-bindings)
- [Porting](#porting)

## Cmake

The core engine of the library is contained in the `core` folder of this repo. It's written in C++ with a C interface for easy integration with other languages. We use cmake to build on all our platforms, and so the easiest way to get started is something like this:

<!-- doc-test: skip -->
```bash
cd core
mkdir -p build
cd build
cmake ..
cmake --build .
```

After that completes you should have a set of binary executables you can run on your own system. These executables are all unit tests, and expect to be run from the `test-assets` folder. You can run the build and test process in one step using the [`scripts/test-core.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-core.sh), or [`scripts/test-core.bat`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-core.bat) for Windows. All tests should compile and run without any errors.

## Language Bindings

There are various scripts for building for different platforms and languages, but to see examples of how to build for all of the supported systems you should look at [`scripts/build-all-platforms.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/build-all-platforms.sh). This is the script we call for every release, and it builds all of the artifacts we upload to the various package manager systems. [`docs/release-process.md`](https://github.com/moonshine-ai/moonshine/blob/main/docs/release-process.md) describes how releases are branched and versioned: `main` always matches the most recently published binaries, and development happens on a `dev-v<version>` candidate branch.

The different platforms and languages have a layer on top of the C interfaces to enable idiomatic use of the library within the different environments. The major systems each have their own folder under [`language-bindings/`](https://github.com/moonshine-ai/moonshine/tree/main/language-bindings), for example: [`language-bindings/python`](https://github.com/moonshine-ai/moonshine/blob/main/language-bindings/python/), [`language-bindings/android`](https://github.com/moonshine-ai/moonshine/blob/main/language-bindings/android/), and [`language-bindings/swift`](https://github.com/moonshine-ai/moonshine/blob/main/language-bindings/swift/) for iOS and MacOS. This is where you'll find the code that calls the underlying core library routines, and handles the event system for each platform.

## Porting

If you have a device that isn't supported, you can try [building using cmake](building-from-source.md#cmake) on your system. The only major dependency that the C++ core library has is [the Onnx Runtime](https://github.com/microsoft/onnxruntime). We include [pre-built binary library files](https://github.com/moonshine-ai/moonshine/blob/main/core/third-party/onnxruntime/lib/) for all our supported systems, but you'll need to find or build your own version if the libraries we offer don't cover your use case.

If you want to call this library from a language we don't support, then you should take a look at [the C interface bindings](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-c-api.h). Most languages have some way to call into C functions, so you can use these and the binding examples for other languages to guide your implementation.
