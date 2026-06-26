# Building the Android library for 32-bit ARM (`armeabi-v7a`)

By default the Android library is built for `arm64-v8a` only. This guide
explains how to also build it for 32-bit ARM (`armeabi-v7a`), e.g. for older
phones or 32-bit Android TV boxes running a 32-bit image on an ARMv8 SoC.

## Background: the SIGBUS fix

Creating a transcriber on `armeabi-v7a` used to crash with
`SIGBUS / BUS_ADRALN` inside `CreateSessionFromArray`, before transcription
started. The auxiliary models are baked into the library as
`static const uint8_t[]` arrays (1-byte aligned); ORT parses the `.ort`
flatbuffer in place using aligned (e.g. 64-bit) loads, which fault on ARM32.

This is already fixed in source: `ort_session_from_memory()` in
[`core/ort-utils/ort-utils.cpp`](../core/ort-utils/ort-utils.cpp) copies any
non-page-aligned buffer into a page-aligned, lifetime-persistent buffer before
handing it to ORT. The fix is a no-op on `arm64-v8a`/x86 and for file/mmap
loaded models, so nothing below changes those builds.

## Steps

### 1. Enable the ABI in `build.gradle.kts`

Add `armeabi-v7a` to the ABI filter:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
}
```

### 2. Provide a 32-bit `libonnxruntime.so`

The repo only ships the `arm64-v8a` ONNX Runtime. Extract the 32-bit one from
the matching ONNX Runtime Android AAR
(`com.microsoft.onnxruntime:onnxruntime-android:1.23.2`) — the file is at
`jni/armeabi-v7a/libonnxruntime.so` inside the AAR (an AAR is a zip) — and copy
it to **both** locations:

- `core/third-party/onnxruntime/lib/android/armeabi-v7a/libonnxruntime.so`
  (used when CMake links the native code)
- `src/main/jniLibs/armeabi-v7a/libonnxruntime.so`
  (packaged into the APK/AAR at runtime)

Verify it is the right binary:

```sh
file core/third-party/onnxruntime/lib/android/armeabi-v7a/libonnxruntime.so
# => ELF 32-bit LSB shared object, ARM, EABI5 ...

nm -D core/third-party/onnxruntime/lib/android/armeabi-v7a/libonnxruntime.so \
    | grep OrtGetApiBase
# => exports OrtGetApiBase
```

Keep the ONNX Runtime version in step 2 in sync with the `arm64-v8a` build so
both ABIs use the same runtime.

### 3. Ship the Silero VAD as `.ort` (not `.onnx`)

The bundled Silero VAD is an `.onnx` protobuf. ORT 1.23.2 cannot load it on
ARM32 even when the buffer is page-aligned (protobuf stores float initializers
at arbitrary sub-4-byte offsets, which ORT's ARM32 build reads with
alignment-sensitive instructions). Every `.ort` flatbuffer model loads fine, so
convert the VAD to `.ort` and re-bake it:

```sh
python -m onnxruntime.tools.convert_onnx_models_to_ort silero_vad.onnx \
    --optimization_style Fixed --output_dir out
# regenerate core/silero-vad-model-data.h from out/silero_vad.ort,
# keeping the symbol names (silero_vad_onnx / silero_vad_onnx_len)
```

`CreateSessionFromArray` auto-detects the format, so no loader changes are
needed — only the baked bytes. Declaring the baked array `alignas(64)` is good
hygiene on top of the runtime fix in step 1.

### 4. Build

```sh
./gradlew :assembleRelease
```

The output AAR now contains native libraries for both `arm64-v8a` and
`armeabi-v7a`.

## Notes

- Real `.ort` models and `*-model-data.cpp` are stored with Git LFS. In a fresh
  checkout run `git lfs install --local && git lfs pull` so they are
  materialized (otherwise they are pointer stubs).
- Speaker diarization loads an additional baked model. The alignment fix in
  step 1 covers it too; if you don't need diarization you can disable it
  (`identify_speakers=false`) to skip that model load entirely.
