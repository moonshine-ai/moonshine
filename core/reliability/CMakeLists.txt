# Fuzzing harnesses for individual core modules.
#
# This directory is only configured when MOONSHINE_RELIABILITY=ON (see the
# parent core/CMakeLists.txt). Nothing here is part of the shipped library.
#
# Each target links a single core module directly so that a crash is isolated
# to one component (rather than being buried in a full integration build), and
# is built with libFuzzer coverage instrumentation (-fsanitize=fuzzer) on top of
# the ASan/UBSan flags inherited from the parent build.

set(CORE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Helper: declare a libFuzzer target. First argument is the target name, the
# rest are source files (the harness plus the module sources under test).
function(moonshine_add_fuzzer name)
    add_executable(${name} ${ARGN})
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    # libFuzzer provides its own main(); apply it only to fuzz targets so it does
    # not collide with the doctest / benchmark mains elsewhere in the build.
    target_compile_options(${name} PRIVATE -fsanitize=fuzzer)
    target_link_options(${name} PRIVATE -fsanitize=fuzzer)
endfunction()

# bin-tokenizer: exercises the in-memory constructor (untrusted blob parsing)
# and the text<->token round trip.
moonshine_add_fuzzer(fuzz_bin_tokenizer
    fuzz-bin-tokenizer.cpp
    ${CORE_DIR}/bin-tokenizer/bin-tokenizer.cpp
    ${CORE_DIR}/moonshine-utils/string-utils.cpp
    ${CORE_DIR}/moonshine-utils/debug-utils.cpp
)
target_include_directories(fuzz_bin_tokenizer PRIVATE
    ${CORE_DIR}/bin-tokenizer
    ${CORE_DIR}/moonshine-utils
)

# WAV parsing: the first-party 16-bit PCM reader plus the vendored header-only
# reader used by diarization. Both parse untrusted file bytes.
moonshine_add_fuzzer(fuzz_wav_pcm
    fuzz-wav-pcm.cpp
    ${CORE_DIR}/moonshine-utils/debug-utils.cpp
    ${CORE_DIR}/moonshine-utils/string-utils.cpp
)
target_include_directories(fuzz_wav_pcm PRIVATE
    ${CORE_DIR}/moonshine-utils
    ${CORE_DIR}/cpp-annote/src
)
# The vendored wav_pcm_float32.h is not held to the project warning settings.
target_compile_options(fuzz_wav_pcm PRIVATE -Wno-error)

# Resampler: linear up/down sampling maths.
moonshine_add_fuzzer(fuzz_resampler
    fuzz-resampler.cpp
    ${CORE_DIR}/resampler.cpp
    ${CORE_DIR}/moonshine-utils/debug-utils.cpp
    ${CORE_DIR}/moonshine-utils/string-utils.cpp
)
target_include_directories(fuzz_resampler PRIVATE
    ${CORE_DIR}
    ${CORE_DIR}/moonshine-utils
)

# String utilities: trimming, splitting, and typed parsers.
moonshine_add_fuzzer(fuzz_string_utils
    fuzz-string-utils.cpp
    ${CORE_DIR}/moonshine-utils/string-utils.cpp
)
target_include_directories(fuzz_string_utils PRIVATE
    ${CORE_DIR}/moonshine-utils
)

# Tensor view: shape/dtype/data construction turns an attacker-influenced shape
# into a size computation, an allocation, and a memcpy. This module is coupled
# to onnxruntime, so rather than list its sources we link the already-built
# ort-utils static library (which pulls in onnxruntime + moonshine-utils); the
# harness runs under the inherited ASan/UBSan flags.
moonshine_add_fuzzer(fuzz_tensor_view
    fuzz-tensor-view.cpp
)
target_include_directories(fuzz_tensor_view PRIVATE
    ${CORE_DIR}/ort-utils
    ${CORE_DIR}/moonshine-utils
    ${CORE_DIR}/third-party/onnxruntime/include
)
target_link_libraries(fuzz_tensor_view PRIVATE
    ort-utils
)
