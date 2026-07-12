cmake_minimum_required(VERSION 3.22.1)

project("moonshine")

enable_testing()

set(CMAKE_CXX_STANDARD 20)

set(MOONSHINE_VERSION "0.0.68")

# Ensure the standard is strictly enforced (optional, but recommended)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable compiler-specific extensions (optional, but recommended)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# Opt-in reliability build (Linux/Clang): a sanitizer build plus a compile-
# commands database for clang-tidy, and (for the address build) the fuzzing
# harnesses under reliability/. This is OFF by default and is never used for
# release builds, so production binaries, their performance, and the set of
# runtime dependencies shipped to library users are all unaffected.
# See core/STYLE_GUIDE.md and scripts/reliability.sh.
option(MOONSHINE_RELIABILITY
    "Build with sanitizers and fuzzing support (Linux/Clang only)" OFF)
# Which sanitizer(s) to build with. "address,undefined" (default) is the ASan +
# UBSan build that also carries the libFuzzer harnesses; "thread" is a separate
# ThreadSanitizer build for the data-race stage (TSan cannot be combined with
# ASan, so scripts/reliability-remote.sh builds it into its own directory).
set(MOONSHINE_SANITIZER "address,undefined" CACHE STRING
    "Sanitizer set for MOONSHINE_RELIABILITY builds (address,undefined|thread)")
if (MOONSHINE_RELIABILITY)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    add_compile_options(
        -fsanitize=${MOONSHINE_SANITIZER}
        -fsanitize-ignorelist=${CMAKE_CURRENT_LIST_DIR}/reliability/sanitizer-ignorelist.txt
        -fno-omit-frame-pointer
        -fno-sanitize-recover=all
        -g
        # Lightweight libstdc++ precondition checks (bounds on vector/string
        # operator[], .front()/.back(), invalid iterator ranges, etc.). These
        # catch logic out-of-bounds that ASan misses when the bad index still
        # lands inside the same heap allocation. Unlike _GLIBCXX_DEBUG this does
        # not change container ABI, so it coexists with the prebuilt onnxruntime.
        -D_GLIBCXX_ASSERTIONS
        # The vendored doctest header pulls in <ciso646>, which very recent
        # libstdc++ versions flag with a deprecation "#warning". We must not edit
        # third-party code, so keep -Werror everywhere except for that directive
        # class; real first-party warnings still fail the build. (Quoted so CMake
        # does not treat the leading '#' as a comment.)
        "-Wno-error=#warnings"
    )
    add_link_options(-fsanitize=${MOONSHINE_SANITIZER})
endif()

include(${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/find-ort-library-path.cmake)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-tts
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-tts/build
)

if (ANDROID)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni/build
    )
endif()

set(MOONSHINE_SRC_FILES
    moonshine-c-api.cpp
    moonshine-model-catalog.cpp
    cosine-distance.cpp
    moonshine-model.cpp
    moonshine-streaming-model.cpp
    voice-activity-detector.cpp
    silero-vad.cpp
    resampler.cpp
    transcriber.cpp
    gemma-embedding-model.cpp
    intent-recognizer.cpp
    speaker-diarizer.cpp
    spelling-fusion.cpp
    spelling-fusion-data.cpp
    spelling-model.cpp
    word-alignment.cpp
)

# cpp-annote speaker diarization library (vendored, see cpp-annote/README.md).
# Its bundled dependencies (Eigen, kaldi-native-fbank, kissfft) live under
# third-party/ alongside the other vendored libraries. Compiled directly into
# the moonshine target so that static framework/library merging on iOS and
# Windows picks it up without extra archives.
set(CPP_ANNOTE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp-annote)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/third-party)
set(CPP_ANNOTE_SRC_FILES
    ${CPP_ANNOTE_DIR}/src/cpp-annote.cpp
    ${CPP_ANNOTE_DIR}/src/cpp-annote-streaming.cpp
    ${CPP_ANNOTE_DIR}/src/community1_cpp_annote_embedded.cpp
    ${CPP_ANNOTE_DIR}/src/community1_ort_embedded.cpp
    ${CPP_ANNOTE_DIR}/src/compute_fbank.cpp
    ${CPP_ANNOTE_DIR}/src/embedding_ort_infer.cpp
    ${CPP_ANNOTE_DIR}/src/filter_train.cpp
    ${CPP_ANNOTE_DIR}/src/clustering_vbx.cpp
    ${CPP_ANNOTE_DIR}/src/plda_vbx.cpp
    ${CPP_ANNOTE_DIR}/src/scipy_linkage.cpp
    ${CPP_ANNOTE_DIR}/src/parity_log.cpp
    ${THIRD_PARTY_DIR}/kissfft/kiss_fft.c
    ${THIRD_PARTY_DIR}/kissfft/kiss_fftr.c
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/feature-fbank.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/feature-functions.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/feature-mfcc.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/feature-raw-audio-samples.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/feature-window.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/istft.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/kaldi-math.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/mel-computations.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/online-feature.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/rfft.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/stft.cc
    ${THIRD_PARTY_DIR}/kaldi-native-fbank/csrc/whisper-feature.cc
)
# Vendored code isn't held to the project's warning settings, and Eigen is
# restricted to its MPL2-licensed subset.
if (WIN32)
    set_source_files_properties(${CPP_ANNOTE_SRC_FILES} PROPERTIES
        COMPILE_OPTIONS "/w"
        COMPILE_DEFINITIONS "EIGEN_MPL2_ONLY=1"
    )
else()
    set_source_files_properties(${CPP_ANNOTE_SRC_FILES} PROPERTIES
        COMPILE_OPTIONS "-w"
        COMPILE_DEFINITIONS "EIGEN_MPL2_ONLY=1"
    )
endif()
list(APPEND MOONSHINE_SRC_FILES ${CPP_ANNOTE_SRC_FILES})

# speaker-diarizer.cpp is first-party and must keep the project's warning
# settings, but it includes the vendored Eigen headers. On ARM/NEON, GCC's
# -Wclass-memaccess fires inside Eigen's PacketMath.h (memcpy of the non-trivial
# Packet4c wrapper), which -Werror then turns into a build failure. Since we
# must not edit third-party code, downgrade only that one diagnostic from an
# error for this file, and only for GCC (Clang/MSVC don't define the flag). All
# other warnings, including first-party class-memaccess misuse, still fail.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(speaker-diarizer.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-error=class-memaccess"
    )
endif()
# Same story on 32-bit ARM (armeabi-v7a): Eigen's NEON GeneralBlockPanelKernel.h
# has an unused 'tmp' parameter that Clang flags with -Wunused-parameter, which
# -Werror turns into a build failure when speaker-diarizer.cpp pulls it in. This
# path is only compiled for armeabi-v7a (arm64-v8a's NEON kernels don't trip it),
# so downgrade just that diagnostic for that ABI. We must not edit third-party
# code, and all other warnings still fail the build.
if (ANDROID AND CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    set_source_files_properties(speaker-diarizer.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-error=unused-parameter"
    )
endif()

# Build as STATIC framework for iOS/Swift (shared frameworks cannot be embedded)
# Build as SHARED library for other platforms
if (NOT MOONSHINE_BUILD_SHARED AND (IOS OR MOONSHINE_BUILD_SWIFT OR WIN32))
    add_library(moonshine STATIC
        ${MOONSHINE_SRC_FILES}
    )
else()
    add_library(moonshine SHARED
        ${MOONSHINE_SRC_FILES}
    )
endif()

set_target_properties(moonshine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# On macOS (non-framework builds), set rpath to @loader_path so the library looks for dependencies
# (like libonnxruntime) in the same directory as libmoonshine.dylib
# Note: Static frameworks don't need rpath since they're linked statically
if(APPLE AND NOT IOS AND NOT MOONSHINE_BUILD_SWIFT)
    set_target_properties(moonshine PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

target_include_directories(moonshine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
    ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
    ${CPP_ANNOTE_DIR}/src
    ${THIRD_PARTY_DIR}
    ${THIRD_PARTY_DIR}/kissfft
)

# Extra static archives merged into moonshine.framework (iOS / Swift xcframework).
set(_MOONSHINE_FRAMEWORK_MERGE_TTS_ARCHIVES "")
if(MOONSHINE_TTS_BUILD_ONNX)
  list(APPEND _MOONSHINE_FRAMEWORK_MERGE_TTS_ARCHIVES
    "$<TARGET_FILE:moonshine_tts_utf8proc>"
    "$<TARGET_FILE:moonshine_tts_core>"
    "$<TARGET_FILE:moonshine_tts_ort>"
  )
endif()

if (IOS OR MOONSHINE_BUILD_SWIFT)
    target_link_libraries(moonshine PUBLIC
        bin-tokenizer
        ort-utils
        "-framework CoreFoundation"
        "-framework Foundation"
    )
    set_target_properties(moonshine PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER ai.moonshine.voice
        PUBLIC_HEADER ${CMAKE_CURRENT_LIST_DIR}/moonshine-c-api.h
        RESOURCE ${CMAKE_CURRENT_LIST_DIR}/../test-assets
        VERSION ${MOONSHINE_VERSION}
        COMPATIBILITY ANYNEWER
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "ai.moonshine.voice"
        XCODE_ATTRIBUTE_PRODUCT_NAME "moonshine"
        XCODE_ATTRIBUTE_MARKETING_VERSION ${MOONSHINE_VERSION}
        XCODE_ATTRIBUTE_CURRENT_PROJECT_VERSION ${MOONSHINE_VERSION}
    )
    # Copy the public header into the framework's Headers directory after build
    # This is needed because Xcode generator doesn't automatically copy PUBLIC_HEADER during build
    add_custom_command(TARGET moonshine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:moonshine>/Headers"
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_LIST_DIR}/moonshine-c-api.h
            "$<TARGET_FILE_DIR:moonshine>/Headers/moonshine-c-api.h"
        COMMENT "Copying public header to framework"
    )
    # For static frameworks, merge all static dependencies into a single library
    # This is required because static libraries don't automatically include their dependencies
    # We merge: moonshine-utils, onnxruntime, ort-utils, bin-tokenizer, optional moonshine-tts
    # archives, then moonshine. Order matters — dependencies first, then the main library.
    add_custom_command(TARGET moonshine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Merging static libraries into framework..."
        COMMAND libtool -static -o "$<TARGET_FILE:moonshine>.merged"
            "$<TARGET_FILE:moonshine-utils>"
            "${ONNXRUNTIME_LIB_PATH}"
            "$<TARGET_FILE:ort-utils>"
            "$<TARGET_FILE:bin-tokenizer>"
            ${_MOONSHINE_FRAMEWORK_MERGE_TTS_ARCHIVES}
            "$<TARGET_FILE:moonshine>"
        COMMAND ${CMAKE_COMMAND} -E remove "$<TARGET_FILE:moonshine>"
        COMMAND ${CMAKE_COMMAND} -E rename "$<TARGET_FILE:moonshine>.merged" "$<TARGET_FILE:moonshine>"
        COMMENT "Merging static dependencies into framework binary"
    )
    # Fix Info.plist by adding required iOS framework keys
    # iOS frameworks require CFBundleName, CFBundleVersion, and CFBundleShortVersionString
    # Remove legacy CFBundleSignature key (not required and can cause installation issues)
    # Convert to binary format after modification (iOS requires binary plist format)
    # Re-sign the framework after modifying Info.plist to ensure it's properly bound
    add_custom_command(TARGET moonshine POST_BUILD
        COMMAND /usr/libexec/PlistBuddy -c "Set :CFBundleName moonshine" "$<TARGET_FILE_DIR:moonshine>/Info.plist" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :CFBundleName string moonshine" "$<TARGET_FILE_DIR:moonshine>/Info.plist"
        COMMAND /usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${MOONSHINE_VERSION}" "$<TARGET_FILE_DIR:moonshine>/Info.plist" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string ${MOONSHINE_VERSION}" "$<TARGET_FILE_DIR:moonshine>/Info.plist"
        COMMAND /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${MOONSHINE_VERSION}" "$<TARGET_FILE_DIR:moonshine>/Info.plist" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string ${MOONSHINE_VERSION}" "$<TARGET_FILE_DIR:moonshine>/Info.plist"
        COMMAND /usr/libexec/PlistBuddy -c "Delete :CFBundleSignature" "$<TARGET_FILE_DIR:moonshine>/Info.plist" 2>/dev/null || true
        COMMAND /usr/bin/plutil -convert binary1 "$<TARGET_FILE_DIR:moonshine>/Info.plist"
        COMMAND codesign --force --sign - "$<TARGET_FILE_DIR:moonshine>" || true
        COMMENT "Adding required Info.plist keys for iOS framework, removing legacy keys, converting to binary format, and re-signing"
    )
else()
    target_link_libraries(moonshine PUBLIC
        bin-tokenizer
        ort-utils
    )
endif()

if(MOONSHINE_TTS_BUILD_ONNX)
    target_link_libraries(moonshine PRIVATE moonshine_tts_ort)
    target_include_directories(moonshine PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-tts/src
    )
    target_compile_definitions(moonshine PRIVATE MOONSHINE_C_API_MOONSHINE_TTS=1)
    moonshine_tts_set_ort_rpath(moonshine)
    if(WIN32)
        copy_onnxruntime_dll(moonshine)
    endif()
    # Windows static lib: merge TTS archives into moonshine.lib so release consumers
    # only need moonshine.lib plus the existing public dependency libs.
    if(WIN32 AND NOT MOONSHINE_BUILD_SHARED)
        add_custom_command(TARGET moonshine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Merging TTS static libraries into moonshine.lib..."
            COMMAND lib /OUT:"$<TARGET_FILE:moonshine>.merged"
                "$<TARGET_FILE:moonshine_tts_utf8proc>"
                "$<TARGET_FILE:moonshine_tts_core>"
                "$<TARGET_FILE:moonshine_tts_ort>"
                "$<TARGET_FILE:moonshine>"
            COMMAND ${CMAKE_COMMAND} -E remove "$<TARGET_FILE:moonshine>"
            COMMAND ${CMAKE_COMMAND} -E rename "$<TARGET_FILE:moonshine>.merged" "$<TARGET_FILE:moonshine>"
            COMMENT "Merging TTS archives into moonshine.lib"
        )
    endif()
else()
    target_compile_definitions(moonshine PRIVATE MOONSHINE_C_API_MOONSHINE_TTS=0)
endif()

# macOS shared lib: replace linker ad-hoc signature with a full codesign pass. Linker-signed
# libmoonshine.dylib can trigger AMFI "CODESIGNING Invalid Page" when dlopen'd from e.g. Homebrew Python.
if(APPLE AND NOT IOS AND NOT MOONSHINE_BUILD_SWIFT)
    add_custom_command(TARGET moonshine POST_BUILD
        COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:moonshine>"
        COMMENT "Ad-hoc codesign libmoonshine.dylib for stable dlopen on macOS"
    )
endif()

add_executable(moonshine-c-api-test moonshine-c-api-test.cpp)
set_target_properties(moonshine-c-api-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(moonshine-c-api-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
)

if (IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(moonshine-c-api-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.moonshine-c-api-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()

target_link_libraries(moonshine-c-api-test PRIVATE
    moonshine
)

add_executable(moonshine-c-api-memory-test moonshine-c-api-memory-test.cpp)
set_target_properties(moonshine-c-api-memory-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(moonshine-c-api-memory-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
)
if (IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(moonshine-c-api-memory-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.moonshine-c-api-memory-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()
target_link_libraries(moonshine-c-api-memory-test PRIVATE
    moonshine
)

# Download smoke tool: resolves model download manifests via the C API and
# loads+runs each engine from a directory of freshly downloaded files. Driven by
# scripts/test-model-downloads.sh (network-dependent, so it is NOT part of the
# hermetic scripts/test-core.sh run). Links moonshine-utils for load_wav_data.
add_executable(moonshine-download-smoke moonshine-download-smoke.cpp)
set_target_properties(moonshine-download-smoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(moonshine-download-smoke PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party
)
if (IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(moonshine-download-smoke PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.moonshine-download-smoke"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()
target_link_libraries(moonshine-download-smoke PRIVATE
    moonshine
    moonshine-utils
)

# Repeated-use TTS memory-growth regression test (Kokoro / Piper / ZipVoice).
# Uses only the public C API, so it links just libmoonshine like the other
# memory tests. Driven from scripts/reliability-remote.sh and scripts/test-core.sh.
add_executable(tts-repeated-memory-test tts-repeated-memory-test.cpp)
set_target_properties(tts-repeated-memory-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(tts-repeated-memory-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
)
if (IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(tts-repeated-memory-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.tts-repeated-memory-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()
target_link_libraries(tts-repeated-memory-test PRIVATE
    moonshine
)

# Make this test C++11 to ensure our API is backwards compatible, even
# though the internal implementation is C++20.
add_executable(moonshine-cpp-test moonshine-cpp-test.cpp)
set_target_properties(moonshine-cpp-test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(moonshine-cpp-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
)
target_link_libraries(moonshine-cpp-test PRIVATE
    moonshine
    moonshine-utils
)

add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
)
target_link_libraries(benchmark PRIVATE
    moonshine
    moonshine-utils
)

# Windows DLLs don't export all symbols by default and so tests that rely on them
# are skipped for shared builds.
if (NOT WIN32 OR NOT MOONSHINE_BUILD_SHARED)
    add_executable(voice-activity-detector-test voice-activity-detector-test.cpp voice-activity-detector.cpp resampler.cpp silero-vad.cpp)
    set_target_properties(voice-activity-detector-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(voice-activity-detector-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
    )

    copy_onnxruntime_dll(voice-activity-detector-test)

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(voice-activity-detector-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.voice-activity-detector-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(voice-activity-detector-test PRIVATE
            ort-utils
            ${ONNXRUNTIME_LIB_PATH}
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(voice-activity-detector-test PRIVATE
            ort-utils
            ${ONNXRUNTIME_LIB_PATH}
        )
    endif()

    add_executable(resampler-test resampler-test.cpp resampler.cpp)
    set_target_properties(resampler-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(resampler-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(resampler-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.resampler-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(resampler-test PRIVATE
            moonshine-utils
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(resampler-test PRIVATE
            moonshine-utils
        )
    endif()

    add_executable(cosine-distance-test cosine-distance-test.cpp)
    set_target_properties(cosine-distance-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(cosine-distance-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(cosine-distance-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.cosine-distance-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(cosine-distance-test PRIVATE
            moonshine
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(cosine-distance-test PRIVATE
            moonshine
        )
    endif()

    add_executable(spelling-fusion-test spelling-fusion-test.cpp spelling-fusion.cpp spelling-fusion-data.cpp)
    set_target_properties(spelling-fusion-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(spelling-fusion-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )
    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(spelling-fusion-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.spelling-fusion-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
    endif()

    add_executable(transcriber-test transcriber-test.cpp transcriber.cpp voice-activity-detector.cpp resampler.cpp)
    set_target_properties(transcriber-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(transcriber-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(transcriber-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.transcriber-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(transcriber-test PRIVATE
            moonshine
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(transcriber-test PRIVATE
            moonshine
        )
    endif()

    add_executable(transcriber-streaming-memory-test
        transcriber-streaming-memory-test.cpp
        transcriber.cpp voice-activity-detector.cpp resampler.cpp)
    set_target_properties(transcriber-streaming-memory-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(transcriber-streaming-memory-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )
    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(transcriber-streaming-memory-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.transcriber-streaming-memory-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(transcriber-streaming-memory-test PRIVATE
            moonshine
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(transcriber-streaming-memory-test PRIVATE
            moonshine
        )
    endif()

    # Multi-threaded stress test for the transcriber's thread-safe API. Built
    # only for the opt-in reliability build (it exists to give ThreadSanitizer
    # real cross-thread contention to analyse), so normal builds are unchanged.
    if (MOONSHINE_RELIABILITY)
        find_package(Threads REQUIRED)
        add_executable(transcriber-concurrency-test
            transcriber-concurrency-test.cpp
            transcriber.cpp voice-activity-detector.cpp resampler.cpp)
        set_target_properties(transcriber-concurrency-test PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO
        )
        target_include_directories(transcriber-concurrency-test PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
            ${CMAKE_CURRENT_LIST_DIR}/ort-utils
            ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
            ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
            ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
            ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
        )
        target_link_libraries(transcriber-concurrency-test PRIVATE
            moonshine
            Threads::Threads
        )
    endif()

    add_executable(word-alignment-test word-alignment-test.cpp)
    set_target_properties(word-alignment-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(word-alignment-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )
    target_link_libraries(word-alignment-test PRIVATE moonshine)

    add_executable(word-alignment-benchmark word-alignment-benchmark.cpp)
    set_target_properties(word-alignment-benchmark PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(word-alignment-benchmark PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
    )
    target_link_libraries(word-alignment-benchmark PRIVATE moonshine moonshine-utils)

    add_executable(intent-recognizer-test intent-recognizer-test.cpp intent-recognizer.cpp gemma-embedding-model.cpp)
    set_target_properties(intent-recognizer-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(intent-recognizer-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    copy_onnxruntime_dll(intent-recognizer-test)

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(intent-recognizer-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.intent-recognizer-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(intent-recognizer-test PRIVATE
            moonshine
            bin-tokenizer
            ${ONNXRUNTIME_LIB_PATH}
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(intent-recognizer-test PRIVATE
            moonshine
            bin-tokenizer
            ${ONNXRUNTIME_LIB_PATH}
        )
    endif()

    add_executable(gemma-embedding-model-test gemma-embedding-model-test.cpp gemma-embedding-model.cpp)
    set_target_properties(gemma-embedding-model-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(gemma-embedding-model-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    copy_onnxruntime_dll(gemma-embedding-model-test)

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(gemma-embedding-model-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.gemma-embedding-model-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(gemma-embedding-model-test PRIVATE
            bin-tokenizer
            ort-utils
            ${ONNXRUNTIME_LIB_PATH}
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(gemma-embedding-model-test PRIVATE
            bin-tokenizer
            ort-utils
            ${ONNXRUNTIME_LIB_PATH}
        )
    endif()

    add_executable(spelling-model-test spelling-model-test.cpp spelling-model.cpp spelling-fusion.cpp spelling-fusion-data.cpp)
    set_target_properties(spelling-model-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(spelling-model-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    copy_onnxruntime_dll(spelling-model-test)

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(spelling-model-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.spelling-model-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(spelling-model-test PRIVATE
            moonshine
            ${ONNXRUNTIME_LIB_PATH}
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(spelling-model-test PRIVATE
            moonshine
            ${ONNXRUNTIME_LIB_PATH}
        )
    endif()

endif()

# Style enforcement: banned constructs must not creep back into first-party
# code. Registered as a ctest so `ctest -R check-banned-constructs` (and the
# reliability script) fail if a banned pattern appears. See core/STYLE_GUIDE.md.
add_test(
    NAME check-banned-constructs
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/../scripts/check-banned-constructs.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)

# Fuzzing harnesses are only configured for the opt-in address/UBSan reliability
# build. libFuzzer's runtime is built for the ASan-style build; the ThreadSanitizer
# build (MOONSHINE_SANITIZER=thread) skips them entirely.
if (MOONSHINE_RELIABILITY AND MOONSHINE_SANITIZER MATCHES "address")
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/reliability
        ${CMAKE_CURRENT_LIST_DIR}/reliability/build
    )
endif()