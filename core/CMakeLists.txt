cmake_minimum_required(VERSION 3.22.1)

project("moonshine")

set(CMAKE_CXX_STANDARD 20)

set(MOONSHINE_VERSION "0.0.48")

# Ensure the standard is strictly enforced (optional, but recommended)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable compiler-specific extensions (optional, but recommended)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()
# Enable this to catch memory leaks and access violations.
# This is only supported on Linux.
# if(LINUX)
#     add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#     add_link_options(-fsanitize=address -fno-omit-frame-pointer)
# endif()

include(${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/find-ort-library-path.cmake)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer/build
)

if (ANDROID)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni/build
    )
endif()

set(MOONSHINE_SRC_FILES
    moonshine-c-api.cpp
    cosine-distance.cpp
    moonshine-model.cpp
    moonshine-streaming-model.cpp
    voice-activity-detector.cpp
    silero-vad.cpp
    resampler.cpp
    transcriber.cpp
    gemma-embedding-model.cpp
    intent-recognizer.cpp
    speaker-embedding-model.cpp
    speaker-embedding-model-data.cpp
    online-clusterer.cpp
)

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
)

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
    # We need to merge: moonshine-utils, onnxruntime, ort-utils, bin-tokenizer, and moonshine
    # Note: Order matters - dependencies first, then the main library
    add_custom_command(TARGET moonshine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Merging static libraries into framework..."
        COMMAND libtool -static -o "$<TARGET_FILE:moonshine>.merged"
            "$<TARGET_FILE:moonshine-utils>"
            "${ONNXRUNTIME_LIB_PATH}"
            "$<TARGET_FILE:ort-utils>"
            "$<TARGET_FILE:bin-tokenizer>"
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
)
target_link_libraries(benchmark PRIVATE
    moonshine
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

    add_executable(speaker-embedding-model-test speaker-embedding-model-test.cpp speaker-embedding-model.cpp resampler.cpp)
    set_target_properties(speaker-embedding-model-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(speaker-embedding-model-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/ort-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )

    copy_onnxruntime_dll(speaker-embedding-model-test)

    if (IOS OR MOONSHINE_BUILD_SWIFT)
        set_target_properties(speaker-embedding-model-test PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.speaker-embedding-model-test"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        )
        target_link_libraries(speaker-embedding-model-test PRIVATE
            moonshine
            ${ONNXRUNTIME_LIB_PATH}
            "-framework CoreFoundation"
            "-framework Foundation"
        )
    else()
        target_link_libraries(speaker-embedding-model-test PRIVATE
            moonshine
            ${ONNXRUNTIME_LIB_PATH}
        )
    endif()

    add_executable(online-clusterer-test online-clusterer-test.cpp online-clusterer.cpp)
    set_target_properties(online-clusterer-test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(online-clusterer-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
        ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    )
    target_link_libraries(online-clusterer-test PRIVATE
        moonshine
    )
endif()