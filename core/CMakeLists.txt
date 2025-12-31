cmake_minimum_required(VERSION 3.22.1)

project("moonshine")

set(CMAKE_CXX_STANDARD 20)

# Ensure the standard is strictly enforced (optional, but recommended)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable compiler-specific extensions (optional, but recommended)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic -Werror)
# Enable this to catch memory leaks and access violations.
# This is only supported on Linux.
# if(LINUX)
#     add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#     add_link_options(-fsanitize=address -fno-omit-frame-pointer)
# endif()

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer/build
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/build
)

if (ANDROID)
    add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni
        ${CMAKE_CURRENT_LIST_DIR}/../android/moonshine-jni/build
    )
endif()

add_library(moonshine SHARED
    moonshine.cpp
    moonshine-model.cpp
    voice-activity-detector.cpp
    resampler.cpp
    transcriber.cpp
)

set_target_properties(moonshine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# On macOS, set rpath to @loader_path so the library looks for dependencies
# (like libonnxruntime) in the same directory as libmoonshine.dylib
if(APPLE)
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
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/
    ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
)

if (IOS)
    target_link_libraries(moonshine PUBLIC
        bin-tokenizer
        ort-utils
        ten_vad
        "-framework CoreFoundation"
        "-framework Foundation"
    )
    set_target_properties(moonshine PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER ai.moonshine.voice
        PUBLIC_HEADER ${CMAKE_CURRENT_LIST_DIR}/moonshine.h
        RESOURCE ${CMAKE_CURRENT_LIST_DIR}/../test-assets
        VERSION 0.0.6
        COMPATIBILITY ANYNEWER
    )
else()
    target_link_libraries(moonshine PUBLIC
        bin-tokenizer
        ort-utils
        ten_vad
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

if (IOS)
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

add_executable(voice-activity-detector-test voice-activity-detector-test.cpp voice-activity-detector.cpp resampler.cpp)
set_target_properties(voice-activity-detector-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(voice-activity-detector-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/
)

if (IOS)
    set_target_properties(voice-activity-detector-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.voice-activity-detector-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
    target_link_libraries(voice-activity-detector-test PRIVATE
        ten_vad
        "-framework CoreFoundation"
        "-framework Foundation"
    )
else()
    target_link_libraries(voice-activity-detector-test PRIVATE
        ten_vad
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

if (IOS)
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
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/
    ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
    ${CMAKE_CURRENT_LIST_DIR}/third-party/doctest
)

if (IOS)
    set_target_properties(transcriber-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.transcriber-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
endif()
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
