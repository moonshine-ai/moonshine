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
    moonshine-streaming-model.cpp
    voice-activity-detector.cpp
    resampler.cpp
    transcriber.cpp
)

set_target_properties(moonshine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(moonshine PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/moonshine-utils
    ${CMAKE_CURRENT_LIST_DIR}/ort-utils
    ${CMAKE_CURRENT_LIST_DIR}/bin-tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/third-party/onnxruntime/include
    ${CMAKE_CURRENT_LIST_DIR}/third-party/ten_vad/
    ${CMAKE_CURRENT_LIST_DIR}/third-party/utf-8/
)

target_link_libraries(moonshine PUBLIC
    bin-tokenizer
    ort-utils
    ten_vad
)

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

target_link_libraries(voice-activity-detector-test PRIVATE
    ten_vad
)

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

target_link_libraries(resampler-test PRIVATE
    moonshine-utils
)

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

target_link_libraries(transcriber-test PRIVATE
    moonshine
)
