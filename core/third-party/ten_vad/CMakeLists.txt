cmake_minimum_required(VERSION 3.22.1)

project("ten_vad")

add_library(ten_vad STATIC
    aed.cc
    biquad.cc
    fftw.c
    fscvrt.cc
    pitch_est.cc
    stft.cc
    ten_vad.cc
    ten-vad-model-data.c
)

target_compile_options(ten_vad PRIVATE
    -fPIC
)

target_include_directories(ten_vad PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../onnxruntime/include
    ${CMAKE_CURRENT_LIST_DIR}/../../moonshine-utils
)

if (ANDROID)
    target_link_libraries(ten_vad PUBLIC
        ort-utils
        android
        log
    )
elseif(IOS)
    target_link_libraries(ten_vad PUBLIC
        ort-utils
        "-framework CoreFoundation"
        "-framework Foundation"
    )
else()
    target_link_libraries(ten_vad PUBLIC
        ort-utils
    )
endif()

add_executable(ten_vad-test ten_vad-test.cpp)

target_include_directories(ten_vad-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../doctest
)

if (IOS OR MOONSHINE_BUILD_SWIFT)
    set_target_properties(ten_vad-test PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.moonshine.voice.ten_vad-test"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
    target_link_libraries(ten_vad-test PRIVATE
        ten_vad
        "-framework CoreFoundation"
        "-framework Foundation"
    )
else()
    target_link_libraries(ten_vad-test PUBLIC
        ten_vad
    )
endif()

