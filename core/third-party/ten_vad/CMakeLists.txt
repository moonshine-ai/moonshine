cmake_minimum_required(VERSION 3.22.1)

project("ten_vad")

add_library(ten_vad SHARED
    aed.cc
    biquad.cc
    fftw.c
    fscvrt.cc
    pitch_est.cc
    stft.cc
    ten_vad.cc
    ten-vad-model-data.c
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

target_link_libraries(ten_vad-test PUBLIC
    ten_vad
)