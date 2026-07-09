# vad -- on-device voice activity detection (TinyVadCNN wrapper + segmenter).
#
# Depends on `tflm` (TFLM, for the interpreter + micro_log) and -- like every
# module here -- may use `feature_generation`. The streaming log-mel front-end
# that feeds it lives in feature-generation (MelStreamer); compose them in your
# application. No app or platform dependency.
#
# On the host (MOONSHINE_MICRO_HOST_TESTS) the interpreter wrapper (vad.cc) is omitted
# so the segmenter logic can be unit-tested without a host TFLM build.

cmake_minimum_required(VERSION 3.13)

set(VAD_SRCS src/vad_segmenter.cc)
if(NOT MOONSHINE_MICRO_HOST_TESTS)
  list(APPEND VAD_SRCS src/vad.cc)
endif()

add_library(vad STATIC ${VAD_SRCS})

target_include_directories(vad
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(vad PUBLIC tflm)
target_compile_features(vad PUBLIC cxx_std_17)

if(MOONSHINE_MICRO_BUILD_TESTS)
  add_subdirectory(tests)
endif()
