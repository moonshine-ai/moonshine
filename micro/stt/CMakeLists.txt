# stt -- on-device isolated-letter/digit classifier (SpellingCNN wrapper).
#
# Depends on `tflm` (TFLM interpreter + micro_log) and may use
# `feature_generation` (the front-end that produces its input). No app or
# platform dependency.
#
# On the host (MOONSHINE_MICRO_HOST_TESTS) the interpreter wrapper (classifier.cc) is
# omitted so the prediction helpers can be unit-tested without a host TFLM build.

cmake_minimum_required(VERSION 3.13)

set(STT_SRCS src/predictor.cc)
if(NOT MOONSHINE_MICRO_HOST_TESTS)
  list(APPEND STT_SRCS src/classifier.cc)
endif()

add_library(stt STATIC ${STT_SRCS})

target_include_directories(stt
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(stt PUBLIC tflm)
target_compile_features(stt PUBLIC cxx_std_17)

if(MOONSHINE_MICRO_BUILD_TESTS)
  add_subdirectory(tests)
endif()
