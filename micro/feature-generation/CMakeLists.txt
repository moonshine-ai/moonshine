# feature-generation -- portable log-mel front-end (batch + streaming).
#
# Depends only on `kissfft` (real FFT) and `tflm` (TFLM headers, for micro_log).
# Both are expected to be provided by the including project as CMake targets:
#   * kissfft -- add_subdirectory(third-party/kissfft) or your own copy.
#   * tflm    -- an INTERFACE target exposing the TFLM headers (so the module
#                stays agnostic about which TFLM implementation is linked).
#
# Build standalone for another platform:
#   add_subdirectory(third-party/kissfft)
#   add_library(tflm INTERFACE)                         # point at your TFLM
#   target_include_directories(tflm INTERFACE <tflm/include>)
#   add_subdirectory(feature-generation)
#   target_link_libraries(my_app PRIVATE feature_generation)

cmake_minimum_required(VERSION 3.13)

add_library(feature_generation STATIC
  src/log_mel.cc
  src/mel_streamer.cc
  src/fft_scratch.cc
)

target_include_directories(feature_generation
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(feature_generation PUBLIC kissfft tflm)

# Match the TFLM build flags so vtables / exception tables are consistent across
# the whole link on the embedded target. Harmless on host.
target_compile_features(feature_generation PUBLIC cxx_std_17)

# Optional per-module unit tests (host or target, depending on the harness the
# parent project provides). Guarded so adding the module to a firmware build
# doesn't pull test executables in unless asked.
if(MOONSHINE_MICRO_BUILD_TESTS)
  add_subdirectory(tests)
endif()
