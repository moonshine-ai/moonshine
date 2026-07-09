# Portable formant (Klatt-style) text-to-speech synth core.

cmake_minimum_required(VERSION 3.13)

add_library(tts STATIC
  src/phonemes.cc
  src/g2p.cc
  src/g2p_rules.cc
  src/g2p_numbers.cc
  src/g2p_dict.cc
  src/klatt.cc
  src/synth_internal.cc
  src/synth_stream.cc
  src/config.cc
)

target_include_directories(tts
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/tts
          ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(tts PUBLIC cxx_std_17)

if(MOONSHINE_MICRO_BUILD_TESTS)
  add_subdirectory(tests)
endif()
