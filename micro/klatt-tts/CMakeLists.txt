# Portable formant (Klatt-style) text-to-speech synth core.

cmake_minimum_required(VERSION 3.13)

add_library(tts STATIC
  src/phonemes.cc
  src/klatt.cc
  src/synth_internal.cc
  src/synth_stream.cc
  src/config.cc
)

target_include_directories(tts
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/tts
          ${CMAKE_CURRENT_SOURCE_DIR}/src)

# The grapheme-to-phoneme front end is the shared g2p library (also used by
# neural-tts); synth_stream.h exposes g2p::Lexicon, so the dependency is
# PUBLIC.
target_link_libraries(tts PUBLIC g2p)

target_compile_features(tts PUBLIC cxx_std_17)

if(MOONSHINE_MICRO_BUILD_TESTS)
  add_subdirectory(tests)
endif()
