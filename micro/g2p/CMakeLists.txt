# English grapheme-to-phoneme front end, shared by the Klatt formant TTS
# (klatt-tts/) and the neural diphone TTS (neural-tts/). Text in, base-phone
# tokens out; no synthesis code lives here.

cmake_minimum_required(VERSION 3.13)

add_library(g2p STATIC
  src/g2p.cc
  src/g2p_rules.cc
  src/g2p_numbers.cc
  src/g2p_dict.cc
  src/g2p_phones.cc
  src/ipa_tokens.cc
)

target_include_directories(g2p
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/g2p
          ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(g2p PUBLIC cxx_std_17)
