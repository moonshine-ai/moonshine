# Neural TTS: the black-box text-to-speech module (text in, 16 kHz PCM
# chunks out; see include/neural_tts/neural_tts.h). Pipeline: shared g2p
# front end -> Klatt rule durations (klatt-tts/synth_internal) -> diphone/word
# unit selection from the flash pack -> RVQ decode through the s16x8 TFLM
# graph -> WORLD-lite vocoder.
#
# worldlite_synth depends only on kissfft (host-testable); pb_decoder and
# the full neural_tts pipeline need the TFLM interpreter and are excluded
# from host-test builds, matching the other modules' interpreter split.

cmake_minimum_required(VERSION 3.13)

set(NEURAL_TTS_SOURCES src/worldlite_synth.cc)
if(NOT MOONSHINE_MICRO_BUILD_TESTS)
  list(APPEND NEURAL_TTS_SOURCES
    src/pb_decoder.cc
    src/neural_tts.cc
    src/hooks.cc
  )
endif()

add_library(neural_tts STATIC ${NEURAL_TTS_SOURCES})

target_include_directories(neural_tts
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(neural_tts PUBLIC cxx_std_17)
target_link_libraries(neural_tts PUBLIC kissfft)

if(NOT MOONSHINE_MICRO_BUILD_TESTS)
  # g2p: text front end; klatt-tts: shared segment/duration rule engine.
  target_link_libraries(neural_tts PUBLIC pico-tflmicro pico_time g2p tts)

  # Device build: enable the on-target paths guarded by PICO_BUILD in the engine
  # sources -- time_us_64() latency stats, the tts_checkpoint watchdog heartbeat
  # (fed before every TFLM op / synth pulse), and the crash-surviving RAM trace
  # ring. These were compiled out after this module was split into its own
  # library (they had relied on a global define), which silently zeroed the TIME
  # stats and left the watchdog unfed during Invoke. The host CLI/test builds use
  # a separate CMake and deliberately leave PICO_BUILD undefined.
  target_compile_definitions(neural_tts PRIVATE PICO_BUILD=1)
endif()

if(MOONSHINE_MICRO_BUILD_TESTS)
  # Host validation harness: raw [T,61] float32 controls on stdin ->
  # raw int16 16 kHz PCM on stdout. Driven by scripts/test_worldlite_c.py
  # to compare against pyworld.
  add_executable(worldlite_synth_cli host/worldlite_synth_cli.cc)
  target_link_libraries(worldlite_synth_cli neural_tts)
endif()
