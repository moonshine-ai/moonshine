# tts unit tests (TFLM micro_test.h). Host-runnable end to end: the synth core
# has no platform dependency. Links `tts` plus the parent-provided
# `tflm_test_support` (for MicroPrintf / InitializeTarget). The test reaches the
# private g2p.h via the module's src include path.

add_executable(tts_test tts_test.cc)
target_link_libraries(tts_test PRIVATE tts tflm_test_support)
target_include_directories(tts_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/tts)
moonshine_micro_register_micro_test(tts_test)
