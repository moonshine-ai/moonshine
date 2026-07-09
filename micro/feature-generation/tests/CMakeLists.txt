# feature-generation unit tests (TFLM micro_test.h).
#
# The test links the module plus `tflm_test_support`, a target the including
# project defines to satisfy micro_test.h's MicroPrintf / InitializeTarget:
#   * host  -> the small desktop stub (test-support/host/tflm_host_stub.cc)
#   * target-> the real pico-tflmicro logging + system_setup

add_executable(feature_generation_test feature_generation_test.cc)
target_link_libraries(feature_generation_test
  PRIVATE feature_generation kissfft tflm_test_support)
moonshine_micro_register_micro_test(feature_generation_test)
