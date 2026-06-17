# stt unit tests (TFLM micro_test.h). Host-runnable: the prediction helpers
# only (no interpreter). Links `stt` (on host: just predictor.cc) plus the
# parent-provided `tflm_test_support`.

add_executable(predictor_test predictor_test.cc)
target_link_libraries(predictor_test PRIVATE stt tflm_test_support)
cpp_tiny_register_micro_test(predictor_test)
