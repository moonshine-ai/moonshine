# vad unit tests (TFLM micro_test.h). Host-runnable: only the segmenter (no
# interpreter). Links `vad` (which on host contains just vad_segmenter.cc) plus
# the parent-provided `tflm_test_support`.

add_executable(vad_segmenter_test vad_segmenter_test.cc)
target_link_libraries(vad_segmenter_test PRIVATE vad tflm_test_support)
moonshine_micro_register_micro_test(vad_segmenter_test)
