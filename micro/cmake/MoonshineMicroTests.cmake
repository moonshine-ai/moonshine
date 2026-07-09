# Helper for registering TFLM micro_test.h unit tests with CTest.
#
# The micro_test.h harness loops forever, so we run each binary through
# test-support/run_micro_test.sh, which reads until the pass/fail banner and maps
# it to an exit code. Module test CMakeLists call moonshine_micro_register_micro_test()
# after creating their test executable.

find_program(BASH_PROGRAM bash)
set(MOONSHINE_MICRO_RUN_MICRO_TEST
    "${CMAKE_CURRENT_LIST_DIR}/../test-support/run_micro_test.sh"
    CACHE INTERNAL "micro_test runner wrapper")

function(moonshine_micro_register_micro_test target)
  if(BASH_PROGRAM)
    add_test(NAME ${target}
             COMMAND ${BASH_PROGRAM} ${MOONSHINE_MICRO_RUN_MICRO_TEST}
                     $<TARGET_FILE:${target}>)
  else()
    # No bash: register the raw binary (useful on a target where the harness's
    # serial output is read externally rather than by CTest).
    add_test(NAME ${target} COMMAND ${target})
  endif()
endfunction()
