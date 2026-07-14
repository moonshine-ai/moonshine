# Native (desktop) build of the on-device neural TTS engine.
#
# Builds the exact C++ pipeline the RP2350 firmware runs (neural_tts +
# pb_decoder + worldlite_synth + g2p + klatt-tts + kissfft) against the TFLM
# *reference* kernels instead of CMSIS-NN, so it compiles and runs on a laptop.
# The three ops the decoder needs that pico-tflmicro ships only as CMSIS-NN
# (CONV_2D, ADD, TRANSPOSE_CONV) get portable reference kernels from tflm_ref/.
#
# Standalone project -- configure it directly, no Pico SDK required:
#   cmake -B build-host-tts -S moonshine-micro/neural-tts/host
#   cmake --build build-host-tts -j
#   ./build-host-tts/tts_cli "hello there" \
#       --pack moonshine-micro/examples/rp2350/generated/neural_tts_pack.bin \
#       -o /tmp/hello.wav

cmake_minimum_required(VERSION 3.16)
project(neural_tts_host CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# host/ -> neural-tts -> moonshine-micro
get_filename_component(MM "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(TFLM "${MM}/third-party/pico-tflmicro/src")
set(NTTS "${MM}/neural-tts")

# ---------------------------------------------------------------------------
# tflm_host: TFLM interpreter + reference kernels, no CMSIS-NN, no Pico SDK.
# ---------------------------------------------------------------------------
set(TFLM_SOURCES
  # core interpreter / allocator / logging
  ${TFLM}/tensorflow/lite/micro/flatbuffer_utils.cpp
  ${TFLM}/tensorflow/lite/micro/memory_helpers.cpp
  ${TFLM}/tensorflow/lite/micro/micro_allocation_info.cpp
  ${TFLM}/tensorflow/lite/micro/micro_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/micro_context.cpp
  ${TFLM}/tensorflow/lite/micro/micro_interpreter.cpp
  ${TFLM}/tensorflow/lite/micro/micro_interpreter_context.cpp
  ${TFLM}/tensorflow/lite/micro/micro_interpreter_graph.cpp
  ${TFLM}/tensorflow/lite/micro/micro_log.cpp
  ${TFLM}/tensorflow/lite/micro/micro_op_resolver.cpp
  ${TFLM}/tensorflow/lite/micro/micro_profiler.cpp
  ${TFLM}/tensorflow/lite/micro/micro_resource_variable.cpp
  ${TFLM}/tensorflow/lite/micro/micro_utils.cpp
  ${TFLM}/tensorflow/lite/micro/recording_micro_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cpp
  ${TFLM}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp
  ${TFLM}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp
  ${TFLM}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cpp
  # flatbuffer + c api
  ${TFLM}/tensorflow/lite/core/api/flatbuffer_conversions.cpp
  ${TFLM}/tensorflow/lite/core/api/tensor_utils.cpp
  ${TFLM}/tensorflow/lite/core/c/common.cpp
  # tflite<->micro bridge + error reporter + schema helpers
  ${TFLM}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cpp
  ${TFLM}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cpp
  ${TFLM}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cpp
  ${TFLM}/tensorflow/compiler/mlir/lite/schema/schema_utils.cpp
  ${TFLM}/tensorflow/lite/kernels/kernel_util.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/common.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/portable_tensor_utils.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/quantization_util.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/runtime_shape.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/tensor_ctypes.cpp
  ${TFLM}/tensorflow/lite/kernels/internal/tensor_utils.cpp
  # micro kernel support + the ops the decoder graph registers
  ${TFLM}/tensorflow/lite/micro/kernels/kernel_util.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/transpose.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/reshape.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/reshape_common.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/gelu.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/conv_common.cpp
  ${TFLM}/tensorflow/lite/micro/kernels/add_common.cpp
  # portable reference kernels for the CMSIS-only ops (see tflm_ref/)
  ${NTTS}/host/tflm_ref/conv.cpp
  ${NTTS}/host/tflm_ref/add.cpp
  ${NTTS}/host/tflm_ref/transpose_conv.cpp
  # host platform hooks (DebugLog / InitializeTarget / micro_time)
  ${NTTS}/host/tflm_ref/host_platform.cpp
)

add_library(tflm_host STATIC ${TFLM_SOURCES})
target_include_directories(tflm_host PUBLIC
  ${TFLM}
  ${TFLM}/third_party/flatbuffers/include
  ${TFLM}/third_party/gemmlowp
  ${TFLM}/third_party/ruy
)
target_compile_definitions(tflm_host PUBLIC
  TF_LITE_STATIC_MEMORY=1
  TF_LITE_DISABLE_X86_NEON=1
  # Mirror the device build: excludes the full-TFLite-only dynamic-tensor
  # helpers (TfLiteTensorDataFree) that the micro runtime doesn't provide.
  ARDUINO=1
)

# ---------------------------------------------------------------------------
# kissfft (float), as the parent build configures it.
# ---------------------------------------------------------------------------
add_library(kissfft_host STATIC
  ${MM}/third-party/kissfft/kiss_fft.c
  ${MM}/third-party/kissfft/kiss_fftr.c
)
target_include_directories(kissfft_host PUBLIC ${MM}/third-party/kissfft)
target_compile_definitions(kissfft_host PUBLIC kiss_fft_scalar=float)

# ---------------------------------------------------------------------------
# tts_cli: engine + g2p + klatt-tts + CLI. Compiled WITHOUT PICO_BUILD, so the
# engine's platform guards take their host fallbacks.
# ---------------------------------------------------------------------------
add_executable(tts_cli
  ${NTTS}/host/tts_cli.cc
  ${NTTS}/src/neural_tts.cc
  ${NTTS}/src/pb_decoder.cc
  ${NTTS}/src/worldlite_synth.cc
  ${NTTS}/src/hooks.cc
  ${MM}/g2p/src/g2p.cc
  ${MM}/g2p/src/g2p_rules.cc
  ${MM}/g2p/src/g2p_numbers.cc
  ${MM}/g2p/src/g2p_dict.cc
  ${MM}/g2p/src/ipa_tokens.cc
  ${MM}/klatt-tts/src/config.cc
  ${MM}/klatt-tts/src/klatt.cc
  ${MM}/klatt-tts/src/phonemes.cc
  ${MM}/klatt-tts/src/synth_internal.cc
  ${MM}/klatt-tts/src/synth_stream.cc
)
target_include_directories(tts_cli PRIVATE
  ${NTTS}/include
  ${NTTS}/src
  ${MM}/g2p/include        # neural_tts.cc: "g2p/g2p.h"
  ${MM}/g2p/include/g2p    # g2p sources: bare "g2p.h"
  ${MM}/g2p/src            # g2p sources: "g2p_dict.h" etc.
  ${MM}/klatt-tts/include        # neural_tts.cc: "tts/config.h"
  ${MM}/klatt-tts/include/tts    # klatt-tts sources: bare "config.h" etc.
  ${MM}/klatt-tts/src
)
target_link_libraries(tts_cli PRIVATE tflm_host kissfft_host)
