# cpp-tiny -- on-device spelling recognition for the Raspberry Pi Pico 2
# (RP2350), built as a set of reusable, platform-agnostic modules plus one
# concrete example app.
#
# Modules (each with its own CMakeLists, public header, README, and tests):
#   feature-generation/  log-mel front-end (batch + streaming)   [kissfft + TFLM]
#   vad/                 voice activity detection                [feature-gen + TFLM]
#   stt/                 SpellingCNN classifier                  [feature-gen + TFLM]
#   tts/                 portable formant TTS synth core         [self-contained]
#   example-rp2350/      the RP2350 firmware that wires them up  [Pico SDK]
#
# TFLM is decoupled behind an INTERFACE target named `tflm`, so the modules don't
# hard-code which TFLM implementation is linked. Here it resolves to the vendored
# pico-tflmicro; on another platform you'd point `tflm` at your own TFLM.
#
# === Two build modes ===
#
# 1. RP2350 firmware (default). Needs the Pico SDK + the Arm GNU toolchain:
#      cd cpp-tiny && mkdir -p build && cd build
#      cmake -DPICO_SDK_PATH=$HOME/projects/pico-sdk .. && cmake --build . -j
#    Result: build/example-rp2350/predict_spelling_tiny.uf2
#
# 2. Host unit tests (no SDK, normal compiler):
#      cmake -B build-host -S . -DCPP_TINY_HOST_TESTS=ON
#      cmake --build build-host -j && ctest --test-dir build-host --output-on-failure

cmake_minimum_required(VERSION 3.13)

option(CPP_TINY_HOST_TESTS
       "Build the per-module host unit tests instead of the RP2350 firmware." OFF)

# ===========================================================================
# Mode 2: host unit tests.
# ===========================================================================
if(CPP_TINY_HOST_TESTS)
  project(cpp_tiny_host_tests C CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
  set(CPP_TINY_BUILD_TESTS ON)

  include(cmake/CppTinyTests.cmake)

  # kissfft (real lib) + a TFLM headers-only interface (the host stub provides
  # the few symbols the tests reference -- MicroPrintf / InitializeTarget).
  add_subdirectory(third-party/kissfft)
  add_library(tflm INTERFACE)
  target_include_directories(tflm INTERFACE third-party/pico-tflmicro/src)

  add_library(tflm_test_support STATIC test-support/host/tflm_host_stub.cc)
  target_include_directories(tflm_test_support PUBLIC third-party/pico-tflmicro/src)

  # Modules. On host the interpreter-dependent sources are auto-excluded, so the
  # logic (front-end, segmenter, predictor, synth) is testable without a host
  # TFLM build.
  add_subdirectory(feature-generation)
  add_subdirectory(vad)
  add_subdirectory(stt)
  add_subdirectory(tts)
  return()
endif()

# ===========================================================================
# Mode 1: RP2350 firmware.
# ===========================================================================

# Target the Pico 2 (RP2350) by default. Override with -DPICO_BOARD=...
set(PICO_BOARD pico2 CACHE STRING "Pico board variant (pico2, pico2_w, ...)")

# Pull in the SDK BEFORE project() per pico_sdk_init() requirements.
include(pico_sdk_import.cmake)

project(predict_spelling_tiny C CXX ASM)
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pico_sdk_init()

include(cmake/CppTinyTests.cmake)

# --- Build options ---------------------------------------------------------
# Declared here (before add_subdirectory) because some affect how the
# pico-tflmicro library itself is compiled, and the example reads all of them.

# Split the int8 SIMD GEMM + 3x3 depthwise conv across both Cortex-M33 cores
# (~1.6x inference speedup, bit-identical output). ON = the deployed config.
option(SPELLING_TINY_MULTICORE
       "Split the int8 SIMD GEMM + depthwise conv across both RP2350 cores." ON)

# Per-op timing breakdown on the first clip.
option(SPELLING_TINY_PROFILE_OPS
       "Record + print a per-op timing breakdown on the first clip." OFF)

# Emit the GreedyMemoryPlanner ASCII memory-plan diagram on every
# AllocateTensors() (the equivalent of desktop TFLM's TF_LITE_SHOW_MEMORY_USE).
option(SPELLING_TINY_PRINT_MEMORY_PLAN
       "Add -DTF_LITE_SHOW_MEMORY_USE to pico-tflmicro for arena debugging." OFF)

# Boot into the embedded-clip accuracy sweep test instead of the live service.
# This is the opt-in development/CI path; it turns the live audio service off.
option(SPELLING_TINY_TEST_SWEEP
       "Boot the embedded-clip accuracy sweep test instead of the live service." OFF)

# Streaming formant TTS engine + USB speak command loop (the latter only in the
# test-sweep build; the live service speaks via the synth core directly).
option(SPELLING_TINY_TTS
       "Build in the streaming formant TTS engine." ON)

# On-device VAD model (+ a demo over the embedded clips in the test build).
option(SPELLING_TINY_VAD
       "Build in the on-device VAD model + segmenter." OFF)

# Live USB mic/speaker audio service (laptop as the RP2350's A/D + D/A) -- the
# DEFAULT app. Uses all three models, so it forces VAD + TTS on. Mutually
# exclusive with the test sweep (which forces this off).
option(SPELLING_TINY_AUDIO
       "Live USB mic/speaker audio service (laptop as the RP2350's A/D + D/A)." ON)

if(SPELLING_TINY_TEST_SWEEP)
  set(SPELLING_TINY_AUDIO OFF CACHE BOOL "off in test-sweep mode" FORCE)
  message(STATUS "cpp-tiny: embedded-clip test sweep selected (live service off)")
endif()
if(SPELLING_TINY_AUDIO)
  set(SPELLING_TINY_VAD ON CACHE BOOL "forced on by SPELLING_TINY_AUDIO" FORCE)
  set(SPELLING_TINY_TTS ON CACHE BOOL "forced on by SPELLING_TINY_AUDIO" FORCE)
  message(STATUS "cpp-tiny: live USB audio service enabled; forcing VAD+TTS on")
endif()

# --- Third-party -----------------------------------------------------------
# pico-tflmicro (vendored; CMakeLists trimmed to be subdirectory-friendly). It
# carries a small local patch -- the dual-core CMSIS-NN GEMM/depthwise split; see
# third-party/pico-tflmicro/PATCHES.md.
add_subdirectory(third-party/pico-tflmicro)
add_subdirectory(third-party/kissfft)

# Decouple the modules from the concrete TFLM: `tflm` is the agnostic handle.
add_library(tflm INTERFACE)
target_link_libraries(tflm INTERFACE pico-tflmicro)

if(SPELLING_TINY_MULTICORE)
  message(STATUS "cpp-tiny: dual-core GEMM enabled (SPELLING_TINY_MULTICORE)")
  target_compile_definitions(pico-tflmicro PRIVATE SPELLING_TINY_MULTICORE)
  target_link_libraries(pico-tflmicro pico_multicore hardware_sync)
endif()

if(SPELLING_TINY_PRINT_MEMORY_PLAN)
  message(STATUS "cpp-tiny: TF_LITE_SHOW_MEMORY_USE enabled (memory plan prints)")
  target_compile_definitions(pico-tflmicro PRIVATE TF_LITE_SHOW_MEMORY_USE)
endif()

# --- Embedded build flags for the modules + example ------------------------
# Match how the firmware links: -Os, no exceptions / RTTI / threadsafe-statics
# (TFLM is built that way too, so vtables and exception tables stay consistent),
# and function/data sections so --gc-sections strips the unused paths (e.g. the
# front-end's runtime-table fallback, which the flash-table build never calls).
# The C++-only flags are guarded so the SDK's C sources don't warn. Applied here
# so it covers the module libraries, not just the executable -- otherwise their
# objects keep exception machinery + dead code and overflow flash.
add_compile_options(
  -Os
  -ffunction-sections
  -fdata-sections
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
)

# --- Modules ---------------------------------------------------------------
add_subdirectory(feature-generation)
add_subdirectory(stt)
add_subdirectory(vad)
add_subdirectory(tts)

# --- The example app -------------------------------------------------------
add_subdirectory(example-rp2350)
