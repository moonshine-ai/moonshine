# moonshine-micro -- on-device spelling recognition for the Raspberry Pi Pico 2
# (RP2350), built as a set of reusable, platform-agnostic modules plus one
# concrete example app.
#
# Modules (each with its own CMakeLists, public header, README, and tests):
#   feature-generation/  log-mel front-end (batch + streaming)   [kissfft + TFLM]
#   vad/                 voice activity detection                [feature-gen + TFLM]
#   stt/                 SpellingCNN classifier                  [feature-gen + TFLM]
#   tts/                 portable formant TTS synth core         [self-contained]
#   examples/rp2350/      the RP2350 firmware that wires them up  [Pico SDK]
#
# TFLM is decoupled behind an INTERFACE target named `tflm`, so the modules don't
# hard-code which TFLM implementation is linked. Here it resolves to the vendored
# pico-tflmicro; on another platform you'd point `tflm` at your own TFLM.
#
# === Two build modes ===
#
# 1. RP2350 firmware (default). Needs the Pico SDK + the Arm GNU toolchain:
#      cd moonshine-micro && mkdir -p build && cd build
#      cmake -DPICO_SDK_PATH=$HOME/projects/pico-sdk .. && cmake --build . -j
#    Builds one .uf2 per app under build/examples/rp2350/:
#      moonshine_micro_echo.uf2       live mic/speaker echo service
#      moonshine_micro_echo_test.uf2  embedded-clip accuracy sweep
#      moonshine_micro_echo_wifi.uf2  voice WiFi setup (only with -DPICO_BOARD=pico2_w)
#
# 2. Host unit tests (no SDK, normal compiler):
#      cmake -B build-host -S . -DMOONSHINE_MICRO_HOST_TESTS=ON
#      cmake --build build-host -j && ctest --test-dir build-host --output-on-failure

cmake_minimum_required(VERSION 3.13)

option(MOONSHINE_MICRO_HOST_TESTS
       "Build the per-module host unit tests instead of the RP2350 firmware." OFF)

# ===========================================================================
# Mode 2: host unit tests.
# ===========================================================================
if(MOONSHINE_MICRO_HOST_TESTS)
  project(moonshine_micro_host_tests C CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
  set(MOONSHINE_MICRO_BUILD_TESTS ON)

  include(cmake/MoonshineMicroTests.cmake)

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

# Target the Pico 2 (RP2350) by default. Build for the Pico 2 W
# (-DPICO_BOARD=pico2_w) to also get the voice WiFi-setup target, which needs the
# CYW43 radio. Override with -DPICO_BOARD=...
set(PICO_BOARD pico2 CACHE STRING "Pico board variant (pico2, pico2_w, ...)")

# Pull in the SDK BEFORE project() per pico_sdk_init() requirements.
include(pico_sdk_import.cmake)

project(moonshine_micro_echo C CXX ASM)
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pico_sdk_init()

include(cmake/MoonshineMicroTests.cmake)

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

# NOTE: which app to run is no longer a compile-time switch -- each app is its
# own executable target (see examples/rp2350/CMakeLists.txt):
#   moonshine_micro_echo       live mic/speaker echo service
#   moonshine_micro_echo_wifi  voice WiFi setup (only on a CYW43 board)
#   moonshine_micro_echo_test  embedded-clip accuracy sweep
# Build all of them (`cmake --build build`) or one (`--target <name>`). The
# live + WiFi apps always link the VAD + STT + TTS pipeline. The options below
# only tune optional demos inside the TEST target.

# Streaming formant TTS engine USB speak command loop in the test sweep (the
# live + WiFi apps always link the synth core directly, regardless of this).
option(SPELLING_TINY_TTS
       "Add the TTS USB speak-loop demo to the test-sweep target." ON)

# A streaming VAD demo over the embedded clips in the test sweep (the live + WiFi
# apps always include the on-device VAD, regardless of this).
option(SPELLING_TINY_VAD
       "Add the on-device VAD demo to the test-sweep target." OFF)

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
  message(STATUS "moonshine-micro: dual-core GEMM enabled (SPELLING_TINY_MULTICORE)")
  target_compile_definitions(pico-tflmicro PRIVATE SPELLING_TINY_MULTICORE)
  target_link_libraries(pico-tflmicro pico_multicore hardware_sync)
endif()

if(SPELLING_TINY_PRINT_MEMORY_PLAN)
  message(STATUS "moonshine-micro: TF_LITE_SHOW_MEMORY_USE enabled (memory plan prints)")
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
add_subdirectory(examples/rp2350)
