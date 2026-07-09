# examples/rp2350 -- the RP2350 (Raspberry Pi Pico 2) firmware that ties the
# feature-generation, stt, vad, and tts modules together into working apps.
#
# This is the ONLY app/platform-specific code: the Pico SDK, USB CDC stdio, the
# dual-core stack linker script, the embedded model/clip blobs, and the audio-I/O
# implementations all live here. The recognition logic itself is in the
# platform-agnostic modules, which a different example could reuse unchanged.
#
# One executable target PER APP (pick which .uf2 to flash) -- no compile-time
# macro picks "the app" any more:
#   * moonshine_micro_echo       live mic/speaker echo service       [main_live.cc]
#   * moonshine_micro_echo_wifi  voice-driven WiFi setup (Pico 2 W)  [main_wifi.cc]
#   * moonshine_micro_echo_test  embedded-clip accuracy sweep        [main_test.cc]
#
# The WiFi target is only created when the selected board has a CYW43 radio
# (PICO_CYW43_SUPPORTED, e.g. -DPICO_BOARD=pico2_w). Build-config knobs
# (SPELLING_TINY_MULTICORE / _VAD / _TTS / _PROFILE_OPS) still come from the
# parent CMakeLists; they tune HOW a target is built, not WHICH app it is.
#
# Assumes the parent CMakeLists has done pico_sdk_init(), added pico-tflmicro and
# the modules, and declared the SPELLING_TINY_* build-config options.

# ----------------------------------------------------------------------------
# Sources every app shares: board bring-up, the model + front-end tables, and
# the class labels (the live service speaks labels too).
set(SPELLING_COMMON_SRC
  src/app_common.cc
  generated/model_data.cc
  generated/classes.cc
  generated/mel_tables.cc
)

# Helper: stamp the shared include dirs, libraries, embedded build flags, heap/
# stack + stdio config, the dual-core macro, and the .uf2 output onto an app
# target. `main_src` is the per-app entry point (the only thing that differs at
# the top level).
function(spelling_add_app target main_src)
  add_executable(${target} ${main_src} ${SPELLING_COMMON_SRC})

  target_include_directories(${target} PRIVATE src generated)

  target_link_libraries(${target}
    feature_generation
    stt
    kissfft
    pico-tflmicro
    pico_stdlib
    # 250 MHz overclock via set_sys_clock_khz + the core-voltage bump.
    hardware_clocks
    hardware_vreg
  )

  # TFLM is built -fno-exceptions / -fno-rtti -- match those so vtables and
  # exception tables are consistent across the whole link. The C++-only flags
  # are guarded with a generator expression so they aren't passed to the C
  # sources the SDK (TinyUSB, pico_stdlib, ...) compiles into this target,
  # which would otherwise warn "valid for C++ but not for C" once per file.
  target_compile_options(${target} PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
  )

  # CRITICAL: make stdio_usb consider the device "connected" as soon as USB
  # enumerates, WITHOUT requiring the host to assert DTR -- otherwise `cat`-based
  # monitors silently discard every printf. Also bump the heap from the SDK
  # default of 2 KiB (the first new[] would otherwise hit null and abort() before
  # any output) and give the heap-free log-mel a comfortable stack margin.
  target_compile_definitions(${target} PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_HEAP_SIZE=24576
    PICO_STACK_SIZE=4096
  )

  # Mirror the dual-core macro into the app (for the boot banner) and pull in
  # pico_multicore. The actual GEMM split is compiled into pico-tflmicro by the
  # parent CMakeLists; the app just reports + uses the state.
  if(SPELLING_TINY_MULTICORE)
    target_compile_definitions(${target} PRIVATE SPELLING_TINY_MULTICORE)
    target_link_libraries(${target} pico_multicore)
  endif()

  # Wire stdio through USB CDC (a single USB cable reads the output); UART off.
  pico_enable_stdio_usb(${target}  1)
  pico_enable_stdio_uart(${target} 0)

  # Generate .uf2 / .elf / .hex / .dis / .map alongside the binary.
  pico_add_extra_outputs(${target})
endfunction()

# Helper: turn an app into a live mic/speaker (audio) app -- the streaming VAD +
# STT + TTS pipeline, streaming binary int16 PCM over CDC (so LF->CRLF must be
# off), with the dual-core stack-swap guard. Shared by the live + WiFi apps.
set(SPELLING_AUDIO_SRC
  src/audio_service.cc
  src/usb_audio_io.cc
  generated/vad_mel_tables.cc
  generated/vad_model_data.cc
)
function(spelling_configure_audio target)
  target_sources(${target} PRIVATE ${SPELLING_AUDIO_SRC})
  target_link_libraries(${target} vad tts)
  # audio_service streams binary int16 PCM over the CDC byte pipe; disable
  # LF->CRLF so a 0x0A sample byte isn't corrupted.
  target_compile_definitions(${target} PRIVATE PICO_STDIO_DEFAULT_CRLF=0)

  # Dual-core stack-overflow guard: swap the per-core stacks so a core-0 overflow
  # during the deep VAD->STT->TTS call chain spills into unused RAM instead of
  # core 1's live stack. See memmap_dualcore_stack.ld and petewarden.com/2024/01/16.
  if(SPELLING_TINY_MULTICORE)
    pico_set_linker_script(${target}
      ${CMAKE_CURRENT_SOURCE_DIR}/memmap_dualcore_stack.ld)
  endif()
endfunction()

# The live (audio) apps embed the VAD model + mel tables. Validate the blobs are
# present once, up front, with an actionable message.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/vad_model_data.cc")
  message(FATAL_ERROR
    "generated/vad_model_data.cc is missing (needed by the live + WiFi apps). "
    "Embed a VAD model first:\n"
    "  python moonshine-micro/vad/scripts/generate_vad_embedded_data.py \\\n"
    "    --tflite moonshine-micro/models/tinyvad_cnn_speech_mel_head16.tflite")
endif()

# ----------------------------------------------------------------------------
# App 1: the live mic/speaker echo service (the default artifact).
spelling_add_app(moonshine_micro_echo src/main_live.cc)
target_sources(moonshine_micro_echo PRIVATE src/echo_app.cc)
spelling_configure_audio(moonshine_micro_echo)
message(STATUS "examples/rp2350: + moonshine_micro_echo (live echo service)")

# ----------------------------------------------------------------------------
# App 2: voice-driven WiFi setup. Only buildable on a CYW43 board (e.g. pico2_w).
if(PICO_CYW43_SUPPORTED)
  spelling_add_app(moonshine_micro_echo_wifi src/main_wifi.cc)
  target_sources(moonshine_micro_echo_wifi PRIVATE src/wifi_app.cc)
  spelling_configure_audio(moonshine_micro_echo_wifi)

  # Trim the tensor arena (384 -> 360 KiB) to free SRAM for the CYW43 driver +
  # lwIP stack; still clears the SpellingCNN's ~346 KiB working set.
  target_compile_definitions(moonshine_micro_echo_wifi PRIVATE
    SPELLING_TINY_ARENA_BYTES=368640)  # 360 * 1024

  # Our minimized lwipopts.h lives next to this CMakeLists; pico_lwip compiles
  # the lwIP sources in this target's context, so it must be on the include path.
  target_include_directories(moonshine_micro_echo_wifi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # Poll-mode CYW43 + lwIP: no IRQ/alarm threads, so it won't fight the dual-core
  # SIMD GEMM; the app pumps cyw43_arch_poll() only in the connect / IP-readback
  # phases (never during inference).
  target_link_libraries(moonshine_micro_echo_wifi pico_cyw43_arch_lwip_poll)
  message(STATUS "examples/rp2350: + moonshine_micro_echo_wifi (voice WiFi setup)")
else()
  message(STATUS
    "examples/rp2350: skipping moonshine_micro_echo_wifi -- board '${PICO_BOARD}' "
    "has no CYW43 (configure with -DPICO_BOARD=pico2_w to build it)")
endif()

# ----------------------------------------------------------------------------
# App 3: embedded-clip accuracy sweep (development / CI).
spelling_add_app(moonshine_micro_echo_test src/main_test.cc)
target_sources(moonshine_micro_echo_test PRIVATE
  src/test_app.cc
  generated/test_clips.cc
)

# Optional: per-op timing breakdown on the first clip.
if(SPELLING_TINY_PROFILE_OPS)
  message(STATUS "examples/rp2350: test sweep -- per-op profiling enabled")
  target_sources(moonshine_micro_echo_test PRIVATE src/op_profiler.cc)
  target_compile_definitions(moonshine_micro_echo_test PRIVATE SPELLING_TINY_PROFILE_OPS)
endif()

# Optional: a streaming VAD demo over the embedded clips (needs the VAD blobs).
if(SPELLING_TINY_VAD)
  message(STATUS "examples/rp2350: test sweep -- VAD demo enabled")
  target_sources(moonshine_micro_echo_test PRIVATE
    generated/vad_mel_tables.cc
    generated/vad_model_data.cc
  )
  target_link_libraries(moonshine_micro_echo_test vad)
  target_compile_definitions(moonshine_micro_echo_test PRIVATE SPELLING_TINY_VAD)
endif()

# Optional: TTS USB speak command loop after the sweep.
if(SPELLING_TINY_TTS)
  message(STATUS "examples/rp2350: test sweep -- TTS speak loop enabled")
  target_sources(moonshine_micro_echo_test PRIVATE src/tts_service.cc)
  target_link_libraries(moonshine_micro_echo_test tts)
  target_compile_definitions(moonshine_micro_echo_test PRIVATE
    SPELLING_TINY_TTS
    PICO_STDIO_DEFAULT_CRLF=0
  )
endif()
message(STATUS "examples/rp2350: + moonshine_micro_echo_test (embedded-clip sweep)")
