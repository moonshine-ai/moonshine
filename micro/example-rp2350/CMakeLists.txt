# example-rp2350 -- the RP2350 (Raspberry Pi Pico 2) firmware that ties the
# feature-generation, stt, vad, and tts modules together into a working app.
#
# This is the ONLY app/platform-specific target: the Pico SDK, USB CDC stdio,
# the dual-core stack linker script, the embedded model/clip blobs, and the
# audio-I/O implementations all live here. The recognition logic itself is in the
# platform-agnostic modules, which a different example on another board could
# reuse unchanged.
#
# Two boot paths, selected at configure time (see the parent CMakeLists options):
#   * default                         -> the live mic/speaker service (spelling_app)
#   * -DSPELLING_TINY_TEST_SWEEP=ON   -> the embedded-clip accuracy sweep (test_app)
#
# Assumes the parent CMakeLists has done pico_sdk_init(), added pico-tflmicro and
# the modules, and declared the SPELLING_TINY_* options.

# ----------------------------------------------------------------------------
# Common to both paths: the entry point, board bring-up, the model + front-end
# tables, and the class labels (the live service speaks labels too).
add_executable(predict_spelling_tiny
  src/main.cc
  src/app_common.cc

  generated/model_data.cc
  generated/classes.cc
  generated/mel_tables.cc
)

target_include_directories(predict_spelling_tiny PRIVATE
  src
  generated
)

target_link_libraries(predict_spelling_tiny
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
# exception tables are consistent across the whole link.
target_compile_options(predict_spelling_tiny PRIVATE
  -Os
  -fno-rtti
  -fno-exceptions
  -fno-threadsafe-statics
  -ffunction-sections
  -fdata-sections
)

# CRITICAL: make stdio_usb consider the device "connected" as soon as USB
# enumerates, WITHOUT requiring the host to assert DTR -- otherwise `cat`-based
# monitors silently discard every printf. Also bump the heap from the SDK
# default of 2 KiB (the first new[] would otherwise hit null and abort() before
# any output) and give the heap-free log-mel a comfortable stack margin.
target_compile_definitions(predict_spelling_tiny PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_HEAP_SIZE=24576
  PICO_STACK_SIZE=4096
)

# Mirror the dual-core macro into the app (for the boot banner) and pull in
# pico_multicore at the final link. The actual GEMM split is compiled into the
# pico-tflmicro library by the parent CMakeLists; the app just reports the state.
if(SPELLING_TINY_MULTICORE)
  target_compile_definitions(predict_spelling_tiny PRIVATE SPELLING_TINY_MULTICORE)
  target_link_libraries(predict_spelling_tiny pico_multicore)
endif()

# ----------------------------------------------------------------------------
# Path selection.
if(SPELLING_TINY_AUDIO)
  # --- Live mic/speaker recognition service (default) ---
  message(STATUS "example-rp2350: live audio service (SPELLING_TINY_AUDIO)")
  target_sources(predict_spelling_tiny PRIVATE
    src/spelling_app.cc
    src/audio_service.cc
    src/usb_audio_io.cc
  )
  # audio_service streams binary int16 PCM over the CDC byte pipe; disable
  # LF->CRLF so a 0x0A sample byte isn't corrupted.
  target_compile_definitions(predict_spelling_tiny PRIVATE
    SPELLING_TINY_AUDIO
    PICO_STDIO_DEFAULT_CRLF=0
  )
  # The service speaks replies via the TTS synth core.
  target_link_libraries(predict_spelling_tiny tts)

  # Dual-core stack-overflow guard: swap the per-core stacks so a core-0 overflow
  # during the deep VAD->STT->TTS call chain spills into unused RAM instead of
  # core 1's live stack. See memmap_dualcore_stack.ld and petewarden.com/2024/01/16.
  if(SPELLING_TINY_MULTICORE)
    pico_set_linker_script(predict_spelling_tiny
      ${CMAKE_CURRENT_SOURCE_DIR}/memmap_dualcore_stack.ld)
    message(STATUS "example-rp2350: using dual-core stack-swap linker script")
  endif()
else()
  # --- Embedded-clip accuracy sweep (opt-in test) ---
  message(STATUS "example-rp2350: embedded-clip test sweep (SPELLING_TINY_TEST_SWEEP)")
  target_sources(predict_spelling_tiny PRIVATE
    src/test_app.cc
    generated/test_clips.cc
  )
  target_compile_definitions(predict_spelling_tiny PRIVATE SPELLING_TINY_TEST_SWEEP)

  # Optional: per-op timing breakdown on the first clip.
  if(SPELLING_TINY_PROFILE_OPS)
    message(STATUS "example-rp2350: per-op profiling enabled (SPELLING_TINY_PROFILE_OPS)")
    target_sources(predict_spelling_tiny PRIVATE src/op_profiler.cc)
    target_compile_definitions(predict_spelling_tiny PRIVATE SPELLING_TINY_PROFILE_OPS)
  endif()

  # Optional: TTS USB speak command loop after the sweep.
  if(SPELLING_TINY_TTS)
    message(STATUS "example-rp2350: TTS speak loop enabled (SPELLING_TINY_TTS)")
    target_sources(predict_spelling_tiny PRIVATE src/tts_service.cc)
    target_link_libraries(predict_spelling_tiny tts)
    target_compile_definitions(predict_spelling_tiny PRIVATE
      SPELLING_TINY_TTS
      PICO_STDIO_DEFAULT_CRLF=0
    )
  endif()
endif()

# ----------------------------------------------------------------------------
# On-device VAD (the live service forces it on; the test sweep can opt in for a
# demo). Requires the generated VAD blobs (see vad/scripts).
if(SPELLING_TINY_VAD)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/vad_model_data.cc")
    message(FATAL_ERROR
      "SPELLING_TINY_VAD=ON but generated/vad_model_data.cc is missing. "
      "Embed a VAD model first:\n"
      "  python cpp-tiny/vad/scripts/generate_vad_embedded_data.py \\\n"
      "    --tflite cpp-tiny/models/tinyvad_cnn_speech_mel_head16.tflite")
  endif()
  message(STATUS "example-rp2350: on-device VAD enabled (SPELLING_TINY_VAD)")
  target_sources(predict_spelling_tiny PRIVATE
    generated/vad_mel_tables.cc
    generated/vad_model_data.cc
  )
  target_link_libraries(predict_spelling_tiny vad)
  target_compile_definitions(predict_spelling_tiny PRIVATE SPELLING_TINY_VAD)
endif()

# Wire stdio through USB CDC (a single USB cable reads the output); UART off.
pico_enable_stdio_usb(predict_spelling_tiny  1)
pico_enable_stdio_uart(predict_spelling_tiny 0)

# Generate .uf2 / .elf / .hex / .dis / .map alongside the binary.
pico_add_extra_outputs(predict_spelling_tiny)
