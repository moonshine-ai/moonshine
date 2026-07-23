# examples/rp2350 -- the RP2350 (Raspberry Pi Pico 2) firmware that ties the
# feature-generation, stt, vad, and klatt-tts modules together into working apps.
#
# This is the ONLY app/platform-specific code: the Pico SDK, USB CDC stdio, the
# dual-core stack linker script, the embedded model/clip blobs, and the audio-I/O
# implementations all live here. The recognition logic itself is in the
# platform-agnostic modules, which a different example could reuse unchanged.
#
# One executable target PER APP (pick which .uf2 to flash) -- no compile-time
# macro picks "the app" any more:
#   * moonshine_micro_echo              live mic/speaker echo (USB bridge)  [main_live.cc]
#   * moonshine_micro_echo_hardware     live echo on I2S mic + I2S amp       [main_echo_hardware.cc]
#   * moonshine_micro_echo_wifi          voice WiFi setup, USB bridge (Pico 2 W)  [main_wifi.cc]
#   * moonshine_micro_echo_wifi_hardware voice WiFi setup, I2S mic + I2S amp    [main_wifi_hardware.cc]
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
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
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

# Helper: attach the neural TTS engine (neural-tts/) + its flash data pack to
# a target. The pack (~2.2 MB: TFLM decoder model, RVQ codebooks, diphone/word
# unit streams, prosody tables) is .incbin'd via the generated .S stub; the
# assembler needs -I to resolve the relative "neural_tts_pack.bin" path.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_pack.bin")
  message(FATAL_ERROR
    "generated/neural_tts_pack.bin is missing (needed by the TTS reply path). "
    "Export the neural TTS pack first:\n"
    "  python scripts/export_neural_tts_pack.py")
endif()
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_pack.S PROPERTIES
  COMPILE_OPTIONS "-Wa,-I${CMAKE_CURRENT_SOURCE_DIR}/generated"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_pack.bin")
function(spelling_configure_neural_tts target)
  target_sources(${target} PRIVATE generated/neural_tts_pack.S)
  target_link_libraries(${target} neural_tts)
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
  target_link_libraries(${target} vad)
  spelling_configure_neural_tts(${target})
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

# Helper: live echo on real I2S mic + I2S amp (MAX98357A; no USB audio bridge).
# Optional 2nd arg: tensor-arena bytes (default 352 KiB). Trimmed from 384 KiB to
# make room for the I2S DMA rings (~40 KiB: 32 KiB out drain + 8 KiB in capture).
function(spelling_configure_hardware_audio target)
  set(hw_arena_bytes "${ARGV1}")
  if(NOT hw_arena_bytes)
    set(hw_arena_bytes 360448)  # 352 * 1024
  endif()
  target_sources(${target} PRIVATE
    src/audio_service.cc
    src/i2s_audio_io.cc
    src/i2s_audio_out.cc
    src/i2s_mic_process.cc
    generated/vad_mel_tables.cc
    generated/vad_model_data.cc
  )
  target_link_libraries(${target} vad hardware_pio hardware_dma)
  spelling_configure_neural_tts(${target})
  target_compile_definitions(${target} PRIVATE
    SPELLING_TINY_ARENA_BYTES=${hw_arena_bytes})
  pico_generate_pio_header(${target}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_mic.pio)
  pico_generate_pio_header(${target}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_out.pio)

  # NB: the verbose recognizer logs (SPELLING_AUDIO_DIAG) and the binary STT
  # capture dump (SPELLING_DUMP_STT_USB, which also needs PICO_STDIO_DEFAULT_CRLF=0
  # so a 0x0A sample byte isn't mangled) are set PER TARGET, not here: the
  # diagnostic echo build wants them; the product wifi_hardware build wants a
  # quiet log, so it leaves them off.

  if(SPELLING_TINY_MULTICORE)
    pico_set_linker_script(${target}
      ${CMAKE_CURRENT_SOURCE_DIR}/memmap_dualcore_stack.ld)
  endif()
endfunction()

# Helper: layer the CYW43 radio + lwIP onto a (already audio-configured) WiFi
# target. Shared by both WiFi apps (USB-bridge audio and on-board hardware
# audio) so the radio wiring lives in exactly one place. The optional second
# arg overrides the tensor-arena byte budget (default 360 KiB): the hardware
# variant needs a smaller arena because the I2S capture path adds an 8 KiB DMA
# ring buffer on top of the CYW43 + lwIP stacks.
function(spelling_configure_wifi target)
  set(arena_bytes "${ARGV1}")
  if(NOT arena_bytes)
    set(arena_bytes 368640)  # 360 * 1024
  endif()
  # Trim the tensor arena (from 384 KiB) to free SRAM for the CYW43 driver +
  # lwIP stack (+ I2S ring on the hardware build); still clears the SpellingCNN's
  # ~346 KiB working set.
  target_compile_definitions(${target} PRIVATE
    SPELLING_TINY_ARENA_BYTES=${arena_bytes})

  # Our minimized lwipopts.h lives next to this CMakeLists; pico_lwip compiles
  # the lwIP sources in this target's context, so it must be on the include path.
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # Poll-mode CYW43 + lwIP: no IRQ/alarm threads, so it won't fight the dual-core
  # SIMD GEMM; the app pumps cyw43_arch_poll() only in the connect / IP-readback
  # phases (never during inference).
  target_link_libraries(${target} pico_cyw43_arch_lwip_poll)
endfunction()

# The live (audio) apps embed the VAD model + mel tables.
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
# Diagnostic build: keep the verbose recognizer heartbeats / signal stats.
target_compile_definitions(moonshine_micro_echo PRIVATE SPELLING_AUDIO_DIAG=1)
message(STATUS "examples/rp2350: + moonshine_micro_echo (live echo, USB bridge)")

# ----------------------------------------------------------------------------
# App 1b: live echo on on-board I2S mic + I2S amp (no USB audio bridge).
spelling_add_app(moonshine_micro_echo_hardware src/main_echo_hardware.cc)
target_sources(moonshine_micro_echo_hardware PRIVATE src/echo_hardware_app.cc)
spelling_configure_hardware_audio(moonshine_micro_echo_hardware)
# Diagnostic build: verbose recognizer logs + the binary STT capture dump
# (scripts/capture_stt.py), which needs LF->CRLF off so a 0x0A sample byte
# isn't corrupted on the CDC.
target_compile_definitions(moonshine_micro_echo_hardware PRIVATE
  PICO_STDIO_DEFAULT_CRLF=0
  SPELLING_DUMP_STT_USB=1
  SPELLING_AUDIO_DIAG=1)
message(STATUS "examples/rp2350: + moonshine_micro_echo_hardware (live echo, hardware I/O)")

# ----------------------------------------------------------------------------
# App 2: voice-driven WiFi setup. Only buildable on a CYW43 board (e.g. pico2_w).
if(PICO_CYW43_SUPPORTED)
  # CYW43 builds use a smaller tensor arena; shorter TTS chunks keep the
  # pb_decoder TFLM sub-arena (~144 KiB) allocatable alongside the frame track.
  target_compile_definitions(neural_tts PRIVATE NEURAL_TTS_MAX_CHUNK_FRAMES=300)

  # 2a: WiFi setup over the USB audio bridge (laptop is the mic + speaker).
  spelling_add_app(moonshine_micro_echo_wifi src/main_wifi.cc)
  target_sources(moonshine_micro_echo_wifi PRIVATE src/wifi_app.cc)
  spelling_configure_audio(moonshine_micro_echo_wifi)
  spelling_configure_wifi(moonshine_micro_echo_wifi)
  message(STATUS "examples/rp2350: + moonshine_micro_echo_wifi (voice WiFi setup, USB bridge)")

  # 2b: same setup flow on the on-board I2S mic + I2S amp. Reuses the shared
  # RunWifiAppWithIo() state machine (src/wifi_app.cc) with the hardware audio
  # backend, then layers the same radio config on top.
  spelling_add_app(moonshine_micro_echo_wifi_hardware src/main_wifi_hardware.cc)
  target_sources(moonshine_micro_echo_wifi_hardware PRIVATE
    src/wifi_app.cc
    src/wifi_hardware_app.cc)
  spelling_configure_hardware_audio(moonshine_micro_echo_wifi_hardware 354304)  # 346 * 1024
  spelling_configure_wifi(moonshine_micro_echo_wifi_hardware 354304)
  message(STATUS "examples/rp2350: + moonshine_micro_echo_wifi_hardware (voice WiFi setup, hardware I/O)")
else()
  message(STATUS
    "examples/rp2350: skipping moonshine_micro_echo_wifi[_hardware] -- board "
    "'${PICO_BOARD}' has no CYW43 (configure with -DPICO_BOARD=pico2_w to build them)")
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
  spelling_configure_neural_tts(moonshine_micro_echo_test)
  target_compile_definitions(moonshine_micro_echo_test PRIVATE
    SPELLING_TINY_TTS
    PICO_STDIO_DEFAULT_CRLF=0
  )
endif()
message(STATUS "examples/rp2350: + moonshine_micro_echo_test (embedded-clip sweep)")

# ----------------------------------------------------------------------------
# App 4: standalone I2S microphone bring-up test. Self-contained (no VAD/STT/TTS,
# no model blobs) -- it just clocks an SPH0645 I2S mic via PIO and prints live
# per-channel signal stats so you can confirm the mic is wired up and working.
add_executable(moonshine_micro_i2s_mic_test
  src/main_i2s_mic_test.cc
  src/i2s_mic_process.cc
  src/usb_audio_io.cc)
pico_generate_pio_header(moonshine_micro_i2s_mic_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_mic.pio)
target_include_directories(moonshine_micro_i2s_mic_test PRIVATE src)
target_link_libraries(moonshine_micro_i2s_mic_test
  pico_stdlib
  hardware_pio
  hardware_clocks
  hardware_vreg
)
target_compile_definitions(moonshine_micro_i2s_mic_test PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_STDIO_DEFAULT_CRLF=0
  PICO_HEAP_SIZE=24576
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_i2s_mic_test  1)
pico_enable_stdio_uart(moonshine_micro_i2s_mic_test 0)
pico_add_extra_outputs(moonshine_micro_i2s_mic_test)
message(STATUS "examples/rp2350: + moonshine_micro_i2s_mic_test (I2S mic bring-up)")

# ----------------------------------------------------------------------------
# App 5: standalone I2S audio-output test. Self-contained (no VAD/STT/TTS, no
# model blobs) -- drives the MAX98357A I2S amp (Adafruit 3006) + speaker via the
# same I2sAudioOutput backend the live apps use, and plays tones/clips so you
# can confirm audio out (i2s_audio_out.cc supplies PeakNormalizeGain).
add_executable(moonshine_micro_i2s_audio_test
  src/main_i2s_audio_test.cc
  src/i2s_audio_out.cc
  generated/speaker_test_clips.cc)
pico_generate_pio_header(moonshine_micro_i2s_audio_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_out.pio)
target_include_directories(moonshine_micro_i2s_audio_test PRIVATE src generated)
# Pull in the neural TTS engine + flash voice pack so the test can also speak a
# few letters/words through the same I2S sink (proves the full render path, not
# just tones/clips).
spelling_configure_neural_tts(moonshine_micro_i2s_audio_test)
target_link_libraries(moonshine_micro_i2s_audio_test
  pico_stdlib
  hardware_pio
  hardware_dma
  hardware_clocks
  hardware_vreg
  hardware_watchdog
)

# The neural-TTS decoder Invoke() runs the dual-core SIMD GEMM (compiled into
# pico-tflmicro under SPELLING_TINY_MULTICORE), which lazily launches core 1 on
# its own SCRATCH_X stack. The deep decode call chain overflows core 0's default
# SCRATCH_Y stack straight into that live core-1 stack, corrupting an op's
# function pointer (observed: hard fault, stacked PC=0, on the first decode op).
# Swap the per-core stacks so a core-0 overflow spills into unused RAM instead --
# same guard the audio apps use. See memmap_dualcore_stack.ld / petewarden.com/2024/01/16.
if(SPELLING_TINY_MULTICORE)
  target_link_libraries(moonshine_micro_i2s_audio_test pico_multicore)
  pico_set_linker_script(moonshine_micro_i2s_audio_test
    ${CMAKE_CURRENT_SOURCE_DIR}/memmap_dualcore_stack.ld)
endif()
target_compile_definitions(moonshine_micro_i2s_audio_test PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  # heap holds the two kissfft plans (~20 KiB) + G2P string transients
  PICO_HEAP_SIZE=49152
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_i2s_audio_test  1)
pico_enable_stdio_uart(moonshine_micro_i2s_audio_test 0)
pico_add_extra_outputs(moonshine_micro_i2s_audio_test)
message(STATUS "examples/rp2350: + moonshine_micro_i2s_audio_test (I2S audio out)")

# ----------------------------------------------------------------------------
# App 5b: dumb PCM -> I2S relay (moonshine_micro_i2s_relay). Device half of the
# hardware-in-the-loop TTS eval harness (scripts/hw_tts_eval): the laptop
# streams synthesized PCM over USB CDC and this app plays it out the MAX98357A
# via the same I2sAudioOutput path the live apps use -- no synthesis, VAD or STT
# on the device, so the loop measures only the real acoustic path. See
# src/main_i2s_relay.cc for the PING / PLAY protocol.
add_executable(moonshine_micro_i2s_relay
  src/main_i2s_relay.cc
  src/i2s_audio_out.cc)
pico_generate_pio_header(moonshine_micro_i2s_relay
  ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_out.pio)
target_include_directories(moonshine_micro_i2s_relay PRIVATE src generated)
target_link_libraries(moonshine_micro_i2s_relay
  pico_stdlib
  hardware_pio
  hardware_dma
  hardware_clocks
  hardware_vreg
  hardware_watchdog
)
target_compile_definitions(moonshine_micro_i2s_relay PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  # binary PCM streams in over CDC: LF->CRLF would corrupt 0x0A sample bytes
  PICO_STDIO_DEFAULT_CRLF=0
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_i2s_relay  1)
pico_enable_stdio_uart(moonshine_micro_i2s_relay 0)
pico_add_extra_outputs(moonshine_micro_i2s_relay)
message(STATUS "examples/rp2350: + moonshine_micro_i2s_relay (PCM->I2S relay)")

# ----------------------------------------------------------------------------
# App 6: mic -> speaker loopback. Records 5 s from the I2S mic, plays it on the
# I2S amp, repeats forever (bring-up / wiring sanity check).
add_executable(moonshine_micro_audio_loopback_test
  src/main_audio_loopback_test.cc
  src/i2s_audio_io.cc
  src/i2s_audio_out.cc
  src/i2s_mic_process.cc)
pico_generate_pio_header(moonshine_micro_audio_loopback_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_mic.pio)
pico_generate_pio_header(moonshine_micro_audio_loopback_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/i2s_out.pio)
target_include_directories(moonshine_micro_audio_loopback_test PRIVATE src)
target_link_libraries(moonshine_micro_audio_loopback_test
  pico_stdlib
  hardware_pio
  hardware_dma
  hardware_clocks
  hardware_vreg
)
target_compile_definitions(moonshine_micro_audio_loopback_test PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_HEAP_SIZE=24576
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_audio_loopback_test  1)
pico_enable_stdio_uart(moonshine_micro_audio_loopback_test 0)
pico_add_extra_outputs(moonshine_micro_audio_loopback_test)
message(STATUS "examples/rp2350: + moonshine_micro_audio_loopback_test (mic/speaker loopback)")

# ----------------------------------------------------------------------------
# App 6b: standalone neural TTS service. Boots straight into the USB speak
# loop (tts_service.cc, same one the test sweep can embed): SPEAK/IPA in,
# framed int16 PCM out. Drive it with scripts/tts_speak.py. Self-contained --
# no VAD/STT models, so the ~1.8 MB voice pack fits with plenty of margin.
add_executable(moonshine_micro_tts
  src/main_tts.cc
  src/tts_service.cc)
target_include_directories(moonshine_micro_tts PRIVATE src generated)
spelling_configure_neural_tts(moonshine_micro_tts)
target_link_libraries(moonshine_micro_tts
  pico_stdlib
  hardware_clocks
  hardware_vreg
  hardware_watchdog
)

# The decoder Invoke() runs the dual-core SIMD GEMM (compiled into pico-tflmicro
# under SPELLING_TINY_MULTICORE), which lazily launches core 1 on its own
# SCRATCH_X stack. A deep-enough decode call chain can overflow core 0's default
# SCRATCH_Y stack into that live core-1 stack and corrupt it. This service's
# Synthesize() runs from a shallow frame so it doesn't trip today, but swap the
# per-core stacks anyway so a core-0 overflow spills into unused RAM -- same
# guard the audio apps use. See memmap_dualcore_stack.ld / petewarden.com/2024/01/16.
if(SPELLING_TINY_MULTICORE)
  target_link_libraries(moonshine_micro_tts pico_multicore)
  pico_set_linker_script(moonshine_micro_tts
    ${CMAKE_CURRENT_SOURCE_DIR}/memmap_dualcore_stack.ld)
endif()
target_compile_options(moonshine_micro_tts PRIVATE
  -Os
  -ffunction-sections
  -fdata-sections
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
  # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
  $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
)
target_compile_definitions(moonshine_micro_tts PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  # binary PCM goes over CDC: LF->CRLF would corrupt 0x0A sample bytes
  PICO_STDIO_DEFAULT_CRLF=0
  # heap holds the two kissfft plans (~20 KiB) + G2P string transients
  PICO_HEAP_SIZE=49152
  # NB: the per-core stack is a single 4 KiB scratch bank and CANNOT be
  # raised; every multi-KB scratch buffer in the synth pipeline lives in
  # the caller arena instead (see neural_tts.cc's Bump allocations).
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_tts  1)
pico_enable_stdio_uart(moonshine_micro_tts 0)
pico_add_extra_outputs(moonshine_micro_tts)
message(STATUS "examples/rp2350: + moonshine_micro_tts (neural TTS speak service)")

# ----------------------------------------------------------------------------
# Minimal bring-up steps: separate tiny apps, one new thing per step.
# Step 1: blink the LED, nothing else.
add_executable(moonshine_micro_step1_blinky src/main_step1_blinky.cc)
target_link_libraries(moonshine_micro_step1_blinky pico_stdlib)
pico_enable_stdio_usb(moonshine_micro_step1_blinky  0)
pico_enable_stdio_uart(moonshine_micro_step1_blinky 0)
pico_add_extra_outputs(moonshine_micro_step1_blinky)
message(STATUS "examples/rp2350: + moonshine_micro_step1_blinky")

# Step 1b: LED blink for Pico 2 W (LED is on the CYW43 radio there).
# Only builds when the configured board has the radio (PICO_BOARD=pico2_w).
if(PICO_CYW43_SUPPORTED)
  add_executable(moonshine_micro_step1_blinky_w src/main_step1_blinky_w.cc)
  target_link_libraries(moonshine_micro_step1_blinky_w
    pico_stdlib
    pico_cyw43_arch_none)
  pico_enable_stdio_usb(moonshine_micro_step1_blinky_w  0)
  pico_enable_stdio_uart(moonshine_micro_step1_blinky_w 0)
  pico_add_extra_outputs(moonshine_micro_step1_blinky_w)
  message(STATUS "examples/rp2350: + moonshine_micro_step1_blinky_w")
endif()

# Step 2: blinky + one printf per second over USB CDC.
add_executable(moonshine_micro_step2_printf src/main_step2_printf.cc)
target_link_libraries(moonshine_micro_step2_printf pico_stdlib)
if(PICO_CYW43_SUPPORTED)
  target_link_libraries(moonshine_micro_step2_printf pico_cyw43_arch_none)
endif()
target_compile_definitions(moonshine_micro_step2_printf PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
)
pico_enable_stdio_usb(moonshine_micro_step2_printf  1)
pico_enable_stdio_uart(moonshine_micro_step2_printf 0)
pico_add_extra_outputs(moonshine_micro_step2_printf)
message(STATUS "examples/rp2350: + moonshine_micro_step2_printf")

# Step 3: step 2 + heap probe + kissfft plans + one FFT per tick.
add_executable(moonshine_micro_step3_fft src/main_step3_fft.cc)
target_link_libraries(moonshine_micro_step3_fft pico_stdlib kissfft)
if(PICO_CYW43_SUPPORTED)
  target_link_libraries(moonshine_micro_step3_fft pico_cyw43_arch_none)
endif()
target_compile_definitions(moonshine_micro_step3_fft PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
)
pico_enable_stdio_usb(moonshine_micro_step3_fft  1)
pico_enable_stdio_uart(moonshine_micro_step3_fft 0)
pico_add_extra_outputs(moonshine_micro_step3_fft)
message(STATUS "examples/rp2350: + moonshine_micro_step3_fft")

# Step 4: same code as step 3, but with the TTS app's custom memory
# config (small stack, fixed heap). Isolates whether that config is what
# broke kissfft allocation in the full app.
add_executable(moonshine_micro_step4_fft_smallheap src/main_step3_fft.cc)
target_link_libraries(moonshine_micro_step4_fft_smallheap pico_stdlib kissfft)
if(PICO_CYW43_SUPPORTED)
  target_link_libraries(moonshine_micro_step4_fft_smallheap pico_cyw43_arch_none)
endif()
target_compile_definitions(moonshine_micro_step4_fft_smallheap PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_STDIO_DEFAULT_CRLF=0
  PICO_HEAP_SIZE=49152
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_step4_fft_smallheap  1)
pico_enable_stdio_uart(moonshine_micro_step4_fft_smallheap 0)
pico_add_extra_outputs(moonshine_micro_step4_fft_smallheap)
message(STATUS "examples/rp2350: + moonshine_micro_step4_fft_smallheap")

# Step 5: step 4 + real WorldLiteSynth construction + FFT self-test/tick.
add_executable(moonshine_micro_step5_synth src/main_step5_synth.cc)
target_link_libraries(moonshine_micro_step5_synth
  pico_stdlib neural_tts kissfft)
if(PICO_CYW43_SUPPORTED)
  target_link_libraries(moonshine_micro_step5_synth pico_cyw43_arch_none)
endif()
target_compile_definitions(moonshine_micro_step5_synth PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_STDIO_DEFAULT_CRLF=0
  PICO_HEAP_SIZE=49152
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_step5_synth  1)
pico_enable_stdio_uart(moonshine_micro_step5_synth 0)
pico_add_extra_outputs(moonshine_micro_step5_synth)
message(STATUS "examples/rp2350: + moonshine_micro_step5_synth")

# Step 6: step 5 + PbDecoder (TFLM) + one tile decode per tick.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_demo_data.cc")
  add_executable(moonshine_micro_step6_decoder
    src/main_step6_decoder.cc
    generated/neural_tts_demo_data.cc)
  target_include_directories(moonshine_micro_step6_decoder PRIVATE generated)
  target_link_libraries(moonshine_micro_step6_decoder
    pico_stdlib neural_tts pico-tflmicro kissfft
    hardware_clocks hardware_vreg)
  if(PICO_CYW43_SUPPORTED)
    target_link_libraries(moonshine_micro_step6_decoder pico_cyw43_arch_none)
  endif()
  target_compile_options(moonshine_micro_step6_decoder PRIVATE
    -Os -ffunction-sections -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_step6_decoder PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
  )
  pico_enable_stdio_usb(moonshine_micro_step6_decoder  1)
  pico_enable_stdio_uart(moonshine_micro_step6_decoder 0)
  pico_add_extra_outputs(moonshine_micro_step6_decoder)
  message(STATUS "examples/rp2350: + moonshine_micro_step6_decoder")

  # Step 7: step 6 + full synthesis per utterance, PCM discarded.
  add_executable(moonshine_micro_step7_synthesize
    src/main_step7_synthesize.cc
    generated/neural_tts_demo_data.cc)
  target_include_directories(moonshine_micro_step7_synthesize PRIVATE generated)
  target_link_libraries(moonshine_micro_step7_synthesize
    pico_stdlib neural_tts pico-tflmicro kissfft hardware_watchdog)
  if(PICO_CYW43_SUPPORTED)
    target_link_libraries(moonshine_micro_step7_synthesize pico_cyw43_arch_none)
  endif()
  target_compile_options(moonshine_micro_step7_synthesize PRIVATE
    -Os -ffunction-sections -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_step7_synthesize PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
  )
  pico_enable_stdio_usb(moonshine_micro_step7_synthesize  1)
  pico_enable_stdio_uart(moonshine_micro_step7_synthesize 0)
  pico_add_extra_outputs(moonshine_micro_step7_synthesize)
  message(STATUS "examples/rp2350: + moonshine_micro_step7_synthesize")

  # Step 7b: step 6 + full GetFrame sweep (tile advance), no synthesis.
  add_executable(moonshine_micro_step7b_framesweep
    src/main_step7b_framesweep.cc
    generated/neural_tts_demo_data.cc)
  target_include_directories(moonshine_micro_step7b_framesweep PRIVATE generated)
  target_link_libraries(moonshine_micro_step7b_framesweep
    pico_stdlib neural_tts pico-tflmicro kissfft)
  if(PICO_CYW43_SUPPORTED)
    target_link_libraries(moonshine_micro_step7b_framesweep pico_cyw43_arch_none)
  endif()
  target_compile_options(moonshine_micro_step7b_framesweep PRIVATE
    -Os -ffunction-sections -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_step7b_framesweep PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
  )
  pico_enable_stdio_usb(moonshine_micro_step7b_framesweep  1)
  pico_enable_stdio_uart(moonshine_micro_step7b_framesweep 0)
  pico_add_extra_outputs(moonshine_micro_step7b_framesweep)
  message(STATUS "examples/rp2350: + moonshine_micro_step7b_framesweep")

  # Step 7c: full Synthesize loop on synthetic frames, no decoder/TFLM.
  add_executable(moonshine_micro_step7c_synthonly
    src/main_step7c_synthonly.cc)
  target_link_libraries(moonshine_micro_step7c_synthonly
    pico_stdlib neural_tts kissfft)
  if(PICO_CYW43_SUPPORTED)
    target_link_libraries(moonshine_micro_step7c_synthonly pico_cyw43_arch_none)
  endif()
  target_compile_options(moonshine_micro_step7c_synthonly PRIVATE
    -Os -ffunction-sections -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_step7c_synthonly PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
  )
  pico_enable_stdio_usb(moonshine_micro_step7c_synthonly  1)
  pico_enable_stdio_uart(moonshine_micro_step7c_synthonly 0)
  pico_add_extra_outputs(moonshine_micro_step7c_synthonly)
  message(STATUS "examples/rp2350: + moonshine_micro_step7c_synthonly")
endif()

# ----------------------------------------------------------------------------
# App 6b1: incremental bring-up ladder for the neural TTS stack (see the
# main file for the stage list). Rebuild with -DTTS_LADDER_STAGE=N.
set(TTS_LADDER_STAGE 0 CACHE STRING "bring-up ladder stage (0..5)")
set(TTS_LADDER_EXECUTE 1 CACHE STRING "0 = link stage code but do not run it")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_demo_data.cc")
  add_executable(moonshine_micro_tts_ladder_test
    src/main_tts_ladder_test.cc
    generated/neural_tts_demo_data.cc)
  target_include_directories(moonshine_micro_tts_ladder_test PRIVATE src generated)
  target_link_libraries(moonshine_micro_tts_ladder_test
    neural_tts
    pico-tflmicro
    kissfft
    pico_stdlib
    hardware_watchdog
  )
  target_compile_options(moonshine_micro_tts_ladder_test PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_tts_ladder_test PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
    TTS_LADDER_STAGE=${TTS_LADDER_STAGE}
    TTS_LADDER_EXECUTE=${TTS_LADDER_EXECUTE}
  )
  pico_enable_stdio_usb(moonshine_micro_tts_ladder_test  1)
  pico_enable_stdio_uart(moonshine_micro_tts_ladder_test 0)
  pico_add_extra_outputs(moonshine_micro_tts_ladder_test)
  message(STATUS "examples/rp2350: + moonshine_micro_tts_ladder_test (stage ${TTS_LADDER_STAGE})")
endif()

# ----------------------------------------------------------------------------
# App 6b2: rung-2 bring-up target -- banner + TFLM init + ONE Invoke of the
# neural-TTS decoder model with a blocking serial print around every op.
# Exists to name the kernel that hangs inside the TTS app's first Invoke.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/neural_tts_demo_data.cc")
  add_executable(moonshine_micro_tflm_invoke_test
    src/main_tflm_invoke_test.cc
    generated/neural_tts_demo_data.cc)
  target_include_directories(moonshine_micro_tflm_invoke_test PRIVATE src generated)
  target_link_libraries(moonshine_micro_tflm_invoke_test
    pico-tflmicro
    pico_stdlib
    hardware_watchdog
    hardware_clocks
    hardware_vreg
  )
  target_compile_options(moonshine_micro_tflm_invoke_test PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    # No __cxa_atexit/__dso_handle for static dtors (see micro/CMakeLists.txt).
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
  )
  target_compile_definitions(moonshine_micro_tflm_invoke_test PRIVATE
    PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
    PICO_STDIO_DEFAULT_CRLF=0
    PICO_HEAP_SIZE=49152
    PICO_STACK_SIZE=4096
  )
  pico_enable_stdio_usb(moonshine_micro_tflm_invoke_test  1)
  pico_enable_stdio_uart(moonshine_micro_tflm_invoke_test 0)
  pico_add_extra_outputs(moonshine_micro_tflm_invoke_test)
  message(STATUS "examples/rp2350: + moonshine_micro_tflm_invoke_test (rung-2 op bisect)")
endif()

# ----------------------------------------------------------------------------
# App 6c: minimal USB CDC stdio soak test -- one printf per second, forever,
# with a hard-fault trap + watchdog. Used to isolate the neural-TTS "USB goes
# mute after a few seconds" wedge: if THIS target also dies, the problem is in
# the stdio/USB/build-config layer, not the TTS code.
add_executable(moonshine_micro_usb_banner_test
  src/main_usb_banner_test.cc)
target_link_libraries(moonshine_micro_usb_banner_test
  pico_stdlib
  hardware_watchdog
)
target_compile_definitions(moonshine_micro_usb_banner_test PRIVATE
  PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1
  PICO_STDIO_DEFAULT_CRLF=0
  PICO_HEAP_SIZE=24576
  PICO_STACK_SIZE=4096
)
pico_enable_stdio_usb(moonshine_micro_usb_banner_test  1)
pico_enable_stdio_uart(moonshine_micro_usb_banner_test 0)
pico_add_extra_outputs(moonshine_micro_usb_banner_test)
message(STATUS "examples/rp2350: + moonshine_micro_usb_banner_test (CDC stdio soak)")
