# kissfft -- minimal vendored subset for kiss_fftr (single-precision real FFT).
#
# Built as a small static library so every module (feature-generation, klatt-tts) and
# example can share one copy and one float-vs-double choice. We compile with
# kiss_fft_scalar=float to keep parity with the desktop cpp/ build.

add_library(kissfft STATIC
  kiss_fft.c
  kiss_fftr.c
)

target_include_directories(kissfft PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=float)
