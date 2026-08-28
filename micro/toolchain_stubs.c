// Bare-metal fallbacks for Arm GNU libstdc++ when crtbegin.o is not linked.
//
// Newer Pico SDK versions pass -nostartfiles for GCC (see raspberrypi/pico-sdk
// #2979), which drops crtbegin.o and its __dso_handle. libstdc++ still
// references that symbol from static constructors (ios_errcat, system_error),
// and some newlib paths call _fini. Both symbols are weak so SDK 2.2.0, which
// still links crtbegin.o, keeps the toolchain definitions (GitHub issue #204).

void *__dso_handle __attribute__((weak, visibility("hidden")));

void __attribute__((weak)) _fini(void) {}
