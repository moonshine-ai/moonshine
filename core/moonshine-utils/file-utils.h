#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <cstddef>
#include <cstdio>

// Wrapper around std::fread that throws std::runtime_error unless the full
// requested number of elements is read. `what` is included in the error
// message to identify the failing read site (e.g. "WAV fmt chunk"). Returns
// `count` on success. A request for zero elements (count == 0) or zero-sized
// elements (size == 0) is always a no-op that returns `count`.
std::size_t fread_exact(void *ptr, std::size_t size, std::size_t count,
                        std::FILE *stream, const char *what = "file");

#endif
