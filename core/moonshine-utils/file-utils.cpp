#include "file-utils.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include "debug-utils.h"

std::size_t fread_exact(void *ptr, std::size_t size, std::size_t count,
                        std::FILE *stream, const char *what) {
  if (count == 0 || size == 0) {
    return count;
  }
  const std::size_t read_count = std::fread(ptr, size, count, stream);
  if (read_count != count) {
    std::ostringstream oss;
    oss << "Failed to read " << (what ? what : "file") << ": expected "
        << count << " element(s) of size " << size << " but read only "
        << read_count;
    if (stream != nullptr && std::feof(stream)) {
      oss << " (unexpected end of file)";
    } else if (stream != nullptr && std::ferror(stream)) {
      oss << " (I/O error)";
    }
    const std::string message = oss.str();
    THROW_WITH_LOG(message.c_str());
  }
  return read_count;
}
