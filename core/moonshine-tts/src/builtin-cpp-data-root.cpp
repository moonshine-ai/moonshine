#include "builtin-cpp-data-root.h"

namespace moonshine_tts {

std::filesystem::path builtin_cpp_data_root() {
  const std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
  const std::filesystem::path cpp_dir = here.parent_path();
  return cpp_dir / "data";
}

}  // namespace moonshine_tts
