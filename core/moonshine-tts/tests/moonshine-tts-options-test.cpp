#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-tts-options.h"

#include <string>
#include <utility>
#include <vector>

using moonshine_tts::MoonshineTTSOptions;

TEST_CASE("MoonshineTTSOptions parse_options ort_providers") {
  MoonshineTTSOptions opt;
  std::string lang;
  bool lang_set = false;
  opt.parse_options({{"ort_providers", "CoreML, CPU"}, {"coreml_cache_dir", "/tmp/cache"}},
                    &lang, &lang_set);
  REQUIRE(opt.ort_provider_names.size() == 2);
  CHECK(opt.ort_provider_names[0] == "coreml");
  CHECK(opt.ort_provider_names[1] == "cpu");
  CHECK(opt.coreml_cache_dir == "/tmp/cache");
}
