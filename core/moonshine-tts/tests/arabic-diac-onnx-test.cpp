#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "arabic-diac-onnx.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>

#include "rule-g2p-test-support.h"

namespace r = moonshine_tts::rule_g2p_test;

namespace {

/// Undiacritized MSA covering a range of lengths, including sequences shorter
/// than the padding threshold the split form relies on.
const std::vector<std::string>& sample_sentences() {
  static const std::vector<std::string> kSentences = {
      "مرحبا",
      "ذهب الولد إلى المدرسة",
      "الطقس اليوم جميل جدا",
      "كتب الطالب الدرس في دفتره الجديد",
      "تقع مدينة القاهرة على ضفاف نهر النيل وهي عاصمة مصر",
  };
  return kSentences;
}

}  // namespace

TEST_CASE("arabic diac onnx: diacritized output matches reference file") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = r::moonshine_tts_bundled_data_dir_relative() / "ar_msa" /
                     "arabertv02_tashkeel_fadel_onnx";
  const std::filesystem::path golden =
      r::tests_data_dir(repo) / "ar" / "diac_onnx_samples.txt";
  if (!r::model_present(model)) {
    return;
  }

  moonshine_tts::ArabicDiacOnnx diac(model, false);
  // A split pair on disk must actually be used. Without this the test would
  // still pass after a silent fall back to a single-file model.
  CHECK(diac.uses_split_weights() == r::model_ships_split(model));

  std::vector<std::string> actual;
  actual.reserve(sample_sentences().size());
  for (const std::string& text : sample_sentences()) {
    actual.push_back(diac.diacritize(text));
  }

  // Regenerate with ARABIC_DIAC_UPDATE_GOLDEN=1 after an intentional model
  // change, and commit the result.
  if (std::getenv("ARABIC_DIAC_UPDATE_GOLDEN") != nullptr) {
    std::filesystem::create_directories(golden.parent_path());
    std::ofstream out(golden, std::ios::binary);
    for (const std::string& line : actual) {
      out << line << "\n";
    }
    MESSAGE("wrote " << golden.string());
    return;
  }

  if (!std::filesystem::is_regular_file(golden)) {
    return;
  }
  const std::vector<std::string> expected = r::load_ref_lines(golden);
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    INFO("sentence " << (i + 1));
    CHECK(actual[i] == expected[i]);
  }
}
