#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vietnamese.h"
#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
#include "moonshine-g2p.h"
#endif

#include <filesystem>
#include <string>

namespace {

std::filesystem::path repo_root() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

std::filesystem::path vi_dict_path() {
  return repo_root() / "data" / "vi" / "dict.tsv";
}

}  // namespace

TEST_CASE("vietnamese: dialect_resolves_to_vietnamese_rules") {
  using moonshine_tts::dialect_resolves_to_vietnamese_rules;
  CHECK(dialect_resolves_to_vietnamese_rules("vi"));
  CHECK(dialect_resolves_to_vietnamese_rules("vi-VN"));
  CHECK(dialect_resolves_to_vietnamese_rules("VI_vn"));
  CHECK(dialect_resolves_to_vietnamese_rules("vietnamese"));
  CHECK_FALSE(dialect_resolves_to_vietnamese_rules("ko"));
}

TEST_CASE("vietnamese: syllable OOV parity with Python samples") {
  using moonshine_tts::VietnameseRuleG2p;
  CHECK(VietnameseRuleG2p::syllable_to_ipa("tra") == "ca\xCB\xA7\xCB\xA7");
  CHECK(VietnameseRuleG2p::syllable_to_ipa("thành") ==
        "t\xCA\xB0\xC9\x9B\xC5\x8B\xCB\xA7\xCB\xA8");
  CHECK(VietnameseRuleG2p::syllable_to_ipa("những") ==
        "\xC9\xB2\xC9\xAF\xC5\x8B\xCB\xA7\xCB\x80\xCB\xA5");
}

TEST_CASE("vietnamese: lexicon line with data/vi/dict.tsv") {
  const auto dict = vi_dict_path();
  if (!std::filesystem::is_regular_file(dict)) {
    return;
  }
  moonshine_tts::VietnameseRuleG2p g(dict);
  CHECK(g.text_to_ipa("tổ chức") == "to\xCB\xA7\xCB\xA9\xCB\xA8 c\xC9\xAFk\xCB\xA6\xCB\xA5");
  // UTF-8 for U+0111 U+1ED9 (độ) — must hit ``dict.tsv`` not OOV rules.
  const std::string do_nang = "\xc4\x91\xe1\xbb\x99";
  CHECK(g.word_to_ipa(do_nang) == "do\xcb\xa8\xcb\x80\xcb\xa9\xca\x94");
}

#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
TEST_CASE("vietnamese: MoonshineG2P vi-VN") {
  const auto dict = vi_dict_path();
  if (!std::filesystem::is_regular_file(dict)) {
    return;
  }
  moonshine_tts::MoonshineG2POptions opt;
  opt.vietnamese_dict_path = dict;
  moonshine_tts::MoonshineG2P g("vi-VN", std::move(opt));
  CHECK(g.dialect_id() == "vi-VN");
  CHECK(g.uses_vietnamese_rules());
  CHECK(g.text_to_ipa("tổ chức") == "to\xCB\xA7\xCB\xA9\xCB\xA8 c\xC9\xAFk\xCB\xA6\xCB\xA5");
}
#endif
