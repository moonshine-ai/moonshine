#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "korean-numbers.h"
#include "korean.h"
#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
#include "moonshine-g2p.h"
#endif

#include "rule-g2p-test-support.h"

#include <filesystem>
#include <string>

namespace r = moonshine_tts::rule_g2p_test;

namespace {

std::filesystem::path ko_dict_path() {
  return r::moonshine_tts_bundled_data_dir_relative() / "ko" / "dict.tsv";
}

}  // namespace

TEST_CASE("korean: dialect_resolves_to_korean_rules") {
  using moonshine_tts::dialect_resolves_to_korean_rules;
  CHECK(dialect_resolves_to_korean_rules("ko"));
  CHECK(dialect_resolves_to_korean_rules("ko-KR"));
  CHECK(dialect_resolves_to_korean_rules("KO_kr"));
  CHECK(dialect_resolves_to_korean_rules("korean"));
  CHECK_FALSE(dialect_resolves_to_korean_rules("ja"));
}

TEST_CASE("korean: normalize strips all combining marks including tense and unreleased") {
  using moonshine_tts::KoreanRuleG2p;
  // ha̠ + k̚ + k͈jo — NFD has combining on first vowel; all MN chars stripped (U+0320, U+031A, U+0348).
  const std::string in =
      "ha"
      "\xCC\xA0"
      "k"
      "\xCC\x9A"
      "k"
      "\xCD\x88"
      "jo";
  CHECK(KoreanRuleG2p::normalize_korean_ipa(in) == "hakkjo");  // all combining marks stripped
  CHECK(KoreanRuleG2p::normalize_korean_ipa("ku\xC5\x8Bmu\xC9\xAD") == "ku\xC5\x8Bmu\xC9\xAB");  // ɭ → ɫ
}

TEST_CASE("korean: int_to_sino_korean_hangul") {
  using moonshine_tts::int_to_sino_korean_hangul;
  CHECK(int_to_sino_korean_hangul(0) == "영");
  CHECK(int_to_sino_korean_hangul(10) == "십");
  CHECK(int_to_sino_korean_hangul(42) == "사십이");
  CHECK(int_to_sino_korean_hangul(105) == "백오");
  CHECK(int_to_sino_korean_hangul(1234) == "천이백삼십사");
  CHECK(int_to_sino_korean_hangul(10000) == "만");
  CHECK(int_to_sino_korean_hangul(100010000) == "일억만");
  CHECK(int_to_sino_korean_hangul(102030400) == "일억이백삼만사백");
}

TEST_CASE("korean: korean_reading_fragments_from_ascii_numeral_token") {
  using moonshine_tts::korean_reading_fragments_from_ascii_numeral_token;
  const auto a = korean_reading_fragments_from_ascii_numeral_token("1,234");
  REQUIRE(a.has_value());
  CHECK((*a).size() == 1);
  CHECK((*a)[0] == "천이백삼십사");
  const auto b = korean_reading_fragments_from_ascii_numeral_token("3.14");
  REQUIRE(b.has_value());
  CHECK((*b)[0] == "삼점일사");
  const auto c = korean_reading_fragments_from_ascii_numeral_token("3,14");
  REQUIRE(c.has_value());
  CHECK((*c)[0] == "삼점일사");
  const auto d = korean_reading_fragments_from_ascii_numeral_token("-10");
  REQUIRE(d.has_value());
  CHECK((*d).size() == 2);
  CHECK((*d)[0] == "마이너스");
  CHECK((*d)[1] == "십");
  const auto e = korean_reading_fragments_from_ascii_numeral_token("007");
  REQUIRE(e.has_value());
  CHECK((*e)[0] == "영영칠");
  CHECK_FALSE(korean_reading_fragments_from_ascii_numeral_token("12a").has_value());
}

TEST_CASE("korean: G2P examples with data/ko/dict.tsv") {
  const auto dict = ko_dict_path();
  if (!std::filesystem::is_regular_file(dict)) {
    return;
  }
  moonshine_tts::KoreanRuleG2p g(dict);
  // ˈ = U+02C8 (CB 88), ɫ = U+026B (C9 AB), ɾ = U+027E (C9 BE)
  CHECK(g.text_to_ipa("\xEB\x8B\xAD\xEC\x9D\xB4") == "\xCB\x88""da\xC9\xABki");        // 닭이 → ˈdaɫki
  CHECK(g.text_to_ipa("\xEB\x8B\xAB\xEB\x8A\x94") == "\xCB\x88""dann\xC9\xAF""n");     // 닫는 → ˈdannɯn
  CHECK(g.text_to_ipa("007") == "\xCB\x88j\xCA\x8C\xC5\x8Bj\xCA\x8C\xC5\x8Bt\xCA\x83hi\xC9\xAB");  // ˈjʌŋjʌŋtʃhiɫ
  CHECK(g.text_to_ipa("3.14") == "\xCB\x88""samt\xCA\x83\xCA\x8Cmi\xC9\xABs\xC9\x90");  // ˈsamtʃʌmiɫsɐ
  moonshine_tts::KoreanRuleG2p::Options no_dig;
  no_dig.expand_cardinal_digits = false;
  moonshine_tts::KoreanRuleG2p g2(dict, no_dig);
  CHECK(g2.text_to_ipa("42") == "");
}

#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
TEST_CASE("korean: MoonshineG2P ko uses KoreanRuleG2p") {
  const auto dict = ko_dict_path();
  if (!std::filesystem::is_regular_file(dict)) {
    return;
  }
  moonshine_tts::MoonshineG2POptions opt;
  opt.files.set_path(moonshine_tts::kG2pKoreanDictKey, dict);
  moonshine_tts::MoonshineG2P g("ko", opt);
  CHECK(g.uses_korean_rules());
  CHECK_FALSE(g.uses_onnx());
  CHECK(g.dialect_id() == "ko-KR");
  moonshine_tts::KoreanRuleG2p direct(dict);
  const std::string w = "닭이";
  CHECK(g.text_to_ipa(w) == direct.text_to_ipa(w));
}
#endif
