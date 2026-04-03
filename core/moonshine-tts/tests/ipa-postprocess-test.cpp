#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "ipa-postprocess.h"

#include <unordered_set>

using namespace moonshine_tts;

TEST_CASE("levenshtein_distance") {
  CHECK(levenshtein_distance({}, {}) == 0);
  CHECK(levenshtein_distance({"a"}, {}) == 1);
  CHECK(levenshtein_distance({"a"}, {"a"}) == 0);
  CHECK(levenshtein_distance({"a", "b"}, {"a", "c"}) == 1);
}

TEST_CASE("pick_closest_cmudict_ipa single") {
  const std::vector<std::string> pred = {"x"};
  CHECK(pick_closest_cmudict_ipa(pred, {"only"}, 0) == "only");
}

TEST_CASE("match_prediction_to_cmudict_ipa") {
  const std::vector<std::string> alts = {"həˈloʊ", "həˈləʊ"};
  const auto m = match_prediction_to_cmudict_ipa("həˈloʊ", alts);
  REQUIRE(m);
  CHECK(*m == "həˈloʊ");
}

TEST_CASE("normalize_g2p_ipa_for_piper_engines") {
  // ``həlˈoʊ wˈɝld`` → ``həlˈoʊ wˈɜːld`` (UTF-8 bytes; stress U+02C8, length U+02D0).
  const std::string hello_world_in =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9D", 2) + "ld";
  const std::string hello_world_out =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9C\xCB\x90", 4) + "ld";  // ɜ + ː
  CHECK(normalize_g2p_ipa_for_piper_engines(hello_world_in) == hello_world_out);
  CHECK(normalize_g2p_ipa_for_piper_engines("t\xCA\x83\xCB\x88\xC9\x9Dt\xCA\x83") ==
        "t\xCA\x83\xCB\x88\xC9\x9C\xCB\x90t\xCA\x83");  // tʃˈɝtʃ → tʃˈɜːtʃ
  CHECK(normalize_g2p_ipa_for_piper_engines("") == "");
  CHECK(normalize_g2p_ipa_for_piper_engines("\xC9\x99") == "\xC9\x99");
}

TEST_CASE("normalize_g2p_ipa_for_piper NFC plus shared rules") {
  const std::string hello_world_in =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9D", 2) + "ld";
  const std::string hello_world_out =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9C\xCB\x90", 4) + "ld";
  CHECK(normalize_g2p_ipa_for_piper(hello_world_in, "en_us") == hello_world_out);
}

TEST_CASE("coerce_unknown_ipa_chars_to_piper_inventory toy map") {
  std::unordered_set<std::string> keys{"a", "z"};
  // U+03B1 Greek alpha — not rewritten by shared G2P→Piper rules; closest in {a,z} is z.
  CHECK(coerce_unknown_ipa_chars_to_piper_inventory("\xCE\xB1", keys, true) == "z");
  std::unordered_set<std::string> keys2{"h"};
  // U+0300 combining grave after h: not in map, category Mn → dropped.
  const std::string in = "h" + std::string("\xCC\x80", 2);
  CHECK(coerce_unknown_ipa_chars_to_piper_inventory(in, keys2, true) == "h");
}

TEST_CASE("ipa_to_piper_ready without coercion") {
  const std::string hello_world_in =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9D", 2) + "ld";
  const std::string hello_world_out =
      "h\xC9\x99l\xCB\x88o\xCA\x8A w\xCB\x88" + std::string("\xC9\x9C\xCB\x90", 4) + "ld";
  std::unordered_set<std::string> keys{"h", "ə", "l", "ˈ", "o", "ʊ", " ", "w", "ɜ", "ː", "d"};
  CHECK(ipa_to_piper_ready(hello_world_in, "en_us", keys, false) == hello_world_out);
}

TEST_CASE("normalize_g2p_ipa_for_piper Korean rule IPA toward eSpeak-ng") {
  // G2P for 안녕하세요 / 여보세요 (moonshine-tts-g2p --language ko).
  const std::string annyeong =
      std::string("an\xc9\xb2j\xca\x8c\xc5\x8bhas") + "\xca\xb0" + "ejo";
  const std::string annyeong_es =
      std::string("\xcb\x88\xc9\x90nnj\xca\x8c\xc5\x8bh\xcb\x8c\xc9\x90sej\xcb\x8c") + "o";
  CHECK(normalize_g2p_ipa_for_piper(annyeong, "ko") == annyeong_es);
  CHECK(normalize_g2p_ipa_for_piper(annyeong, "ko_kr") == annyeong_es);

  const std::string yeobo = std::string("j\xca\x8c" "bos") + "\xca\xb0" + "ejo";
  const std::string yeobo_es = std::string("j\xcb\x88\xca\x8c" "bos\xcb\x8c") + "ejo";
  CHECK(normalize_g2p_ipa_for_piper(yeobo, "ko") == yeobo_es);
}
