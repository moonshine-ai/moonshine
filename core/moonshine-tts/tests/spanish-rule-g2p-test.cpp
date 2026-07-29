#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "rule-g2p-test-support.h"
#include "spanish.h"

namespace r = moonshine_tts::rule_g2p_test;

namespace {

void check_wiki_parity(const std::filesystem::path& wiki,
                       const std::filesystem::path& golden,
                       const moonshine_tts::SpanishDialect& dialect) {
  constexpr std::size_t kWikiParityLines = 100;
  if (!std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(moonshine_tts::spanish_text_to_ipa(src[i], dialect) == py[i]);
  }
}

}  // namespace

TEST_CASE("spanish: tap and trill are distinguished in every context") {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"pero", "pˈeɾo"},
      {"perro", "pˈero"},
      {"caro", "kˈaɾo"},
      {"carro", "kˈaro"},
      {"oro", "ˈoɾo"},
      {"para", "pˈaɾa"},
      {"honra", "ˈonra"},
      {"Israel", "isrˈael"},
      {"enredo", "enrˈeðo"},
      {"sonrisa", "sonrˈisa"},
      {"alrededor", "alreðedˈoɾ"},
      {"rosa", "rˈosa"},
      {"por", "pˈoɾ"},
      {"arte", "ˈaɾte"},
      {"tren", "tɾˈen"},
      {"tres", "tɾˈes"},
  };
  for (const char* id : {"es-ES", "es-MX"}) {
    const auto dialect = moonshine_tts::spanish_dialect_from_cli_id(id);
    for (const auto& [word, expected] : cases) {
      INFO(id << " " << word);
      CHECK(moonshine_tts::spanish_text_to_ipa(word, dialect) == expected);
    }
  }
}

TEST_CASE("spanish: rising diphthongs use glides and stress the nucleus") {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"sonrió", "sonrjˈo"},
      {"sonrío", "sonrˈio"},
      {"salió", "saljˈo"},
      {"comió", "komjˈo"},
      {"cambio", "kˈambjo"},
      {"serio", "sˈeɾjo"},
      {"radio", "rˈaðjo"},
      {"patio", "pˈatjo"},
      {"limpio", "lˈimpjo"},
      {"cuando", "kwˈando"},
      {"pueblo", "pwˈeblo"},
      {"bueno", "bwˈeno"},
      {"fuerte", "fwˈeɾte"},
      {"agua", "ˈaɣwa"},
      {"cuota", "kwˈota"},
      {"cuidar", "kwidˈaɾ"},
      {"antiguo", "antˈiɣwo"},
      {"viuda", "bjˈuða"},
      {"hielo", "jˈelo"},
      {"huevo", "wˈeβo"},
      {"efectuará", "efektwaɾˈa"},
      {"continuó", "kontinwˈo"},
      {"río", "rˈio"},
      {"mío", "mˈio"},
      {"día", "dˈia"},
      {"país", "paˈis"},
      {"queso", "kˈeso"},
      {"guerra", "ɡˈera"},
      {"guitarra", "ɡitˈara"},
      {"sigue", "sˈiɣe"},
      {"pingüino", "pinɡwˈino"},
      {"averigüe", "aβeɾˈiɣwe"},
  };
  for (const char* id : {"es-ES", "es-MX"}) {
    const auto dialect = moonshine_tts::spanish_dialect_from_cli_id(id);
    for (const auto& [word, expected] : cases) {
      INFO(id << " " << word);
      CHECK(moonshine_tts::spanish_text_to_ipa(word, dialect) == expected);
    }
  }
}

TEST_CASE("spanish: seseo and distincion keep glides in -cion words") {
  const auto es = moonshine_tts::spanish_dialect_from_cli_id("es-ES");
  const auto mx = moonshine_tts::spanish_dialect_from_cli_id("es-MX");
  CHECK(moonshine_tts::spanish_text_to_ipa("nación", es) == "naθjˈon");
  CHECK(moonshine_tts::spanish_text_to_ipa("nación", mx) == "nasjˈon");
  CHECK(moonshine_tts::spanish_text_to_ipa("novecientos", es) ==
        "noβeθjˈentos");
  CHECK(moonshine_tts::spanish_text_to_ipa("novecientos", mx) ==
        "noβesjˈentos");
  CHECK(moonshine_tts::spanish_text_to_ipa("cincuenta", es) == "θinkwˈenta");
  CHECK(moonshine_tts::spanish_text_to_ipa("cincuenta", mx) == "sinkwˈenta");
  CHECK(moonshine_tts::spanish_text_to_ipa("ciudad", es) == "θjudˈad");
  CHECK(moonshine_tts::spanish_text_to_ipa("ciudad", mx) == "sjudˈad");
}

TEST_CASE("spanish: En 1891 matches reference IPA when golden exists") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path golden =
      r::tests_data_dir(repo) / "es_mx" / "rule_g2p_en_1891.txt";
  if (!std::filesystem::is_regular_file(golden)) {
    return;
  }
  const std::string expected = r::load_ref_text_trimmed(golden);
  const auto dialect = moonshine_tts::spanish_dialect_from_cli_id("es-MX");
  CHECK(moonshine_tts::spanish_text_to_ipa("En 1891", dialect) == expected);
}

TEST_CASE("spanish: dialect ids include es-MX and es-ES") {
  const auto ids = moonshine_tts::spanish_dialect_cli_ids();
  CHECK(std::find(ids.begin(), ids.end(), "es-MX") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "es-ES") != ids.end());
}

TEST_CASE(
    "spanish: wiki-text first 100 lines es_mx match reference IPA when data "
    "and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path wiki =
      r::moonshine_tts_bundled_data_dir_relative() / "es_mx" / "wiki-text.txt";
  const std::filesystem::path golden =
      r::tests_data_dir(repo) / "es_mx" / "rule_g2p_wiki_100.txt";
  const auto dialect = moonshine_tts::spanish_dialect_from_cli_id("es-MX");
  check_wiki_parity(wiki, golden, dialect);
}

TEST_CASE(
    "spanish: wiki-text first 100 lines es_es match reference IPA when data "
    "and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path wiki =
      r::moonshine_tts_bundled_data_dir_relative() / "es_es" / "wiki-text.txt";
  const std::filesystem::path golden =
      r::tests_data_dir(repo) / "es_es" / "rule_g2p_wiki_100.txt";
  const auto dialect = moonshine_tts::spanish_dialect_from_cli_id("es-ES");
  check_wiki_parity(wiki, golden, dialect);
}
