#include "hindi.h"
#include "hindi-numbers.h"
#include "g2p-word-log.h"
#include "utf8-utils.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <istream>
#include <sstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

constexpr char32_t kVirama = 0x094D;
constexpr char32_t kNukta = 0x093C;
constexpr char32_t kAnusvara = 0x0902;
constexpr char32_t kChandra = 0x0901;
constexpr char32_t kVisarga = 0x0903;
constexpr char32_t kZwj = 0x200D;
constexpr char32_t kZwnj = 0x200C;

std::string utf8_nfc_utf8proc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

struct Syllable {
  std::vector<std::string> onset;
  std::optional<std::string> vowel;
  bool inherent_schwa = false;
  bool chandrabindu = false;
  bool anusvara = false;
  bool visarga = false;
};

const std::unordered_map<char32_t, std::string>& indep_vowel_map() {
  static const std::unordered_map<char32_t, std::string> m{
      {0x0905, "ə"},  {0x0906, "aː"}, {0x0907, "ɪ"},  {0x0908, "iː"}, {0x0909, "ʊ"},
      {0x090A, "uː"}, {0x090F, "eː"}, {0x0910, "ɛː"}, {0x0913, "oː"}, {0x0914, "ɔː"},
  };
  return m;
}

const std::unordered_map<char32_t, std::string>& matra_map() {
  static const std::unordered_map<char32_t, std::string> m{
      {0x093E, "aː"}, {0x093F, "ɪ"},  {0x0940, "iː"}, {0x0941, "ʊ"},  {0x0942, "uː"},
      {0x0947, "eː"}, {0x0948, "ɛː"}, {0x094B, "oː"}, {0x094C, "ɔː"},
  };
  return m;
}

const std::unordered_map<char32_t, std::string>& base_cons_map() {
  static const std::unordered_map<char32_t, std::string> m{
      {0x0915, "k"},   {0x0916, "kʰ"},  {0x0917, "g"},   {0x0918, "gʰ"}, {0x0919, "ŋ"},
      {0x091A, "tʃ"},  {0x091B, "tʃʰ"}, {0x091C, "dʒ"},  {0x091D, "dʒʰ"}, {0x091E, "ɲ"},
      {0x091F, "ʈ"},   {0x0920, "ʈʰ"},  {0x0921, "ɖ"},   {0x0922, "ɖʰ"}, {0x0923, "ɳ"},
      {0x0924, "t"},   {0x0925, "tʰ"},  {0x0926, "d"},   {0x0927, "dʰ"}, {0x0928, "n"},
      {0x092A, "p"},   {0x092B, "pʰ"},  {0x092C, "b"},   {0x092D, "bʰ"}, {0x092E, "m"},
      {0x092F, "j"},   {0x0930, "r"},   {0x0932, "l"},   {0x0933, "ɭ"},  {0x0935, "ʋ"},
      {0x0936, "ʃ"},   {0x0937, "ʂ"},   {0x0938, "s"},   {0x0939, "ɦ"},
  };
  return m;
}

const std::unordered_map<char32_t, std::string>& nukta_override_map() {
  static const std::unordered_map<char32_t, std::string> m{
      {0x0915, "q"}, {0x0916, "x"}, {0x0917, "ɣ"}, {0x091C, "z"},
      {0x0921, "ɽ"}, {0x0922, "ɽʰ"}, {0x092B, "f"},
  };
  return m;
}

bool is_devanagari_digit(char32_t cp) {
  return cp >= 0x0966 && cp <= 0x096F;
}

bool is_consonant(char32_t cp) {
  return base_cons_map().find(cp) != base_cons_map().end();
}

std::string cons_ipa(char32_t base, bool nukta) {
  if (nukta) {
    const auto& no = nukta_override_map();
    const auto it = no.find(base);
    if (it != no.end()) {
      return it->second;
    }
  }
  return base_cons_map().at(base);
}

bool sv_starts_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::string nasal_for_place(std::string_view first_onset) {
  if (first_onset.empty()) {
    return "\xC5\x8B";  // ŋ
  }
  // UTF-8 bytes for IPA (must match ``base_cons_map`` literals in this TU).
  // Palatal / postalveolar (before plain ``t`` / ``d`` prefix checks).
  if (sv_starts_with(first_onset, "t\xCA\x83") || sv_starts_with(first_onset, "t\xCA\x83\xCA\xB0") ||
      sv_starts_with(first_onset, "d\xCA\x92") || sv_starts_with(first_onset, "d\xCA\x92\xCA\xB0")) {
    return "\xC9\xB2";  // ɲ (matches Python: ʃ alone → final ``n``)
  }
  if (first_onset == "\xC9\xB2") {
    return "\xC9\xB2";
  }
  if (sv_starts_with(first_onset, "k") || sv_starts_with(first_onset, "g") || first_onset == "q") {
    return "\xC5\x8B";
  }
  if (sv_starts_with(first_onset, "\xCA\x88") || sv_starts_with(first_onset, "\xCA\x88\xCA\xB0") ||
      sv_starts_with(first_onset, "\xC9\x96") || sv_starts_with(first_onset, "\xC9\x96\xCA\xB0") ||
      sv_starts_with(first_onset, "\xC9\xB3") || sv_starts_with(first_onset, "\xC9\xBD") ||
      sv_starts_with(first_onset, "\xC9\xBD\xCA\xB0")) {
    return "\xC9\xB3";  // ɳ
  }
  if (sv_starts_with(first_onset, "t") || sv_starts_with(first_onset, "d") ||
      sv_starts_with(first_onset, "n")) {
    return "n";
  }
  if (sv_starts_with(first_onset, "p") || sv_starts_with(first_onset, "b") ||
      sv_starts_with(first_onset, "m")) {
    return "m";
  }
  return "n";
}

int syllable_weight(const Syllable& s) {
  if (!s.vowel.has_value() || s.vowel->empty()) {
    return 0;
  }
  const std::string& v = *s.vowel;
  if (v == "aː" || v == "iː" || v == "uː" || v == "eː" || v == "oː" || v == "ɛː" || v == "ɔː") {
    return 2;
  }
  return 1;
}

std::string assign_stress(const std::vector<std::string>& ipa_syllables,
                          const std::vector<int>& weights) {
  if (ipa_syllables.empty()) {
    return "";
  }
  if (ipa_syllables.size() == 1) {
    return ipa_syllables[0];
  }
  size_t best_i = 0;
  int best_w = -1;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (weights[i] > best_w) {
      best_w = weights[i];
      best_i = i;
    }
  }
  if (best_w <= 0) {
    best_i = ipa_syllables.size() >= 2 ? ipa_syllables.size() - 2 : 0;
  }
  std::string out;
  for (size_t i = 0; i < ipa_syllables.size(); ++i) {
    if (i > 0) {
      out.push_back('.');
    }
    if (i == best_i && best_w > 0) {
      out += "ˈ";
      out += ipa_syllables[i];
    } else {
      out += ipa_syllables[i];
    }
  }
  return out;
}

void apply_schwa_syncope(std::vector<Syllable>& syls) {
  if (syls.size() < 2) {
    return;
  }
  Syllable& last = syls.back();
  if (last.vowel.has_value() && *last.vowel == "ə" && last.inherent_schwa) {
    last.vowel = "";
    last.inherent_schwa = false;
  }
  for (size_t i = 0; i + 1 < syls.size(); ++i) {
    Syllable& a = syls[i];
    const Syllable& b = syls[i + 1];
    if (!a.vowel.has_value() || *a.vowel != "ə" || !a.inherent_schwa || b.onset.empty()) {
      continue;
    }
    const std::string& bo = b.onset[0];
    const std::string_view bov = bo;
    if (sv_starts_with(bov, "dʒ") || sv_starts_with(bov, "tʃ") || sv_starts_with(bov, "ʃ") ||
        bo == "ɲ") {
      a.vowel = "";
      a.inherent_schwa = false;
    }
  }
}

std::optional<std::vector<Syllable>> parse_devanagari_to_syllables(const std::string& word) {
  const std::string s = utf8_nfc_utf8proc(word);
  const std::u32string cps = utf8_str_to_u32(s);
  const size_t n = cps.size();
  size_t i = 0;
  std::vector<Syllable> out;
  bool has_letter = false;

  auto skip_joiners = [&]() {
    while (i < n && (cps[i] == kZwj || cps[i] == kZwnj)) {
      ++i;
    }
  };

  const auto& indep = indep_vowel_map();
  const auto& matra = matra_map();

  while (i < n) {
    skip_joiners();
    if (i >= n) {
      break;
    }
    const char32_t cp = cps[i];
    if (is_devanagari_digit(cp)) {
      return std::nullopt;
    }
    const auto iv = indep.find(cp);
    if (iv != indep.end()) {
      has_letter = true;
      Syllable sy;
      sy.vowel = iv->second;
      ++i;
      skip_joiners();
      if (i < n && cps[i] == kChandra) {
        sy.chandrabindu = true;
        ++i;
      }
      if (i < n && cps[i] == kAnusvara) {
        sy.anusvara = true;
        ++i;
      }
      if (i < n && cps[i] == kVisarga) {
        sy.visarga = true;
        ++i;
      }
      out.push_back(std::move(sy));
      continue;
    }
    if (!is_consonant(cp)) {
      ++i;
      continue;
    }

    has_letter = true;
    std::vector<std::string> onset;
    bool halant_end = false;
    while (i < n) {
      skip_joiners();
      if (i >= n || !is_consonant(cps[i])) {
        break;
      }
      const char32_t base = cps[i];
      ++i;
      bool nukta = i < n && cps[i] == kNukta;
      if (nukta) {
        ++i;
      }
      onset.push_back(cons_ipa(base, nukta));
      if (i < n && cps[i] == kVirama) {
        ++i;
        skip_joiners();
        if (i < n && is_consonant(cps[i])) {
          continue;
        }
        halant_end = true;
        break;
      }
      break;
    }

    if (halant_end) {
      Syllable sy;
      sy.onset = std::move(onset);
      sy.vowel = std::nullopt;
      if (i < n && cps[i] == kVisarga) {
        sy.visarga = true;
        ++i;
      }
      out.push_back(std::move(sy));
      continue;
    }

    std::optional<std::string> v;
    bool inc = false;
    if (i < n) {
      const auto mi = matra.find(cps[i]);
      if (mi != matra.end()) {
        v = mi->second;
        ++i;
      }
    }
    if (!v.has_value()) {
      v = std::string("ə");
      inc = true;
    }
    Syllable sy;
    sy.onset = std::move(onset);
    sy.vowel = std::move(v);
    sy.inherent_schwa = inc;
    if (i < n && cps[i] == kChandra) {
      sy.chandrabindu = true;
      ++i;
    }
    if (i < n && cps[i] == kAnusvara) {
      sy.anusvara = true;
      ++i;
    }
    if (i < n && cps[i] == kVisarga) {
      sy.visarga = true;
      ++i;
    }
    out.push_back(std::move(sy));
  }

  if (!has_letter) {
    return std::nullopt;
  }
  return out;
}

std::string render_syllables(const std::vector<Syllable>& syls, bool with_stress) {
  if (syls.empty()) {
    return "";
  }
  auto render_one = [&](size_t j) -> std::string {
    const Syllable& s = syls[j];
    std::string body;
    for (const std::string& p : s.onset) {
      body += p;
    }
    if (!s.vowel.has_value()) {
      if (s.visarga) {
        body += "ɦ";
      }
      return body;
    }
    std::string v = *s.vowel;
    if (s.chandrabindu && !v.empty()) {
      v += "\xCC\x83";  // U+0303 combining tilde
    }
    if (s.anusvara) {
      std::string_view nxt_onset;
      for (size_t k = j + 1; k < syls.size(); ++k) {
        if (!syls[k].onset.empty()) {
          nxt_onset = syls[k].onset[0];
          break;
        }
      }
      if (nxt_onset.empty()) {
        v += "\xCC\x83";
      } else {
        body += nasal_for_place(nxt_onset);
      }
    }
    body += v;
    if (s.visarga) {
      body += "ɦ";
    }
    return body;
  };

  std::vector<std::string> raw;
  std::vector<int> weights;
  raw.reserve(syls.size());
  weights.reserve(syls.size());
  for (size_t j = 0; j < syls.size(); ++j) {
    std::string r = render_one(j);
    if (!r.empty()) {
      raw.push_back(std::move(r));
      weights.push_back(syllable_weight(syls[j]));
    }
  }
  if (!with_stress || raw.empty()) {
    std::string o;
    for (size_t i = 0; i < raw.size(); ++i) {
      if (i > 0) {
        o.push_back('.');
      }
      o += raw[i];
    }
    return o;
  }
  return assign_stress(raw, weights);
}

void strip_edges_punct(std::string_view w, std::string& core) {
  const std::u32string u = utf8_str_to_u32(std::string(w));
  auto punct_cp = [](char32_t cp) {
    const auto c =
        static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(cp)));
    return c == UTF8PROC_CATEGORY_PO || c == UTF8PROC_CATEGORY_PD || c == UTF8PROC_CATEGORY_PE ||
           c == UTF8PROC_CATEGORY_PI || c == UTF8PROC_CATEGORY_PS || c == UTF8PROC_CATEGORY_PF ||
           c == UTF8PROC_CATEGORY_PC;
  };
  size_t a = 0;
  size_t b = u.size();
  while (a < b && punct_cp(u[a])) {
    ++a;
  }
  while (b > a && punct_cp(u[b - 1])) {
    --b;
  }
  core.clear();
  for (size_t i = a; i < b; ++i) {
    utf8_append_codepoint(core, u[i]);
  }
}

bool has_devanagari(std::string_view s) {
  const std::u32string u = utf8_str_to_u32(std::string(s));
  for (char32_t c : u) {
    if (c >= 0x0900 && c <= 0x097F) {
      return true;
    }
  }
  return false;
}

bool all_ascii_digits_sv(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (unsigned char c : s) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::filesystem::path builtin_hindi_dict_path() {
  return std::filesystem::current_path() / "hi" / "dict.tsv";
}

HindiRuleG2p::HindiRuleG2p(std::filesystem::path dict_tsv)
    : HindiRuleG2p(std::move(dict_tsv), Options{}) {}

void load_hindi_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::string>& lexicon) {
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string surf = trim_ascii_ws_copy(std::string_view(line).substr(0, tab));
    std::string ipa_val = trim_ascii_ws_copy(std::string_view(line).substr(tab + 1));
    if (surf.empty()) {
      continue;
    }
    const std::string k = utf8_nfc_utf8proc(surf);
    if (lexicon.find(k) == lexicon.end()) {
      lexicon.emplace(k, std::move(ipa_val));
    }
  }
}

HindiRuleG2p::HindiRuleG2p(std::filesystem::path dict_tsv, Options options)
    : options_(std::move(options)) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("Hindi G2P: lexicon not found at " + dict_tsv.generic_string());
  }
  std::ifstream in(dict_tsv);
  if (!in) {
    throw std::runtime_error("Hindi G2P: cannot open " + dict_tsv.generic_string());
  }
  load_hindi_lexicon_stream(in, lexicon_);
}

HindiRuleG2p::HindiRuleG2p(std::string dict_tsv_utf8, Options options)
    : options_(std::move(options)) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_hindi_lexicon_stream(in, lexicon_);
}

std::string HindiRuleG2p::word_to_ipa(const std::string& word) const {
  return g2p_single_word(word);
}

std::string HindiRuleG2p::g2p_single_word(std::string_view word) const {
  std::string core;
  strip_edges_punct(word, core);
  if (core.empty()) {
    return "";
  }
  const std::string nfc = utf8_nfc_utf8proc(core);
  const auto it = lexicon_.find(nfc);
  if (it != lexicon_.end()) {
    return it->second;
  }
  auto syls_opt = parse_devanagari_to_syllables(nfc);
  if (!syls_opt.has_value()) {
    return "";
  }
  std::vector<Syllable> syls = std::move(*syls_opt);
  apply_schwa_syncope(syls);
  return render_syllables(syls, options_.with_stress);
}

std::string HindiRuleG2p::text_to_ipa_no_expand(std::string text,
                                                 std::vector<G2pWordLog>* per_word_log) const {
  (void)per_word_log;
  const std::string raw = utf8_nfc_utf8proc(trim_ascii_ws_copy(text));
  if (raw.empty()) {
    return "";
  }
  std::vector<std::string> ipa_words;
  std::istringstream iss(raw);
  std::string w;
  while (iss >> w) {
    if (has_devanagari(w)) {
      const std::string ipa = g2p_single_word(w);
      if (!ipa.empty()) {
        ipa_words.push_back(ipa);
      }
    } else if (options_.expand_cardinal_digits) {
      std::string core;
      strip_edges_punct(w, core);
      if (all_ascii_digits_sv(core)) {
        const std::string expanded = expand_cardinal_digits_to_hindi_words(core);
        std::istringstream iss2(expanded);
        std::string frag;
        while (iss2 >> frag) {
          const std::string ipa = g2p_single_word(frag);
          if (!ipa.empty()) {
            ipa_words.push_back(ipa);
          }
        }
      }
    }
  }
  std::string out;
  for (size_t i = 0; i < ipa_words.size(); ++i) {
    if (i > 0) {
      out.push_back(' ');
    }
    out += ipa_words[i];
  }
  return out;
}

std::string HindiRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_hindi_digit_tokens_in_text(std::move(text));
    text = expand_devanagari_digit_runs_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(std::move(text), per_word_log);
}

std::vector<std::string> HindiRuleG2p::dialect_ids() {
  return {"hi", "hi-IN", "hindi"};
}

bool dialect_resolves_to_hindi_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "hi" || s == "hi-in" || s == "hindi";
}

std::filesystem::path resolve_hindi_dict_path(const std::filesystem::path& model_root) {
  return model_root / "hi" / "dict.tsv";
}

std::string hindi_text_to_ipa(const std::string& text, bool with_stress,
                                std::vector<G2pWordLog>* per_word_log, bool expand_cardinal_digits,
                                const std::filesystem::path& dict_tsv) {
  HindiRuleG2p::Options o;
  o.with_stress = with_stress;
  o.expand_cardinal_digits = expand_cardinal_digits;
  const std::filesystem::path path =
      dict_tsv.empty() ? builtin_hindi_dict_path() : dict_tsv;
  HindiRuleG2p g(path, std::move(o));
  return g.text_to_ipa(text, per_word_log);
}

}  // namespace moonshine_tts
