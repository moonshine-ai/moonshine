#include "korean.h"
#include "korean-numbers.h"
#include "g2p-word-log.h"
#include "utf8-utils.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

constexpr char32_t kHangulBase = 0xAC00;
constexpr char32_t kHangulEnd = 0xD7A3;

constexpr int kIdxO = 11;

// Choseong indices (same order as Python CHO_NAMES).
constexpr int kChoK = 0;
constexpr int kChoKk = 1;
constexpr int kChoN = 2;
constexpr int kChoT = 3;
constexpr int kChoTt = 4;
constexpr int kChoR = 5;
constexpr int kChoM = 6;
constexpr int kChoP = 7;
constexpr int kChoPp = 8;
constexpr int kChoS = 9;
constexpr int kChoSs = 10;
constexpr int kChoNgOnset = 11;
constexpr int kChoC = 12;
constexpr int kChoCc = 13;
constexpr int kChoCh = 14;
constexpr int kChoKh = 15;
constexpr int kChoTh = 16;
constexpr int kChoPh = 17;
constexpr int kChoH = 18;

constexpr int kJongNgCoda = 21;

struct Syllable {
  int cho = 0;
  int jung = 0;
  int jong = 0;
};

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return;
  }
  for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size())) {
    s.replace(p, from.size(), to);
  }
}

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

std::string strip_mn_after_nfd(const std::string& ipa) {
  utf8proc_uint8_t* nfd =
      utf8proc_NFD(reinterpret_cast<const utf8proc_uint8_t*>(ipa.c_str()));
  if (nfd == nullptr) {
    return ipa;
  }
  const std::string nfd_str(reinterpret_cast<char*>(nfd));
  std::free(nfd);
  // Decode with the same lenient UTF-8 scan as the rest of rule-G2P (lexicon IPA may be messy).
  const std::u32string u32 = utf8_str_to_u32(nfd_str);
  std::string filtered;
  for (char32_t cp : u32) {
    if (utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_MN) {
      continue;  // strip all combining marks (incl. U+031A unreleased, U+0348 tense)
    }
    utf8_append_codepoint(filtered, cp);
  }
  utf8proc_uint8_t* nfc =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(filtered.c_str()));
  if (nfc == nullptr) {
    return filtered;
  }
  std::string composed(reinterpret_cast<char*>(nfc));
  std::free(nfc);
  return composed;
}

std::optional<std::pair<int, int>> jong_split_for_linking(int jong) {
  switch (jong) {
  case 1:
    return {{0, kChoK}};
  case 2:
    return {{0, kChoKk}};
  case 3:
    return {{1, kChoS}};
  case 4:
    return {{0, kChoN}};
  case 5:
    return {{4, kChoC}};
  case 6:
    return {{4, kChoH}};
  case 7:
    return {{0, kChoT}};
  case 8:
    return {{0, kChoR}};
  case 9:
    return {{8, kChoK}};
  case 10:
    return {{8, kChoM}};
  case 11:
    return {{8, kChoP}};
  case 12:
    return {{8, kChoS}};
  case 13:
    return {{8, kChoTh}};
  case 14:
    return {{8, kChoPh}};
  case 15:
    return {{8, kChoH}};
  case 16:
    return {{0, kChoM}};
  case 17:
    return {{0, kChoP}};
  case 18:
    return {{17, kChoS}};
  case 19:
    return {{0, kChoS}};
  case 20:
    return {{0, kChoSs}};
  case 22:
    return {{0, kChoC}};
  case 23:
    return {{0, kChoCh}};
  case 24:
    return {{0, kChoKh}};
  case 25:
    return {{0, kChoTh}};
  case 26:
    return {{0, kChoPh}};
  case 27:
    return {{0, kChoH}};
  default:
    return std::nullopt;
  }
}

bool jong_triggers_tense(int jong) {
  switch (jong) {
  case 1:
  case 2:
  case 3:
  case 7:
  case 17:
  case 18:
  case 19:
  case 20:
  case 22:
  case 23:
  case 24:
  case 25:
  case 26:
    return true;
  default:
    return false;
  }
}

int tense_cho(int plain_cho) {
  switch (plain_cho) {
  case kChoK:
    return kChoKk;
  case kChoT:
    return kChoTt;
  case kChoP:
    return kChoPp;
  case kChoS:
    return kChoSs;
  case kChoC:
    return kChoCc;
  default:
    return plain_cho;
  }
}

std::optional<Syllable> decompose_syllable_cp(char32_t ch) {
  if (ch < kHangulBase || ch > kHangulEnd) {
    return std::nullopt;
  }
  const std::uint32_t code = static_cast<std::uint32_t>(ch - kHangulBase);
  Syllable s;
  s.jong = static_cast<int>(code % 28);
  s.jung = static_cast<int>((code / 28) % 21);
  s.cho = static_cast<int>(code / 28 / 21);
  return s;
}

std::vector<Syllable> text_to_syllables(std::string_view text) {
  const std::string nfc = utf8_nfc_utf8proc(trim_ascii_ws_copy(text));
  std::vector<Syllable> out;
  size_t i = 0;
  while (i < nfc.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(nfc, i, cp, adv);
    if (const auto d = decompose_syllable_cp(cp)) {
      out.push_back(*d);
    }
    i += adv;
  }
  return out;
}

void apply_linking(std::vector<Syllable>& syls) {
  std::size_t i = 0;
  while (i + 1 < syls.size()) {
    Syllable& cur = syls[i];
    Syllable& nxt = syls[i + 1];
    if (cur.jong == 0) {
      ++i;
      continue;
    }
    if (cur.jong == kJongNgCoda) {
      ++i;
      continue;
    }
    if (nxt.cho != kIdxO) {
      ++i;
      continue;
    }
    const auto spec = jong_split_for_linking(cur.jong);
    if (!spec.has_value()) {
      ++i;
      continue;
    }
    cur.jong = spec->first;
    nxt.cho = spec->second;
    ++i;
  }
}

void apply_lateralization(std::vector<Syllable>& syls) {
  for (std::size_t i = 0; i + 1 < syls.size(); ++i) {
    // ㄴ-coda before ㄹ-onset → lateralize coda to ɫ (유음화 rule).
    if (syls[i].jong == 4 && syls[i + 1].cho == kChoR) {
      syls[i].jong = 8;
    }
    // ɫ-coda + ɾ-onset → drop ɾ onset (유음화: long lateral, Piper style).
    // e.g. 질량 ɫ+ɾ → ɫ (no onset), 가로질러 ɫ+ɾ → ɫ+vowel
    if (syls[i].jong == 8 && syls[i + 1].cho == kChoR) {
      syls[i + 1].cho = kIdxO;
    }
  }
}

std::string ipa_onset(int cho, bool tense, bool aspirate) {
  if (cho == kIdxO) {
    return "";
  }
  std::string ip;
  switch (cho) {
  case kChoK:
    ip = "k";
    break;
  case kChoKk:
    ip = "k";  // tense ㄲ: ͈ stripped → plain k
    break;
  case kChoN:
    ip = "n";
    break;
  case kChoT:
    ip = "d";
    break;
  case kChoTt:
    ip = "t";  // tense ㄸ: ͈ stripped → plain t
    break;
  case kChoR:
    ip = "\xC9\xBE";  // ɾ
    break;
  case kChoM:
    ip = "m";
    break;
  case kChoP:
    ip = "p";
    break;
  case kChoPp:
    ip = "p";  // tense ㅃ: ͈ stripped → plain p
    break;
  case kChoS:
    ip = "s";
    break;
  case kChoSs:
    ip = "s";  // tense ㅆ: ͈ stripped → plain s
    break;
  case kChoC:
    ip = "t\xC9\x95";  // tɕ (alveolo-palatal, eSpeak style for lenis ㅈ)
    break;
  case kChoCc:
    ip = "t\xC9\x95";  // tɕ (tense ㅉ, tense mark stripped)
    break;
  case kChoCh:
    ip = "t\xCA\x83h";  // tʃh (eSpeak style, was tɕʰ)
    break;
  case kChoKh:
    ip = "kh";  // ASCII h for aspiration (eSpeak style, was kʰ)
    break;
  case kChoTh:
    ip = "th";  // (was tʰ)
    break;
  case kChoPh:
    ip = "ph";  // (was pʰ)
    break;
  case kChoH:
    ip = "h";
    break;
  default:
    return "";
  }
  if (tense) {
    const int tc = tense_cho(cho);
    if (tc != cho) {
      return ipa_onset(tc, false, false);
    }
  }
  if (aspirate) {
    if (cho == kChoK) return "kh";
    if (cho == kChoT) return "th";
    if (cho == kChoP) return "ph";
    if (cho == kChoC) return "t\xCA\x83h";  // tʃh for aspirated ㅊ (postalveolar, eSpeak convention)
  }
  return ip;
}

std::string ipa_nucleus(int jung) {
  static const char* const vmap[] = {
      "a",   "\xC9\x9B",   "ja",  "j\xC9\x9B",  "\xCA\x8C",   "e",   "j\xCA\x8C",  "je",  "o",   "wa",  "w\xC9\x9B",  "we",
      "jo",  "u",   "w\xCA\x8C",  "we",  "wi",  "ju",  "\xC9\xAF",   "\xC9\xAFj",  "i",
  };
  if (jung < 0 || jung > 20) {
    return "ə";
  }
  return vmap[jung];
}

std::string ipa_coda_simple(int jong) {
  if (jong == 0) {
    return "";
  }
  if (jong == 1 || jong == 2 || jong == 24) {
    return "q";
  }
  if (jong == 7 || jong == 25 || jong == 19 || jong == 20 || jong == 22 || jong == 23) {
    return "t-";
  }
  if (jong == 17 || jong == 26 || jong == 18) {
    return "p-";
  }
  if (jong == 4) {
    return "n";
  }
  if (jong == 8) {
    return "\xC9\xAB";  // ɫ
  }
  if (jong == 16) {
    return "m";
  }
  if (jong == kJongNgCoda) {
    return "\xC5\x8B";  // ŋ
  }
  if (jong == 27) {
    return "t-";
  }
  if (jong == 3) {
    return "q";
  }
  if (jong == 5 || jong == 6) {
    return "n";
  }
  if (jong >= 9 && jong <= 15) {
    return "l";
  }
  return "";
}

std::string coda_nasal_assimilate(int jong, std::optional<int> next_cho) {
  if (!next_cho.has_value() || (*next_cho != kChoM && *next_cho != kChoN && *next_cho != kChoNgOnset)) {
    return ipa_coda_simple(jong);
  }
  if (*next_cho != kChoM && *next_cho != kChoN) {
    return ipa_coda_simple(jong);
  }
  if (jong == 1 || jong == 2 || jong == 3 || jong == 24 || jong == 9) {
    return "ŋ";
  }
  if (jong == 7 || jong == 19 || jong == 20 || jong == 22 || jong == 23 || jong == 25 || jong == 27 ||
      jong == 12 || jong == 13 || jong == 14 || jong == 15) {
    return "n";
  }
  if (jong == 17 || jong == 18 || jong == 26 || jong == 11 || jong == 14) {
    return "m";
  }
  return ipa_coda_simple(jong);
}

std::string syllables_to_ipa(const std::vector<Syllable>& syls, std::string_view syllable_sep) {
  if (syls.empty()) {
    return "";
  }
  std::string pieces_joined;
  for (std::size_t i = 0; i < syls.size(); ++i) {
    const Syllable& s = syls[i];
    const Syllable* nxt = (i + 1 < syls.size()) ? &syls[i + 1] : nullptr;
    const Syllable* prev = (i > 0) ? &syls[i - 1] : nullptr;
    const int cho = s.cho;

    std::string onset_ipa;
    if (cho != kIdxO) {
      const bool after_h =
          prev != nullptr && prev->jong == 27 &&
          (cho == kChoK || cho == kChoT || cho == kChoP || cho == kChoC);
      const bool tense_after =
          prev != nullptr && jong_triggers_tense(prev->jong) &&
          (cho == kChoK || cho == kChoT || cho == kChoP || cho == kChoS || cho == kChoC);
      // Voiced allophones: ㄱ→ɡ, ㅂ→b, ㅈ→dʑ between voiced sounds (vowels, nasals, liquids).
      const bool voiced_context =
          i > 0 && prev != nullptr &&
          (prev->jong == 0 ||           // after vowel (no coda)
           prev->jong == 4 ||           // after n
           prev->jong == 8 ||           // after ɫ
           prev->jong == 16 ||          // after m
           prev->jong == kJongNgCoda);  // after ŋ
      if (after_h) {
        onset_ipa = ipa_onset(cho, false, true);
      } else if (tense_after) {
        onset_ipa = ipa_onset(cho, true, false);
      } else if (voiced_context && cho == kChoK) {
        onset_ipa = "\xC9\xA1";  // ɡ
      } else if (voiced_context && cho == kChoP) {
        onset_ipa = "b";
      } else if (voiced_context && cho == kChoC) {
        onset_ipa = "d\xCA\x91";  // dʑ
      } else {
        onset_ipa = ipa_onset(cho, false, false);
      }
    }

    // Unstressed ㅏ (jung==0) in non-initial syllables reduces to ɐ (U+0250).
    const std::string nucleus = (i > 0 && s.jung == 0) ? "\xC9\x90" : ipa_nucleus(s.jung);

    // Embed stress marker immediately before the nucleus vowel (eSpeak-ng Piper convention).
    // Primary stress ˈ (U+02C8) on syllable 0; secondary stress ˌ (U+02CC) on syllables 2,4,6…
    std::string stress;
    if (i == 0) {
      stress = "\xCB\x88";  // ˈ primary
    } else if (i % 2 == 0 && syls.size() >= 3) {
      stress = "\xCB\x8C";  // ˌ secondary (even syllables in words with 3+ syllables)
    }

    std::string coda_ipa;
    if (s.jong != 0) {
      const bool h_lost_before_asp =
          nxt != nullptr && s.jong == 27 &&
          (nxt->cho == kChoK || nxt->cho == kChoT || nxt->cho == kChoP || nxt->cho == kChoC);
      if (h_lost_before_asp) {
        coda_ipa = "";
      } else if (nxt == nullptr) {
        coda_ipa = ipa_coda_simple(s.jong);
      } else if (nxt->cho != kIdxO) {
        if (nxt->cho == kChoN || nxt->cho == kChoM) {
          coda_ipa = coda_nasal_assimilate(s.jong, nxt->cho);
        } else {
          coda_ipa = ipa_coda_simple(s.jong);
        }
      } else {
        coda_ipa = ipa_coda_simple(s.jong);
      }
    }

    if (!pieces_joined.empty()) {
      pieces_joined += syllable_sep;
    }
    pieces_joined += onset_ipa + stress + nucleus + coda_ipa;
  }
  return pieces_joined;
}

// Split n into natural Korean speech units for TTS (千/百/나머지 boundaries).
// e.g. 1986 → ["천","구백","팔십육"],  2002 → ["이천","이"],  7 → ["칠"].
std::vector<std::string> sino_cardinal_speech_units(std::uint64_t n) {
  if (n == 0) return {"영"};
  if (n >= 100000000ULL) {
    return {int_to_sino_korean_hangul(n)};  // 억+ too complex, single unit
  }
  std::vector<std::string> units;
  if (n >= 10000ULL) {
    const std::uint64_t man = n / 10000ULL;
    units.push_back(int_to_sino_korean_hangul(man * 10000ULL));  // 만 group
    n %= 10000ULL;
    if (n == 0) return units;
  }
  const unsigned v = static_cast<unsigned>(n);
  const unsigned q = v / 1000U;
  const unsigned r = v % 1000U;
  const unsigned b = r / 100U;
  const unsigned r2 = r % 100U;
  if (q > 0) units.push_back(int_to_sino_korean_hangul(static_cast<std::uint64_t>(q) * 1000ULL));
  if (b > 0) units.push_back(int_to_sino_korean_hangul(static_cast<std::uint64_t>(b) * 100ULL));
  if (r2 > 0) units.push_back(int_to_sino_korean_hangul(static_cast<std::uint64_t>(r2)));
  return units;
}

std::string g2p_hangul_rules_only_inner(std::string_view hangul, std::string_view syllable_sep) {
  if (hangul.empty()) {
    return "";
  }
  std::vector<Syllable> syls = text_to_syllables(hangul);
  if (syls.empty()) {
    return "";
  }
  apply_linking(syls);
  apply_lateralization(syls);
  return syllables_to_ipa(syls, syllable_sep);
}

}  // namespace

std::string KoreanRuleG2p::normalize_korean_ipa(std::string ipa) {
  ipa = trim_ascii_ws_copy(ipa);
  if (ipa.empty()) {
    return ipa;
  }
  std::string t = strip_mn_after_nfd(ipa);
  // Normalize tie-bar affricates.
  // ㅊ (aspirated): postalveolar tʃh (eSpeak convention).
  // ㅈ (lenis unvoiced): alveolo-palatal tɕ; (voiced): dʑ — preserved as-is.
  const std::string tie = "\xCD\xA1";
  replace_all(t, "t" + tie + "\xCA\x83h",        "t\xCA\x83h");   // t͡ʃh → tʃh
  replace_all(t, "t" + tie + "\xC9\x95\xCA\xB0", "t\xCA\x83h");   // t͡ɕʰ → tʃh (aspirated → postalveolar)
  replace_all(t, "d" + tie + "\xCA\x91\xCA\xB0", "t\xCA\x83h");   // d͡ʑʰ → tʃh
  replace_all(t, "t" + tie + "\xCA\x83",          "t\xCA\x83");    // t͡ʃ  → tʃ
  replace_all(t, "t" + tie + "\xC9\x95",          "t\xC9\x95");    // t͡ɕ  → tɕ (keep alveolo-palatal)
  replace_all(t, "d" + tie + "\xCA\x91",          "d\xCA\x91");    // d͡ʑ  → dʑ (keep voiced)
  replace_all(t, "t" + tie + "s",                 "ts");            // t͡s  → ts
  replace_all(t, "p" + tie + "\xC9\xB8",          "ph");           // p͡ɸ  → ph
  replace_all(t, "k" + tie + "x",                 "kh");           // k͡x  → kh
  // Aspiration: sʰ → s; bare ɕʰ → tʃh; remaining ʰ → ASCII h
  replace_all(t, "s\xCA\xB0",         "s");          // sʰ → s (eSpeak treats ㅅ as plain s)
  replace_all(t, "\xC9\x95\xCA\xB0",  "t\xCA\x83h"); // ɕʰ → tʃh
  replace_all(t, "\xCA\x83\xCA\xB0",  "t\xCA\x83h"); // ʃʰ → tʃh
  replace_all(t, "\xCA\xB0",          "h");           // ʰ  → h (kh, th, ph etc.)
  // tɕh (from ɕʰ after ʰ→h — shouldn't remain, but guard) → tʃh
  replace_all(t, "t\xC9\x95h",        "t\xCA\x83h"); // tɕh → tʃh (bare aspirated affricate)
  replace_all(t, "\xC9\xAD",  "\xC9\xAB");  // ɭ → ɫ
  replace_all(t, "\xC9\xB0",  "\xC9\xAF");  // ɰ → ɯ (velar approximant not in vocoder map)
  replace_all(t, "\xC9\xB2",  "n");          // ɲ → n
  replace_all(t, "\xCE\xB2",  "b");
  replace_all(t, "\xC9\xA6",  "h");          // ɦ → h
  replace_all(t, "\xC3\xA7",  "h");          // ç → h
  // ɡ (U+0261) preserved: vocoder trained on eSpeak which uses ɡ for voiced ㄱ
  replace_all(t, "\xC9\x9F",  "t\xCA\x83");  // ɟ → tʃ
  return t;
}

std::string KoreanRuleG2p::extract_hangul(std::string_view s) const {
  const std::string nfc = utf8_nfc_utf8proc(s);
  std::string out;
  size_t i = 0;
  while (i < nfc.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(nfc, i, cp, adv);
    if (cp >= kHangulBase && cp <= kHangulEnd) {
      utf8_append_codepoint(out, cp);
    }
    i += adv;
  }
  return out;
}

std::string KoreanRuleG2p::g2p_hangul_rules_only(std::string_view hangul) const {
  if (hangul.empty()) {
    return "";
  }
  return normalize_korean_ipa(
      g2p_hangul_rules_only_inner(hangul, options_.syllable_sep));
}

std::string KoreanRuleG2p::g2p_single_fragment(std::string_view frag) const {
  const std::string f = utf8_nfc_utf8proc(trim_ascii_ws_copy(frag));
  if (f.empty()) {
    return "";
  }
  const auto it = lexicon_.find(f);
  if (it != lexicon_.end()) {
    return it->second;
  }
  const std::string h = extract_hangul(f);
  if (h.empty()) {
    return "";
  }
  const auto it2 = lexicon_.find(h);
  if (it2 != lexicon_.end()) {
    return it2->second;
  }
  return g2p_hangul_rules_only(h);
}

KoreanRuleG2p::KoreanRuleG2p(std::filesystem::path dict_tsv)
    : KoreanRuleG2p(std::move(dict_tsv), Options{}) {}

void load_korean_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::string>& lexicon) {
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
      lexicon.emplace(k, KoreanRuleG2p::normalize_korean_ipa(std::move(ipa_val)));
    }
  }
}

KoreanRuleG2p::KoreanRuleG2p(std::filesystem::path dict_tsv, Options options)
    : options_(std::move(options)) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("Korean G2P: lexicon not found at " + dict_tsv.generic_string());
  }
  std::ifstream in(dict_tsv);
  if (!in) {
    throw std::runtime_error("Korean G2P: cannot open " + dict_tsv.generic_string());
  }
  load_korean_lexicon_stream(in, lexicon_);
}

KoreanRuleG2p::KoreanRuleG2p(std::string dict_tsv_utf8, Options options)
    : options_(std::move(options)) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_korean_lexicon_stream(in, lexicon_);
}

std::string KoreanRuleG2p::text_to_ipa(std::string text,
                                       std::vector<G2pWordLog>* per_word_log) {
  (void)per_word_log;
  const std::string raw = utf8_nfc_utf8proc(trim_ascii_ws_copy(text));
  if (raw.empty()) {
    return "";
  }
  // Ensure IPA has a stress marker. Rule-based output embeds ˈ/ˌ within syllables; lexicon
  // entries that already start with a marker are returned as-is. Any entry without any stress
  // marker (rare) gets ˈ prepended as a fallback.
  auto add_word_stress = [](std::string ipa) -> std::string {
    if (ipa.find("\xCB\x88") != std::string::npos) return ipa;  // already has ˈ
    if (ipa.find("\xCB\x8C") != std::string::npos) return ipa;  // already has ˌ
    return "\xCB\x88" + ipa;  // fallback: prepend primary stress
  };

  // Pre-tokenize: replace brackets/parens with spaces to prevent adjacent Korean words from
  // merging. e.g. "파동함수(켓)에" → "파동함수 켓 에", "(大韓)은" → " 大韓 은".
  std::string tokenizable;
  tokenizable.reserve(raw.size());
  for (unsigned char c : raw) {
    if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}') {
      tokenizable += ' ';
    } else {
      tokenizable += static_cast<char>(c);
    }
  }

  std::vector<std::string> ipa_words;
  std::istringstream iss(tokenizable);
  std::string w;
  while (iss >> w) {
    if (options_.expand_cardinal_digits) {
      if (const auto frags = korean_reading_fragments_from_ascii_numeral_token(w)) {
        for (const std::string& frag : *frags) {
          const std::string ipa = g2p_single_fragment(frag);
          if (!ipa.empty()) {
            ipa_words.push_back(add_word_stress(ipa));
          }
        }
        continue;
      }
    }
    // Handle mixed numeric+Hangul tokens such as "1986년", "7월", "3일", "8개월".
    // Split into natural speech units: 1986 → ["천","구백","팔십육"] + "년" on last unit.
    if (options_.expand_cardinal_digits) {
      size_t num_end = 0;
      bool has_digit = false;
      while (num_end < w.size()) {
        const unsigned char uc = static_cast<unsigned char>(w[num_end]);
        if (uc >= '0' && uc <= '9') { has_digit = true; ++num_end; }
        else if ((uc == ',' || uc == '_') && has_digit) { ++num_end; }
        else { break; }
      }
      if (has_digit && num_end < w.size()) {
        const std::string hangul_tail = extract_hangul(std::string_view(w).substr(num_end));
        if (!hangul_tail.empty()) {
          // Parse as simple non-negative integer for speech-unit splitting.
          bool is_simple = true;
          for (size_t k = 0; k < num_end; ++k) {
            if (w[k] < '0' || w[k] > '9') { is_simple = false; break; }
          }
          if (is_simple) {
            std::uint64_t n = 0;
            bool overflow = false;
            for (size_t k = 0; k < num_end; ++k) {
              const std::uint64_t d = static_cast<std::uint64_t>(w[k] - '0');
              if (n > (std::numeric_limits<std::uint64_t>::max() - d) / 10ULL) {
                overflow = true; break;
              }
              n = n * 10ULL + d;
            }
            if (!overflow) {
              auto speech_units = sino_cardinal_speech_units(n);
              if (!speech_units.empty()) {
                speech_units.back() += hangul_tail;  // attach 년/월/일 to last unit
                for (const auto& unit : speech_units) {
                  const std::string ipa = g2p_single_fragment(unit);
                  if (!ipa.empty()) ipa_words.push_back(add_word_stress(ipa));
                }
                continue;
              }
            }
          }
          // Fallback for comma-separated or out-of-range numbers.
          const std::string num_sv(w, 0, num_end);
          if (const auto num_frags = korean_reading_fragments_from_ascii_numeral_token(num_sv)) {
            std::string combined;
            for (const auto& frag : *num_frags) combined += frag;
            combined += hangul_tail;
            const std::string ipa = g2p_single_fragment(combined);
            if (!ipa.empty()) ipa_words.push_back(add_word_stress(ipa));
            continue;
          }
        }
      }
    }
    const std::string h = extract_hangul(w);
    if (h.empty()) {
      continue;
    }
    const auto it = lexicon_.find(h);
    if (it != lexicon_.end()) {
      if (!it->second.empty()) ipa_words.push_back(add_word_stress(it->second));
    } else {
      const std::string ipa = g2p_hangul_rules_only(h);
      if (!ipa.empty()) ipa_words.push_back(add_word_stress(ipa));
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

std::vector<std::string> KoreanRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"ko", "ko-KR", "ko_kr", "korean", "Korean"});
}

bool dialect_resolves_to_korean_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "ko" || s == "ko-kr" || s == "korean";
}

std::filesystem::path resolve_korean_dict_path(const std::filesystem::path& model_root) {
  return model_root / "ko" / "dict.tsv";
}

}  // namespace moonshine_tts
