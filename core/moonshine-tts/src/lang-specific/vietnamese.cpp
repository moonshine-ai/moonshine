#include "vietnamese.h"
#include "g2p-word-log.h"
#include "utf8-utils.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

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

std::string utf8_lower_nfc(std::string_view s) {
  const std::string nfc = utf8_nfc_utf8proc(s);
  std::string out;
  for (size_t i = 0; i < nfc.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(nfc, i, cp, adv)) {
      out.push_back(nfc[i]);
      ++i;
      continue;
    }
    const utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
    utf8_append_codepoint(out, static_cast<char32_t>(lo));
    i += adv;
  }
  return utf8_nfc_utf8proc(out);
}

bool starts_with_sv(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

bool ends_with_str(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Tone combining marks (NFD) -> id 2..6; default 1 (ngang).
int split_tone(std::string_view in, std::string& body_nfc_out) {
  const std::string nfc = utf8_nfc_utf8proc(in);
  utf8proc_uint8_t* nfd = utf8proc_NFD(reinterpret_cast<const utf8proc_uint8_t*>(nfc.c_str()));
  if (nfd == nullptr) {
    body_nfc_out = nfc;
    return 1;
  }
  int tone = 1;
  std::string stripped;
  for (const utf8proc_uint8_t* p = nfd; *p != 0;) {
    utf8proc_int32_t cp = 0;
    utf8proc_ssize_t adv = utf8proc_iterate(p, -1, &cp);
    if (adv <= 0) {
      break;
    }
    if (cp == 0x0300) {
      tone = 2;
    } else if (cp == 0x0301) {
      tone = 5;
    } else if (cp == 0x0303) {
      tone = 4;
    } else if (cp == 0x0309) {
      tone = 3;
    } else if (cp == 0x0323) {
      tone = 6;
    } else {
      char32_t u = static_cast<char32_t>(cp);
      utf8_append_codepoint(stripped, u);
    }
    p += adv;
  }
  std::free(nfd);
  body_nfc_out = utf8_nfc_utf8proc(stripped);
  return tone;
}

bool is_vowel_letter_char(char32_t cp) {
  const utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
  const char32_t c = static_cast<char32_t>(lo);
  if (c == U'y') {
    return true;
  }
  if (c == U'i') {
    return true;
  }
  // Vietnamese vowel letters (lowercase NFC covers most)
  return (c >= U'a' && c <= U'z' && std::string_view("aeiouy").find(static_cast<char>(c)) != std::string_view::npos) ||
         c == U'\u0103' || c == U'\u00e2' || c == U'\u00ea' || c == U'\u00f4' || c == U'\u01a1' ||
         c == U'\u01b0' || c == U'\u0111';
}

bool is_vowel_first_utf8(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(std::string(s), 0, cp, adv)) {
    return false;
  }
  return is_vowel_letter_char(cp);
}

bool front_vowel_utf8(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(std::string(s), 0, cp, adv)) {
    return false;
  }
  const utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
  const char32_t c = static_cast<char32_t>(lo);
  if (c == U'i' || c == U'\u00ed' || c == U'\u00ec' || c == U'\u1ec9' || c == U'\u0129' || c == U'\u1ecb') {
    return true;
  }
  return c == U'e' || c == U'\u00ea' || c == U'\u00e9' || c == U'\u00e8' || c == U'\u1ebb' || c == U'\u1ebd' ||
         c == U'\u1eb9' || c == U'\u1ebf' || c == U'\u1ec1' || c == U'\u1ec3' || c == U'\u1ec5' || c == U'\u1ec7';
}

bool rime_is_only_i(std::string_view rest) {
  if (rest.empty()) {
    return false;
  }
  if (rest == "i") {
    return true;
  }
  // Single letter i with marks already stripped -> "i"
  if (rest.size() <= 4) {
    char32_t cp = 0;
    size_t adv = 0;
    if (utf8_decode_at(std::string(rest), 0, cp, adv) && adv == rest.size()) {
      const utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
      return static_cast<char32_t>(lo) == U'i';
    }
  }
  return false;
}

// Returns (onset_ipa, rime)
std::pair<std::string, std::string> parse_onset(std::string_view body_sv) {
  const std::string body(body_sv);
  if (body.empty()) {
    return {{}, {}};
  }
  const size_t n = body.size();

  if (n >= 4 && body.compare(0, 3, "ngh") == 0) {
    const std::string r = body.substr(3);
    if (!r.empty() && front_vowel_utf8(r)) {
      return {"\xC5\x8B", r};  // ŋ
    }
  }
  if (n >= 3 && body.compare(0, 2, "ng") == 0) {
    const std::string r = body.substr(2);
    if (!r.empty() && is_vowel_first_utf8(r)) {
      return {"\xC5\x8B", r};
    }
  }
  if (n >= 3 && body.compare(0, 2, "ch") == 0) {
    return {"c", body.substr(2)};
  }
  if (n >= 3 && body.compare(0, 2, "gh") == 0) {
    const std::string r = body.substr(2);
    if (!r.empty() && front_vowel_utf8(r)) {
      return {"\xC9\xA3", r};  // ɣ
    }
  }
  if (n >= 2 && body.compare(0, 2, "gi") == 0) {
    if (n == 2) {
      return {"\xC9\xA3", "i"};
    }
    const std::string r = body.substr(2);
    if (rime_is_only_i(r)) {
      return {"\xC9\xA3", r};
    }
    return {"z", r};
  }
  if (n >= 3 && body.compare(0, 2, "qu") == 0) {
    return {"kw", body.substr(2)};
  }
  if (n >= 3 && body.compare(0, 2, "tr") == 0) {
    return {"c", body.substr(2)};
  }
  if (n >= 3 && body.compare(0, 2, "th") == 0) {
    return {"t\xCA\xB0", body.substr(2)};  // tʰ
  }
  if (n >= 3 && body.compare(0, 2, "ph") == 0) {
    return {"f", body.substr(2)};
  }
  if (n >= 3 && body.compare(0, 2, "kh") == 0) {
    return {"x", body.substr(2)};
  }
  if (n >= 3 && body.compare(0, 2, "nh") == 0) {
    return {"\xC9\xB2", body.substr(2)};  // ɲ
  }
  if (body.size() >= 2) {
    char32_t cp0 = 0;
    size_t a0 = 0;
    utf8_decode_at(body, 0, cp0, a0);
    if (cp0 == 0x0111) {  // đ
      return {"d", body.substr(a0)};
    }
  }
  if (n >= 2 && body[0] == 'g' && body.compare(0, 2, "gh") != 0 && body.compare(0, 2, "gi") != 0) {
    return {"\xC9\xA3", body.substr(1)};
  }
  if (n >= 2) {
    const char c0 = body[0];
    if (c0 == 'b') {
      return {"b", body.substr(1)};
    }
    if (c0 == 'c' || c0 == 'k') {
      return {"k", body.substr(1)};
    }
    if (c0 == 'd') {
      return {"z", body.substr(1)};
    }
    if (c0 == 'h') {
      return {"h", body.substr(1)};
    }
    if (c0 == 'l') {
      return {"l", body.substr(1)};
    }
    if (c0 == 'm') {
      return {"m", body.substr(1)};
    }
    if (c0 == 'n' && body.compare(0, 3, "ngh") != 0 && body.compare(0, 2, "ng") != 0 &&
        body.compare(0, 2, "nh") != 0) {
      return {"n", body.substr(1)};
    }
    if (c0 == 'p') {
      return {"p", body.substr(1)};
    }
    if (c0 == 'r') {
      return {"z", body.substr(1)};
    }
    if (c0 == 's') {
      return {"s", body.substr(1)};
    }
    if (c0 == 't' && body.compare(0, 2, "th") != 0 && body.compare(0, 2, "tr") != 0) {
      return {"t", body.substr(1)};
    }
    if (c0 == 'v') {
      return {"v", body.substr(1)};
    }
    if (c0 == 'x') {
      return {"s", body.substr(1)};
    }
  }
  if (!body.empty() && is_vowel_first_utf8(body)) {
    return {{}, body};
  }
  return {{}, body};
}

const std::vector<std::string> kCodas = {"ch", "nh", "ng", "c", "k", "m", "n", "p", "t"};

std::pair<std::string, std::string> parse_rime(std::string_view rime_sv) {
  const std::string rime(rime_sv);
  if (rime.empty()) {
    return {{}, {}};
  }
  for (const std::string& cd : kCodas) {
    if (rime.size() > cd.size() && ends_with_str(rime, cd)) {
      return {rime.substr(0, rime.size() - cd.size()), cd};
    }
  }
  return {rime, {}};
}

bool wants_labial_coda(const std::string& nuc_ipa) {
  if (nuc_ipa.empty()) {
    return false;
  }
  if (nuc_ipa.find("\xC9\xAF") != std::string::npos || starts_with_sv(nuc_ipa, "\xC9\xAF")) {  // ɯ
    return false;
  }
  if (nuc_ipa.find("\xC9\xAF\xC9\x99") != std::string::npos) {  // ɯə
    return false;
  }
  return ends_with_str(nuc_ipa, "o") || ends_with_str(nuc_ipa, "\xC9\x94") || ends_with_str(nuc_ipa, "u") ||
         ends_with_str(nuc_ipa, "w") || ends_with_str(nuc_ipa, "\xC9\x99w") || ends_with_str(nuc_ipa, "ow");
}

std::string coda_simple(const std::string& coda, const std::string& nuc_ipa) {
  static const std::string kTie = "\xCD\xA1";
  const bool lab = wants_labial_coda(nuc_ipa);
  if (coda == "nh") {
    return "\xC5\x8B";
  }
  if (coda == "ch") {
    return "k";
  }
  if (coda == "ng") {
    return lab ? std::string("\xC5\x8B") + kTie + "m" : std::string("\xC5\x8B");
  }
  if (coda == "c" || coda == "k") {
    return lab ? std::string("k") + kTie + "p" : std::string("k");
  }
  if (coda == "n") {
    return "n";
  }
  if (coda == "m") {
    return "m";
  }
  if (coda == "p") {
    return "p";
  }
  if (coda == "t") {
    return "t";
  }
  return {};
}

// UTF-8 NFC multigraphs (lowercase), same order as ``vietnamese_rule_g2p._NUCLEUS_PREFIX``.
std::string nucleus_to_ipa(std::string_view nuc_sv) {
  std::string n(nuc_sv);
  std::string out;
  while (!n.empty()) {
    const char* p = n.c_str();
    auto take = [&](size_t bytes, const char* ipa_utf8) {
      out += ipa_utf8;
      n = n.substr(bytes);
    };
    // iêu  i + ê (2 bytes) + u
    if (n.size() >= 4 && p[0] == 'i' && static_cast<unsigned char>(p[1]) == 0xC3 &&
        static_cast<unsigned char>(p[2]) == 0xAA && p[3] == 'u') {
      take(4, "i\xC9\x99w");
      continue;
    }
    // ươi / ươu / ươ  (ư = C6 B0, ơ = C6 A1)
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC6 && static_cast<unsigned char>(p[1]) == 0xB0) {
      if (n.size() >= 4 && static_cast<unsigned char>(p[2]) == 0xC6 && static_cast<unsigned char>(p[3]) == 0xA1) {
        if (n.size() >= 5 && p[4] == 'i') {
          take(5, "\xC9\xAF\xC9\x99j");
          continue;
        }
        if (n.size() >= 5 && p[4] == 'u') {
          take(5, "\xC9\xAF\xC9\x99w");
          continue;
        }
        take(4, "\xC9\xAF\xC9\x99");
        continue;
      }
    }
    // iê / yê
    if (n.size() >= 3 && p[0] == 'i' && static_cast<unsigned char>(p[1]) == 0xC3 &&
        static_cast<unsigned char>(p[2]) == 0xAA) {
      take(3, "i\xC9\x99");
      continue;
    }
    if (n.size() >= 3 && p[0] == 'y' && static_cast<unsigned char>(p[1]) == 0xC3 &&
        static_cast<unsigned char>(p[2]) == 0xAA) {
      take(3, "i\xC9\x99");
      continue;
    }
    // uô
    if (n.size() >= 3 && p[0] == 'u' && static_cast<unsigned char>(p[1]) == 0xC3 &&
        static_cast<unsigned char>(p[2]) == 0xB4) {
      take(3, "uo");
      continue;
    }
    if (starts_with_sv(n, "oa")) {
      take(2, "wa");
      continue;
    }
    if (starts_with_sv(n, "oe")) {
      take(2, "w\xC9\x9B");
      continue;
    }
    if (starts_with_sv(n, "uy")) {
      take(2, "wj");
      continue;
    }
    if (starts_with_sv(n, "ai")) {
      take(2, "aj");
      continue;
    }
    if (starts_with_sv(n, "ay")) {
      take(2, "aj");
      continue;
    }
    if (starts_with_sv(n, "ao")) {
      take(2, "aw");
      continue;
    }
    if (starts_with_sv(n, "au")) {
      take(2, "aw");
      continue;
    }
    // âu / ây
    if (n.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xA2 &&
        p[2] == 'u') {
      take(3, "\xC9\x99w");
      continue;
    }
    if (n.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xA2 &&
        p[2] == 'y') {
      take(3, "\xC9\x99j");
      continue;
    }
    // ơi / ơu (ơ = C6 A1)
    if (n.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xC6 && static_cast<unsigned char>(p[1]) == 0xA1) {
      if (p[2] == 'i') {
        take(3, "\xC9\xA4j");
        continue;
      }
      if (p[2] == 'u') {
        take(3, "\xC9\xA4w");
        continue;
      }
    }
    // ưa / ưi / ưu
    if (n.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xC6 && static_cast<unsigned char>(p[1]) == 0xB0) {
      if (p[2] == 'a') {
        take(3, "\xC9\xAF\xC9\x99");
        continue;
      }
      if (p[2] == 'i') {
        take(3, "\xC9\xAFj");
        continue;
      }
      if (p[2] == 'u') {
        take(3, "\xC9\xAFw");
        continue;
      }
    }
    if (starts_with_sv(n, "ia")) {
      take(2, "i\xC9\x99");
      continue;
    }
    if (starts_with_sv(n, "iu")) {
      take(2, "iw");
      continue;
    }
    // êu
    if (n.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xAA &&
        p[2] == 'u') {
      take(3, "ew");
      continue;
    }
    // single letters (NFC lowercase)
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC6 && static_cast<unsigned char>(p[1]) == 0xA1) {
      take(2, "\xC9\xA4");  // ơ
      continue;
    }
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC6 && static_cast<unsigned char>(p[1]) == 0xB0) {
      take(2, "\xC9\xAF");  // ư
      continue;
    }
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xB4) {
      take(2, "o");  // ô
      continue;
    }
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xA2) {
      take(2, "\xC9\xA4\xCC\x86");  // â -> ɤ̆
      continue;
    }
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC4 && static_cast<unsigned char>(p[1]) == 0x83) {
      take(2, "\xC9\x90");  // ă -> ɐ
      continue;
    }
    if (n.size() >= 2 && static_cast<unsigned char>(p[0]) == 0xC3 && static_cast<unsigned char>(p[1]) == 0xAA) {
      take(2, "e");  // ê
      continue;
    }
    if (p[0] == 'e') {
      take(1, "\xC9\x9B");
      continue;
    }
    if (p[0] == 'o') {
      take(1, "\xC9\x94");
      continue;
    }
    if (p[0] == 'a') {
      take(1, "a");
      continue;
    }
    if (p[0] == 'i') {
      take(1, "i");
      continue;
    }
    if (p[0] == 'u') {
      take(1, "u");
      continue;
    }
    if (p[0] == 'y') {
      take(1, "i");
      continue;
    }
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(n, 0, cp, adv)) {
      n.erase(0, 1);
    } else {
      n = n.substr(adv);
    }
  }
  return out;
}

std::string combine_nucleus_coda(const std::string& nuc_orth, const std::string& nuc_ipa,
                                 const std::string& coda) {
  if (coda.empty()) {
    return nuc_ipa;
  }
  const std::string a_plain = "a";
  if (coda == "nh") {
    if (nuc_orth == a_plain || starts_with_sv(nuc_orth, "\xc3\xa1") || starts_with_sv(nuc_orth, "\xc3\xa0") ||
        starts_with_sv(nuc_orth, "\xe1\xba\xa3") || starts_with_sv(nuc_orth, "\xc3\xa3") ||
        starts_with_sv(nuc_orth, "\xe1\xba\xa1")) {
      return "\xC9\x9B\xC5\x8B";  // ɛŋ — same bytes as dict style
    }
    if (!nuc_orth.empty() && (nuc_orth[0] == '\xc3' || nuc_orth.find("\xc3\xaa") == 0)) {
      // ê…
      if (starts_with_sv(nuc_orth, "\xc3\xaa") || starts_with_sv(nuc_orth, "\xe1\xba\xbf") ||
          starts_with_sv(nuc_orth, "\xe1\xbb\x81") || starts_with_sv(nuc_orth, "\xe1\xbb\x83") ||
          starts_with_sv(nuc_orth, "\xe1\xbb\x85") || starts_with_sv(nuc_orth, "\xe1\xbb\x87")) {
        return "e\xC5\x8B";
      }
    }
    return nuc_ipa + coda_simple("nh", nuc_ipa);
  }
  if (coda == "ch") {
    if (nuc_orth == a_plain || starts_with_sv(nuc_orth, "\xc3\xa1") || starts_with_sv(nuc_orth, "\xc3\xa0") ||
        starts_with_sv(nuc_orth, "\xe1\xba\xa3") || starts_with_sv(nuc_orth, "\xc3\xa3") ||
        starts_with_sv(nuc_orth, "\xe1\xba\xa1")) {
      return "\xC9\x9Bk";  // ɛk
    }
    return nuc_ipa + coda_simple("ch", nuc_ipa);
  }
  return nuc_ipa + coda_simple(coda, nuc_ipa);
}

bool coda_obstruent_sac(const std::string& coda) {
  return coda == "ch" || coda == "c" || coda == "k" || coda == "p" || coda == "t";
}

std::string tone_suffix_ipa(int tone, const std::string& coda_orth) {
  // Bytes per Python vietnamese_rule_g2p._TONE_SUFFIX / _SẮC_OPEN
  static const std::string k1 = "\xcb\xa7\xcb\xa7";
  static const std::string k2 = "\xcb\xa7\xcb\xa8";
  static const std::string k3 = "\xcb\xa7\xcb\xa9\xcb\xa8";
  static const std::string k4 = "\xcb\xa7\xcb\x80\xcb\xa5";
  static const std::string k5o = "\xcb\xa6\xcb\xa5";
  static const std::string k5c = "\xcb\xa8\xcb\xa6";
  static const std::string k6 = "\xcb\xa8\xcb\x80\xcb\xa9";
  if (tone == 5 && !coda_obstruent_sac(coda_orth)) {
    return k5c;
  }
  switch (tone) {
  case 1:
    return k1;
  case 2:
    return k2;
  case 3:
    return k3;
  case 4:
    return k4;
  case 5:
    return k5o;
  case 6:
    return k6;
  default:
    return k1;
  }
}

std::string apply_tone(const std::string& base, int tone, bool has_coda, const std::string& coda_orth,
                       const std::string& nuc_ipa) {
  std::string suf = tone_suffix_ipa(tone, coda_orth);
  if (tone == 6 && !has_coda && !base.empty()) {
    return base + suf + "\xCA\x94";  // ʔ
  }
  if (tone == 6 && has_coda && coda_orth == "ng" && wants_labial_coda(nuc_ipa)) {
    return base + suf + "\xCA\x94";
  }
  return base + suf;
}

bool is_unicode_edge_punct(char32_t cp, bool leading) {
  if (cp <= 0x7F) {
    const char* s = leading ? "\"'([{" : "\")]},.:;!?";
    return cp >= 32 && std::strchr(s, static_cast<char>(cp)) != nullptr;
  }
  switch (cp) {
  case U'«':
  case U'»':
  case U'‹':
  case U'›':
  case U'“':
  case U'”':
  case U'‘':
  case U'’':
  case U'…':
    return true;
  default:
    return false;
  }
}

/// Strip leading/trailing punctuation by **code point** (never ``strchr`` on UTF-8 bytes — a byte like
/// 0x99 can appear inside multi-byte punctuation and corrupt syllables ending in ``ộ``).
std::string strip_edge_punct(std::string_view tok) {
  std::string t = utf8_nfc_utf8proc(trim_ascii_ws_copy(tok));
  for (;;) {
    if (t.empty()) {
      break;
    }
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(t, 0, cp, adv)) {
      break;
    }
    if (!is_unicode_edge_punct(cp, true)) {
      break;
    }
    t = t.substr(adv);
  }
  for (;;) {
    if (t.empty()) {
      break;
    }
    size_t i = 0;
    size_t last = 0;
    size_t last_adv = 0;
    while (i < t.size()) {
      char32_t cp = 0;
      size_t adv = 0;
      if (!utf8_decode_at(t, i, cp, adv)) {
        break;
      }
      last = i;
      last_adv = adv;
      i += adv;
    }
    char32_t cp = 0;
    if (!utf8_decode_at(t, last, cp, last_adv)) {
      break;
    }
    if (!is_unicode_edge_punct(cp, false)) {
      break;
    }
    t.erase(last, last_adv);
  }
  return utf8_nfc_utf8proc(trim_ascii_ws_copy(t));
}

int max_lex_key_words(const std::unordered_map<std::string, std::string>& lex) {
  int m = 1;
  for (const auto& pr : lex) {
    int w = 1;
    for (char c : pr.first) {
      if (c == ' ') {
        ++w;
      }
    }
    m = std::max(m, w);
  }
  return m;
}

void load_vietnamese_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::string>& lex) {
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
    std::string key = utf8_nfc_utf8proc(trim_ascii_ws_copy(std::string_view(line).substr(0, tab)));
    std::string ipa = trim_ascii_ws_copy(std::string_view(line).substr(tab + 1));
    if (key.empty()) {
      continue;
    }
    if (lex.find(key) == lex.end()) {
      lex.emplace(std::move(key), std::move(ipa));
    }
  }
}

}  // namespace

std::string VietnameseRuleG2p::syllable_to_ipa(std::string_view syllable_utf8) {
  const std::string trimmed = trim_ascii_ws_copy(syllable_utf8);
  if (trimmed.empty()) {
    return {};
  }
  const std::string raw = utf8_lower_nfc(trimmed);
  std::string body;
  const int tone = split_tone(raw, body);
  auto [onset_ipa, rime] = parse_onset(body);
  if (rime.empty()) {
    return {};
  }
  auto [nuc_orth, coda_orth] = parse_rime(rime);
  std::string nuc_ipa = nucleus_to_ipa(nuc_orth);
  if (nuc_ipa.empty() && onset_ipa.empty() && coda_orth.empty()) {
    return {};
  }
  std::string rime_ipa =
      coda_orth.empty() ? nuc_ipa : combine_nucleus_coda(nuc_orth, nuc_ipa, coda_orth);
  const std::string base = onset_ipa + rime_ipa;
  if (base.empty()) {
    return {};
  }
  return apply_tone(base, tone, !coda_orth.empty(), coda_orth, nuc_ipa);
}

VietnameseRuleG2p::VietnameseRuleG2p(std::filesystem::path dict_tsv) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("Vietnamese G2P: lexicon not found at " + dict_tsv.generic_string());
  }
  std::ifstream in(dict_tsv);
  if (!in) {
    throw std::runtime_error("Vietnamese G2P: cannot open " + dict_tsv.generic_string());
  }
  load_vietnamese_lexicon_stream(in, lex_);
  if (lex_.empty()) {
    throw std::runtime_error("Vietnamese G2P: empty lexicon: " + dict_tsv.generic_string());
  }
  max_key_words_ = max_lex_key_words(lex_);
}

VietnameseRuleG2p::VietnameseRuleG2p(std::string dict_tsv_utf8) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_vietnamese_lexicon_stream(in, lex_);
  if (lex_.empty()) {
    throw std::runtime_error("Vietnamese G2P: empty lexicon (in-memory)");
  }
  max_key_words_ = max_lex_key_words(lex_);
}

std::string VietnameseRuleG2p::word_to_ipa(std::string_view word) const {
  return g2p_single_token(word);
}

std::string VietnameseRuleG2p::g2p_single_token(std::string_view token) const {
  const std::string w = utf8_nfc_utf8proc(strip_edge_punct(token));
  if (w.empty()) {
    return {};
  }
  auto it = lex_.find(w);
  if (it != lex_.end()) {
    return it->second;
  }
  return syllable_to_ipa(w);
}

std::string VietnameseRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  const std::string raw = utf8_nfc_utf8proc(trim_ascii_ws_copy(text));
  if (raw.empty()) {
    return {};
  }
  std::vector<std::string> tokens;
  {
    std::istringstream iss(raw);
    std::string w;
    while (iss >> w) {
      tokens.push_back(std::move(w));
    }
  }
  std::vector<std::string> ipa_words;
  for (size_t pos = 0; pos < tokens.size();) {
    bool matched = false;
    const int max_span = std::min(max_key_words_, static_cast<int>(tokens.size() - pos));
    for (int span = max_span; span >= 1; --span) {
      std::string key;
      for (int j = 0; j < span; ++j) {
        if (j > 0) {
          key.push_back(' ');
        }
        key += strip_edge_punct(tokens[pos + j]);
      }
      key = utf8_nfc_utf8proc(key);
      auto it = lex_.find(key);
      if (it != lex_.end()) {
        if (per_word_log != nullptr) {
          per_word_log->push_back(G2pWordLog{key, key, G2pWordPath::kRuleBasedG2p, it->second});
        }
        if (!it->second.empty()) {
          ipa_words.push_back(it->second);
        }
        pos += static_cast<size_t>(span);
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }
    const std::string t = strip_edge_punct(tokens[pos]);
    ++pos;
    if (t.empty()) {
      continue;
    }
    bool all_ascii = true;
    for (unsigned char c : t) {
      if (c > 127) {
        all_ascii = false;
        break;
      }
      if (!(std::isalpha(c) || c == '-' || c == '\'')) {
        all_ascii = false;
        break;
      }
    }
    if (all_ascii && !t.empty()) {
      std::string lo;
      lo.reserve(t.size());
      for (unsigned char c : t) {
        lo.push_back(static_cast<char>(std::tolower(c)));
      }
      ipa_words.push_back(std::move(lo));
      continue;
    }
    if (t.find('-') != std::string::npos && t.front() != '-') {
      std::string part;
      std::vector<std::string> subs;
      for (size_t i = 0; i <= t.size();) {
        const size_t j = t.find('-', i);
        const size_t end = (j == std::string::npos) ? t.size() : j;
        if (end > i) {
          subs.push_back(g2p_single_token(std::string_view(t).substr(i, end - i)));
        }
        if (j == std::string::npos) {
          break;
        }
        i = j + 1;
      }
      std::string joined;
      for (size_t i = 0; i < subs.size(); ++i) {
        if (subs[i].empty()) {
          continue;
        }
        if (!joined.empty()) {
          joined.push_back('-');
        }
        joined += subs[i];
      }
      if (!joined.empty()) {
        ipa_words.push_back(std::move(joined));
      }
      continue;
    }
    const std::string wipa = g2p_single_token(t);
    if (!wipa.empty()) {
      ipa_words.push_back(wipa);
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

std::vector<std::string> VietnameseRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first(
      {"vi", "vi-VN", "vi_vn", "vie", "vietnamese", "Vietnamese"});
}

bool dialect_resolves_to_vietnamese_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "vi" || s == "vi-vn" || s == "vie" || s == "vietnamese";
}

std::filesystem::path resolve_vietnamese_dict_path(const std::filesystem::path& model_root) {
  return model_root / "vi" / "dict.tsv";
}

}  // namespace moonshine_tts
