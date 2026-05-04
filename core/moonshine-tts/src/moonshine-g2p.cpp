#include "moonshine-g2p.h"
#include "debug-utils.h"
#include "rule-based-g2p-factory.h"
#include "rule-based-g2p.h"
#include "dutch.h"
#include "english.h"
#include "french.h"
#include "german.h"
#include "chinese.h"
#include "korean.h"
#include "vietnamese.h"
#include "japanese.h"
#include "arabic.h"
#include "italian.h"
#include "portuguese.h"
#include "russian.h"
#include "spanish.h"
#include "turkish.h"
#include "ukrainian.h"
#include "hindi.h"
#include "utf8-utils.h"

#include <cctype>
#include <stdexcept>

namespace moonshine_tts {

namespace {

std::string trim_copy(std::string_view s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])) != 0) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) {
    --b;
  }
  return std::string(s.substr(a, b - a));
}

/// Normalize user input like ``es_ar`` / ``es-mx`` to keys accepted by
/// ``spanish_dialect_from_cli_id`` (e.g. ``es-AR``, ``es-MX``).
std::string normalize_spanish_dialect_cli_key(std::string_view raw) {
  std::string s = normalize_rule_based_dialect_cli_key(raw);
  if (s.size() >= 3 && s[0] == 'e' && s[1] == 's' && s[2] == '-') {
    size_t i = 3;
    while (i < s.size() && s[i] != '-') {
      if (std::isalpha(static_cast<unsigned char>(s[i])) != 0) {
        s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
      }
      ++i;
    }
    if (i < s.size() && s[i] == '-') {
      ++i;
      while (i < s.size()) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        ++i;
      }
    }
  }
  return s;
}

const char* rule_backend_name(RuleBasedG2pKind k) {
  switch (k) {
    case RuleBasedG2pKind::English:    return "English";
    case RuleBasedG2pKind::Spanish:    return "Spanish";
    case RuleBasedG2pKind::German:     return "German";
    case RuleBasedG2pKind::French:     return "French";
    case RuleBasedG2pKind::Dutch:      return "Dutch";
    case RuleBasedG2pKind::Italian:    return "Italian";
    case RuleBasedG2pKind::Russian:    return "Russian";
    case RuleBasedG2pKind::Chinese:    return "Chinese";
    case RuleBasedG2pKind::Korean:     return "Korean";
    case RuleBasedG2pKind::Vietnamese: return "Vietnamese";
    case RuleBasedG2pKind::Japanese:   return "Japanese";
    case RuleBasedG2pKind::Arabic:     return "Arabic";
    case RuleBasedG2pKind::Portuguese: return "Portuguese";
    case RuleBasedG2pKind::Turkish:    return "Turkish";
    case RuleBasedG2pKind::Ukrainian:  return "Ukrainian";
    case RuleBasedG2pKind::Hindi:      return "Hindi";
  }
  return "Unknown";
}

}  // namespace

bool dialect_resolves_to_spanish_rules(std::string_view dialect_id, bool spanish_narrow_obstruents) {
  const std::string spanish_key = normalize_spanish_dialect_cli_key(dialect_id);
  if (spanish_key.empty()) {
    return false;
  }
  try {
    (void)spanish_dialect_from_cli_id(spanish_key, spanish_narrow_obstruents);
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

bool dialect_uses_rule_based_g2p(std::string_view dialect_id, const MoonshineG2POptions& options) {
  const std::string norm = normalize_rule_based_dialect_cli_key(dialect_id);
  if (norm.empty()) {
    return false;
  }
  if (dialect_resolves_to_english_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_spanish_rules(norm, options.spanish_narrow_obstruents)) {
    return true;
  }
  if (dialect_resolves_to_german_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_french_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_dutch_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_italian_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_russian_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_chinese_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_korean_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_vietnamese_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_japanese_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_arabic_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_brazilian_portuguese_rules(norm) || dialect_resolves_to_portugal_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_turkish_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_ukrainian_rules(norm)) {
    return true;
  }
  if (dialect_resolves_to_hindi_rules(norm)) {
    return true;
  }
  return false;
}

MoonshineG2P::~MoonshineG2P() = default;

MoonshineG2P::MoonshineG2P(MoonshineG2P&&) noexcept = default;
MoonshineG2P& MoonshineG2P::operator=(MoonshineG2P&&) noexcept = default;

MoonshineG2P::MoonshineG2P(std::string dialect_id, MoonshineG2POptions options) {
  log_profiling_ = options.log_profiling;
  TIMER_START_IF(log_profiling_, g2p_init);
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    throw std::invalid_argument("empty dialect id");
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(trimmed);
  if (norm.empty()) {
    throw std::invalid_argument("empty dialect id");
  }

  if (auto rb = create_rule_based_g2p(norm, options)) {
    dialect_id_ = std::move(rb->canonical_dialect_id);
    rules_ = std::move(rb->engine);
    rule_backend_ = rb->kind;
    LOGF_IF(log_profiling_, "MoonshineG2P: dialect='%s', rule backend=%s",
            dialect_id_.c_str(), rule_backend_name(rb->kind));
    TIMER_END_IF(log_profiling_, g2p_init);
    return;
  }

  throw std::runtime_error(
      "MoonshineG2P: unsupported dialect \"" + trimmed +
      "\". Only rule-based locales are supported (e.g. en_us, en_gb, es-MX, de, fr, nl, it, ru, zh, ko, vi, ja, ar, pt_br, tr, uk, hi); "
      "see dialect_uses_rule_based_g2p() and rule_based_g2p_dialect_catalog().");
}

std::string MoonshineG2P::text_to_ipa(std::string_view text, std::vector<G2pWordLog>* per_word_log) {
  TIMER_START_IF(log_profiling_, g2p_text_to_ipa);
  if (rules_) {
    std::string result = rules_->text_to_ipa(std::string(text), per_word_log);
    LOGF_IF(log_profiling_, "MoonshineG2P: text_to_ipa input='%.*s'%s",
            (int)std::min(text.size(), (size_t)200), text.data(),
            text.size() > 200 ? "..." : "");
    LOGF_IF(log_profiling_, "MoonshineG2P: text_to_ipa IPA output='%.*s'%s",
            (int)std::min(result.size(), (size_t)500), result.c_str(),
            result.size() > 500 ? "..." : "");
    TIMER_END_IF(log_profiling_, g2p_text_to_ipa);
    return result;
  }
  throw std::logic_error("MoonshineG2P: no backend initialized");
}

}  // namespace moonshine_tts
