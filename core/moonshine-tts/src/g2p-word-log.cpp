#include "g2p-word-log.h"

#include <sstream>

namespace moonshine_tts {

const char* g2p_word_path_tag(G2pWordPath path) {
  switch (path) {
    case G2pWordPath::kSkippedEmptyKey:
      return "skip-empty-key";
    case G2pWordPath::kTokenNotLocatedInText:
      return "skip-not-located";
    case G2pWordPath::kRuleBasedG2p:
      return "rules";
    case G2pWordPath::kDictUnambiguous:
      return "dict";
    case G2pWordPath::kDictHeteronym:
      return "heteronym";
    case G2pWordPath::kDictFirstAlternativeNoHeteronymModel:
      return "dict-first-alt";
    case G2pWordPath::kOovModel:
      return "oov";
    case G2pWordPath::kOovModelNoOutput:
      return "oov-empty";
    case G2pWordPath::kUnknownNoOovModel:
      return "unknown-no-oov";
    case G2pWordPath::kEnglishNumber:
      return "english-number";
    case G2pWordPath::kOovHandRules:
      return "oov-hand-rules";
  }
  return "?";
}

std::string format_g2p_word_log_line(const G2pWordLog& e) {
  std::ostringstream o;
  o << '[' << g2p_word_path_tag(e.path) << "] token=";
  if (e.surface_token.empty()) {
    o << "(empty)";
  } else {
    o << '"' << e.surface_token << '"';
  }
  o << " key=";
  if (e.lookup_key.empty()) {
    o << "(empty)";
  } else {
    o << '"' << e.lookup_key << '"';
  }
  if (!e.ipa.empty()) {
    o << " ipa=\"" << e.ipa << '"';
  }
  return o.str();
}

}  // namespace moonshine_tts
