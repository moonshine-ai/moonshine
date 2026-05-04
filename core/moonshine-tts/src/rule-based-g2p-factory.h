#ifndef MOONSHINE_TTS_RULE_BASED_G2P_FACTORY_H
#define MOONSHINE_TTS_RULE_BASED_G2P_FACTORY_H

#include "moonshine-g2p-options.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

class RuleBasedG2p;

enum class RuleBasedG2pKind {
  English,
  Spanish,
  German,
  French,
  Dutch,
  Italian,
  Russian,
  Chinese,
  Korean,
  Vietnamese,
  Japanese,
  Arabic,
  Portuguese,
  Turkish,
  Ukrainian,
  Hindi,
};

struct RuleBasedG2pInstance {
  std::unique_ptr<RuleBasedG2p> engine;
  std::string canonical_dialect_id;
  RuleBasedG2pKind kind;
};

/// Try to construct a rule-based engine for *dialect_id* (trimmed). Returns nullopt if unknown.
/// Order: English, Spanish, German, French, Dutch, Italian, Russian, Chinese, Korean, Vietnamese,
/// Japanese, Arabic, Portuguese, Turkish, Ukrainian, Hindi.
std::optional<RuleBasedG2pInstance> create_rule_based_g2p(
    std::string_view dialect_id,
    const MoonshineG2POptions& options);

/// All static ``dialect_ids()`` from each rule engine, in factory resolution order (for discovery).
std::vector<std::pair<RuleBasedG2pKind, std::vector<std::string>>> rule_based_g2p_dialect_catalog();

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_RULE_BASED_G2P_FACTORY_H
