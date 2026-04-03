#ifndef MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_RULES_H
#define MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_RULES_H

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace moonshine_tts::portuguese_rules {

char32_t pt_tolower(char32_t c);
std::string normalize_lookup_key_utf8(const std::string& word);

std::optional<std::string> roman_numeral_token_to_ipa(const std::string& letters_lower,
                                                                   bool is_pt_pt);
std::string rules_word_to_ipa_utf8(const std::string& raw, bool is_pt_pt, bool with_stress);
std::string pt_pt_apply_rules_final_s_to_esh(std::string ipa, const std::string& letters_key);

const std::unordered_map<std::string, std::string>& fw_pt();
const std::unordered_map<std::string, std::string>& fw_br();
const std::unordered_map<std::string, std::string>& sc_straddle();

}  // namespace moonshine_tts::portuguese_rules

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_RULES_H
