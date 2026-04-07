#ifndef MOONSHINE_TTS_IPA_POSTPROCESS_H
#define MOONSHINE_TTS_IPA_POSTPROCESS_H

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace moonshine_tts {

/// NFC (Unicode canonical composition) via utf8proc; frees the same way as other call sites.
std::string utf8_nfc_copy(std::string_view ipa_utf8);

/// Replace ASCII ``c`` + U+0327 COMBINING CEDILLA with precomposed ``ç`` (espeak NFD quirk) before Piper
/// ``phoneme_id_map`` tokenization.
void repair_ascii_c_combining_cedilla_to_ccedilla_utf8(std::string& ipa_utf8);

/// Rewrite Russian IPA from Moonshine rule/lexicon G2P toward Piper / Kokoro / espeak-ng-like symbols
/// (NFC; combining acute U+0301 → ˈ on the nucleus; affricate digraphs; non-retroflex fricatives;
/// stress+nucleus units like ˈɨ→ˈy before bare ɨ→y; ɫ→ɭ, ʉ→u, ʌ→a, ɪ→i, ʊ→u; *что* as /ʃto/, etc.;
/// boundary ASCII ``i`` may become ɪ after vowel-to-ASCII passes).
std::string normalize_russian_ipa_piper_style(std::string ipa);

/// Rewrite German IPA from Moonshine lexicon/rules toward Piper / espeak-ng-style strings (NFC;
/// tie-bar affricates → digraphs; ɐ̯ and ʁ → ɾ; ``.:`` from abbreviations → ``. `` for cleaner
/// word-boundary spacing before the next phoneme run).
std::string normalize_german_ipa_piper_style(std::string ipa);

/// NFC, then shared + per-language substring replacements (``piper_lang_key`` matches Python
/// ``piper_ipa_normalization.normalize_g2p_ipa_for_piper``, e.g. ``en_us``, ``de``, ``es_mx``).
std::string normalize_g2p_ipa_for_piper(std::string_view ipa_utf8, std::string_view piper_lang_key);

/// Map unknown codepoints into the Piper ``phoneme_id_map`` key set: drop Mn/Me/P*/S* (when not in
/// map); otherwise closest Unicode scalar in the IPA-like inventory pool (deterministic ties).
std::string coerce_unknown_ipa_chars_to_piper_inventory(std::string_view ipa_utf8,
                                                        const std::unordered_set<std::string>& phoneme_id_map_keys,
                                                        bool use_closest_scalar = true);

/// NFC → ``normalize_g2p_ipa_for_piper`` → optional ``coerce_unknown_ipa_chars_to_piper_inventory``.
std::string ipa_to_piper_ready(std::string_view ipa_utf8, std::string_view piper_lang_key,
                               const std::unordered_set<std::string>& phoneme_id_map_keys,
                               bool apply_coercion = true);

/// Shared substring replacements only (caller supplies NFC IPA). Same rules as the first stage of
/// ``normalize_g2p_ipa_for_piper`` without NFC or per-language tables.
std::string normalize_g2p_ipa_for_piper_engines(std::string_view ipa_utf8);

std::vector<std::string> ipa_string_to_phoneme_tokens(const std::string& s);

int levenshtein_distance(const std::vector<std::string>& a, const std::vector<std::string>& b);

int pick_closest_alternative_index(const std::vector<std::string>& predicted_phoneme_tokens,
                                    const std::vector<std::string>& ipa_alternatives,
                                    int n_valid,
                                    int extra_phonemes);

std::string pick_closest_cmudict_ipa(const std::vector<std::string>& predicted_phoneme_tokens,
                                     const std::vector<std::string>& cmudict_alternatives,
                                     int extra_phonemes);

// Returns canonical string from alternatives or nullopt (Python None).
std::optional<std::string> match_prediction_to_cmudict_ipa(const std::string& predicted,
                                                             const std::vector<std::string>& alts);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_IPA_POSTPROCESS_H
