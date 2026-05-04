#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ARABIC_IPA_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ARABIC_IPA_H

#include <string>
#include <string_view>

namespace moonshine_tts {

std::string arabic_msa_strip_diacritics_utf8(std::string_view utf8);
std::string arabic_msa_apply_onnx_partial_postprocess_utf8(std::string_view utf8);
/// *assimilation_prefix_source_utf8* (when non-empty) is the pre-ONNX word; sun-letter assimilation uses its
/// undiacritized prefix so leading punctuation matches ``arabic_ipa.word_to_ipa_with_assimilation`` (the
/// diacritizer may drop non-Arabic brackets from *filled_diac_utf8*).
std::string arabic_msa_word_to_ipa_with_assimilation_utf8(
    std::string_view filled_diac_utf8,
    std::string_view assimilation_prefix_source_utf8 = {});

}  // namespace moonshine_tts

#endif
