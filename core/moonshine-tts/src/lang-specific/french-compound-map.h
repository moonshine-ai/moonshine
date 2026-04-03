#ifndef MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_COMPOUND_MAP_H
#define MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_COMPOUND_MAP_H

#include <string>
#include <unordered_map>

namespace moonshine_tts::french_compound_map {

const std::unordered_map<std::string, std::string>& cardinal_compound_ipa_entries();

}  // namespace moonshine_tts::french_compound_map

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_COMPOUND_MAP_H
