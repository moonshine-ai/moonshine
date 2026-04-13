// Bundled Piper ONNX stems, kept in sync with ``moonshine-tts/data/*/piper-voices/*.onnx``.
// Regenerate the initializer from that tree when adding voices (see repo ``data`` layout).

#include "piper-voice-catalog.h"

#include <unordered_map>

namespace moonshine_tts {

const std::vector<std::string>& piper_bundled_voice_stems_for_data_subdir(const std::string& data_subdir) {
  static const std::unordered_map<std::string, std::vector<std::string>> k = [] {
    std::unordered_map<std::string, std::vector<std::string>> m;
    m["ar_msa"] = std::vector<std::string>{
        "ar_JO-kareem-low",
        "ar_JO-kareem-medium",
    };
    m["de"] = std::vector<std::string>{
        "de_DE-eva_k-x_low",
        "de_DE-karlsson-low",
        "de_DE-kerstin-low",
        "de_DE-mls-medium",
        "de_DE-pavoque-low",
        "de_DE-ramona-low",
        "de_DE-thorsten-high",
        "de_DE-thorsten-low",
        "de_DE-thorsten-medium",
        "de_DE-thorsten_emotional-medium",
    };
    m["en_gb"] = std::vector<std::string>{
        "en_GB-alan-low",
        "en_GB-alan-medium",
        "en_GB-alba-medium",
        "en_GB-aru-medium",
        "en_GB-cori-high",
        "en_GB-cori-medium",
        "en_GB-jenny_dioco-medium",
        "en_GB-northern_english_male-medium",
        "en_GB-semaine-medium",
        "en_GB-southern_english_female-low",
        "en_GB-vctk-medium",
    };
    m["en_us"] = std::vector<std::string>{
        "en_US-amy-low",
        "en_US-amy-medium",
        "en_US-arctic-medium",
        "en_US-bryce-medium",
        "en_US-danny-low",
        "en_US-hfc_female-medium",
        "en_US-hfc_male-medium",
        "en_US-joe-medium",
        "en_US-john-medium",
        "en_US-kathleen-low",
        "en_US-kristin-medium",
        "en_US-kusal-medium",
        "en_US-l2arctic-medium",
        "en_US-lessac-high",
        "en_US-lessac-low",
        "en_US-lessac-medium",
        "en_US-libritts-high",
        "en_US-libritts_r-medium",
        "en_US-ljspeech-high",
        "en_US-ljspeech-medium",
        "en_US-norman-medium",
        "en_US-reza_ibrahim-medium",
        "en_US-ryan-high",
        "en_US-ryan-low",
        "en_US-ryan-medium",
        "en_US-saikat",
        "en_US-sam-medium",
    };
    m["es_ar"] = std::vector<std::string>{
        "es_AR-daniela-high",
    };
    m["es_es"] = std::vector<std::string>{
        "es_ES-carlfm-x_low",
        "es_ES-davefx-medium",
        "es_ES-mls_10246-low",
        "es_ES-mls_9972-low",
        "es_ES-sharvard-medium",
    };
    m["es_mx"] = std::vector<std::string>{
        "es_MX-ald-medium",
        "es_MX-claude-high",
    };
    m["fr"] = std::vector<std::string>{
        "fr_FR-gilles-low",
        "fr_FR-mls-medium",
        "fr_FR-mls_1840-low",
        "fr_FR-siwis-low",
        "fr_FR-siwis-medium",
        "fr_FR-tom-medium",
        "fr_FR-upmc-medium",
    };
    m["hi"] = std::vector<std::string>{
        "hi_IN-pratham-medium",
        "hi_IN-priyamvada-medium",
    };
    m["it"] = std::vector<std::string>{
        "it_IT-paola-medium",
        "it_IT-riccardo-x_low",
    };
    m["ko"] = std::vector<std::string>{
        "ko_KR-melotts-medium",
    };
    m["nl"] = std::vector<std::string>{
        "nl_BE-nathalie-medium",
        "nl_BE-nathalie-x_low",
        "nl_BE-rdh-medium",
        "nl_BE-rdh-x_low",
        "nl_NL-mls-medium",
        "nl_NL-mls_5809-low",
        "nl_NL-mls_7432-low",
        "nl_NL-pim-medium",
        "nl_NL-ronnie-medium",
    };
    m["pt_br"] = std::vector<std::string>{
        "pt_BR-cadu-medium",
        "pt_BR-edresson-low",
        "pt_BR-faber-medium",
        "pt_BR-jeff-medium",
    };
    m["pt_pt"] = std::vector<std::string>{
        "pt_PT-tugão-medium",
    };
    m["ru"] = std::vector<std::string>{
        "ru_RU-denis-medium",
        "ru_RU-dmitri-medium",
        "ru_RU-irina-medium",
        "ru_RU-ruslan-medium",
    };
    m["tr"] = std::vector<std::string>{
        "tr_TR-dfki-medium",
        "tr_TR-fahrettin-medium",
        "tr_TR-fettah-medium",
    };
    m["uk"] = std::vector<std::string>{
        "uk_UA-lada-x_low",
        "uk_UA-ukrainian_tts-medium",
    };
    m["vi"] = std::vector<std::string>{
        "vi_VN-25hours_single-low",
        "vi_VN-vais1000-medium",
        "vi_VN-vivos-x_low",
    };
    m["zh_hans"] = std::vector<std::string>{
        "zh_CN-huayan-medium",
        "zh_CN-huayan-x_low",
    };
    return m;
  }();
  static const std::vector<std::string> k_empty{};
  const auto it = k.find(data_subdir);
  return it == k.end() ? k_empty : it->second;
}

}  // namespace moonshine_tts
