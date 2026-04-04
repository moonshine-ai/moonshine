#ifndef MOONSHINE_TTS_CMUDICT_TSV_H
#define MOONSHINE_TTS_CMUDICT_TSV_H

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

// word key (normalized grapheme) -> sorted unique IPA strings (TSV: word<TAB>ipa).
class CmudictTsv {
 public:
  explicit CmudictTsv(const std::filesystem::path& path);
  explicit CmudictTsv(std::string_view utf8_contents);

  const std::vector<std::string>* lookup(std::string_view key) const;

 private:
  std::unordered_map<std::string, std::vector<std::string>> ipa_by_word_;
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_CMUDICT_TSV_H
