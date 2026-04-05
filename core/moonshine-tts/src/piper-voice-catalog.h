#ifndef MOONSHINE_TTS_PIPER_VOICE_CATALOG_H
#define MOONSHINE_TTS_PIPER_VOICE_CATALOG_H

#include <string>
#include <vector>

namespace moonshine_tts {

/// ONNX stems (no ``.onnx``) shipped under ``moonshine-tts/data/<data_subdir>/piper-voices/``.
/// Used to populate voice lists even when only a subset is downloaded to ``g2p_root``.
const std::vector<std::string>& piper_bundled_voice_stems_for_data_subdir(const std::string& data_subdir);

}  // namespace moonshine_tts

#endif
