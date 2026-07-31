#ifndef MOONSHINE_TTS_PIPER_VOICE_CATALOG_H
#define MOONSHINE_TTS_PIPER_VOICE_CATALOG_H

#include <string>
#include <vector>

namespace moonshine_tts {

/// ONNX stems (no ``.onnx``) shipped under
/// ``moonshine-tts/data/<data_subdir>/piper-voices/``. Used to populate voice
/// lists even when only a subset is downloaded to ``g2p_root``.
const std::vector<std::string>& piper_bundled_voice_stems_for_data_subdir(
    const std::string& data_subdir);

/// Model file names a voice ships as, relative to its ``piper-voices``
/// directory, in the order a client must fetch them.
///
/// Voices whose weights are stored as int8 ship as a split ORT pair
/// (``<stem>.model.ort`` plus ``<stem>.weights.ort``); the rest ship as a single
/// ``<stem>.ort``. See ``scripts/convert-models-to-ort.py`` for why the two
/// forms exist. Callers must not consult the local disk instead: this list is
/// what a client with no files yet needs to download.
std::vector<std::string> piper_voice_model_filenames(const std::string& stem);

/// Whether a voice ships as a split ORT pair rather than a single ``.ort``.
bool piper_voice_ships_split(const std::string& stem);

}  // namespace moonshine_tts

#endif
