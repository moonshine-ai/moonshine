#ifndef MOONSHINE_TTS_G2P_TRANSFORMER_MODEL_H
#define MOONSHINE_TTS_G2P_TRANSFORMER_MODEL_H

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "split-weights.h"

namespace moonshine_tts {

struct MoonshineG2POptions;

/// A loaded transformer G2P model, in whichever form it shipped.
struct G2pTransformerModel {
  std::unique_ptr<Ort::Session> session;
  /// Non-empty only for the split ORT pair, in which case these must be passed
  /// as extra inputs on every run. See split-weights.h.
  std::vector<SplitWeight> split_weights;
  /// Backing store for a session created from a memory buffer, which ORT does
  /// not copy and so must outlive the session.
  std::vector<std::uint8_t> model_bytes;

  bool is_split() const { return !split_weights.empty(); }
};

/// Loads *model_file* from *opt*'s asset bundle when available, else from disk
/// under *model_dir*.
///
/// Prefers the split ORT pair (``<stem>.model.ort`` plus
/// ``<stem>.weights.ort``) and falls back to a single model, which itself
/// resolves a sibling ``.ort`` ahead of the named ``.onnx``. *owner* only
/// labels error messages.
G2pTransformerModel load_g2p_transformer_model(
    Ort::Env& env, const MoonshineG2POptions* opt, std::string_view bundle_key,
    std::string_view model_file, const std::filesystem::path& model_dir,
    const std::vector<std::string>& ort_providers,
    const std::string& coreml_cache_dir, std::string_view owner);

/// The split pair's file names for a model named *model_file*.
std::string g2p_split_model_file(std::string_view model_file);
std::string g2p_split_weights_file(std::string_view model_file);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_G2P_TRANSFORMER_MODEL_H
