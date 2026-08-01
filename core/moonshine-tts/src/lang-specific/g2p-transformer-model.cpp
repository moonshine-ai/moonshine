#include "g2p-transformer-model.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "g2p-path.h"
#include "moonshine-g2p-options.h"
#include "ort-session-options.h"

namespace moonshine_tts {

namespace {

std::string stem_of(std::string_view model_file) {
  const size_t dot = model_file.rfind('.');
  if (dot == std::string_view::npos) {
    return std::string(model_file);
  }
  return std::string(model_file.substr(0, dot));
}

struct FileCloser {
  void operator()(std::FILE* file) const { std::fclose(file); }
};

/// Reads *path* into *out*.
///
/// Uses stdio rather than ifstream because ``fread`` takes ``void*`` and so
/// needs no cast to fill a byte vector, and because a weights model can run to
/// a hundred megabytes, which rules out reading through a char-based
/// intermediate buffer.
bool read_file(const std::filesystem::path& path,
               std::vector<std::uint8_t>& out) {
#ifdef _WIN32
  std::unique_ptr<std::FILE, FileCloser> file(
      _wfopen(path.wstring().c_str(), L"rb"));
#else
  std::unique_ptr<std::FILE, FileCloser> file(
      std::fopen(path.string().c_str(), "rb"));
#endif
  if (!file) {
    return false;
  }
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    return false;
  }
  out.resize(static_cast<size_t>(size));
  if (out.empty()) {
    return true;
  }
  return std::fread(out.data(), 1, out.size(), file.get()) == out.size();
}

/// Loads *file* by exact name: from the bundle when present, else from disk.
bool load_exact(const MoonshineG2POptions* opt, std::string_view bundle_key,
                std::string_view file, const std::filesystem::path& disk_dir,
                std::vector<std::uint8_t>& out) {
  if (opt != nullptr && !bundle_key.empty()) {
    const std::string key = g2p_bundle_file_key(bundle_key, file);
    if (opt->asset_is_available(key)) {
      out = opt->read_binary_asset(key);
      return true;
    }
  }
  const auto path = disk_dir / std::string(file);
  if (!std::filesystem::is_regular_file(path)) {
    return false;
  }
  return read_file(path, out);
}

/// Loads the single-model form, which may resolve to a sibling ``.ort``.
bool load_single(const MoonshineG2POptions* opt, std::string_view bundle_key,
                 std::string_view file, const std::filesystem::path& disk_dir,
                 std::vector<std::uint8_t>& out) {
  if (opt != nullptr && !bundle_key.empty()) {
    const std::string key = g2p_bundle_file_key(bundle_key, file);
    if (opt->asset_is_available(key)) {
      out = opt->read_binary_asset(key);
      return true;
    }
  }
  const auto path = ort_model_path(disk_dir, file);
  if (!std::filesystem::is_regular_file(path)) {
    return false;
  }
  return read_file(path, out);
}

}  // namespace

std::string g2p_split_model_file(std::string_view model_file) {
  return stem_of(model_file) + ".model.ort";
}

std::string g2p_split_weights_file(std::string_view model_file) {
  return stem_of(model_file) + ".weights.ort";
}

G2pTransformerModel load_g2p_transformer_model(
    Ort::Env& env, const MoonshineG2POptions* opt, std::string_view bundle_key,
    std::string_view model_file, const std::filesystem::path& model_dir,
    const std::vector<std::string>& ort_providers,
    const std::string& coreml_cache_dir, std::string_view owner) {
  G2pTransformerModel loaded;

  const std::string split_model = g2p_split_model_file(model_file);
  const std::string split_weights = g2p_split_weights_file(model_file);
  std::vector<std::uint8_t> graph_bytes;
  std::vector<std::uint8_t> weight_bytes;
  if (load_exact(opt, bundle_key, split_model, model_dir, graph_bytes) &&
      !graph_bytes.empty() &&
      load_exact(opt, bundle_key, split_weights, model_dir, weight_bytes) &&
      !weight_bytes.empty()) {
    // The weights session holds the int8 data and is released as soon as it has
    // produced the float32 tensors, so only the results stay resident.
    loaded.split_weights = run_split_weights_model(
        env, weight_bytes.data(), weight_bytes.size(),
        make_g2p_ort_session_options(ort_providers, coreml_cache_dir));
    loaded.model_bytes = std::move(graph_bytes);
    loaded.session = std::make_unique<Ort::Session>(
        env, loaded.model_bytes.data(), loaded.model_bytes.size(),
        make_g2p_ort_session_options(ort_providers, coreml_cache_dir));
    return loaded;
  }

  std::vector<std::uint8_t> bytes;
  if (load_single(opt, bundle_key, model_file, model_dir, bytes) &&
      !bytes.empty()) {
    loaded.model_bytes = std::move(bytes);
    loaded.session = std::make_unique<Ort::Session>(
        env, loaded.model_bytes.data(), loaded.model_bytes.size(),
        make_g2p_ort_session_options(ort_providers, coreml_cache_dir));
    return loaded;
  }

  throw std::runtime_error(std::string(owner) + ": missing " +
                           (model_dir / std::string(model_file)).string() +
                           " (no split ORT pair and no .ort found)");
}

}  // namespace moonshine_tts
