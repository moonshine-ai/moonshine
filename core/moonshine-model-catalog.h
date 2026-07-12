#ifndef MOONSHINE_MODEL_CATALOG_H
#define MOONSHINE_MODEL_CATALOG_H

// Native catalog of downloadable model assets (speech-to-text transcription,
// the optional alphanumeric spelling model, and intent-recognition embedding
// models). This is the C++ port of the tables that previously lived only in
// python/src/moonshine_voice/download.py, promoted here so every language
// binding resolves the exact same download manifest from a single source of
// truth. The TTS / G2P dependency catalog lives separately under
// core/moonshine-tts/ and is surfaced by moonshine_get_tts_dependencies /
// moonshine_get_g2p_dependencies; this file covers the remaining model types.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace moonshine {

// One set of files that share a single base URL. A model's full download list
// is one or more of these groups: STT is a single group, plus an optional
// second group for the alphanumeric spelling model (which lives under a
// different CDN path). Callers download `base_url + "/" + file` for each file.
struct ModelDependencyGroup {
  std::string base_url;
  std::vector<std::string> files;
};

struct ModelDependencies {
  std::vector<ModelDependencyGroup> groups;
};

// Returns the download manifest for a speech-to-text model.
//
// `language` is a language code (e.g. "en") or English name (e.g. "English"),
// matching the Python catalog's lookup rules. `model_arch`, when present, is
// one of the MOONSHINE_MODEL_ARCH_* constants; when absent, the first
// (default) model registered for the language is used. When `include_spelling`
// is true and a spelling model is published for the language, a second group
// carrying the spelling files is appended.
//
// Returns std::nullopt if the language (or the language+arch combination) is
// unknown.
std::optional<ModelDependencies> stt_model_dependencies(
    const std::string& language, std::optional<int32_t> model_arch,
    bool include_spelling);

// Returns the download manifest for an intent-recognition embedding model.
//
// `model_name` is an embedding model id (e.g. "embeddinggemma-300m").
// `variant` is one of the published variants ("q4", "q8", "fp16", "fp32",
// "q4f16"); an empty string selects the model's default variant. Returns
// std::nullopt if the model or variant is unknown.
std::optional<ModelDependencies> intent_model_dependencies(
    const std::string& model_name, const std::string& variant);

// Language codes with at least one registered STT model, in catalog order.
std::vector<std::string> stt_supported_languages();

// Registered embedding model ids, in catalog order.
std::vector<std::string> intent_supported_models();

// Published variants for an embedding model (empty if the model is unknown).
std::vector<std::string> intent_supported_variants(const std::string& model_name);

}  // namespace moonshine

#endif  // MOONSHINE_MODEL_CATALOG_H
