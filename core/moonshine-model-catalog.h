#ifndef MOONSHINE_MODEL_CATALOG_H
#define MOONSHINE_MODEL_CATALOG_H

// Native catalog of downloadable model assets (speech-to-text transcription,
// the optional alphanumeric spelling model, and text embedding models). This
// is the C++ port of the tables that previously lived only in
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

// A single downloadable model file, with the metadata a client needs to fetch
// and verify it. `url` is the fully-qualified download URL (`base_url + "/" +
// name`). `size` is the expected size in bytes, or -1 when unknown. `checksum`
// is a base64-encoded digest of type `checksum_type` (e.g. "crc32c"), or empty
// strings when unknown. Size/checksum come from the generated registry in
// core/moonshine-model-file-metadata.* (see
// scripts/generate-model-file-metadata.py); they are absent (-1 / "") until the
// registry is regenerated for a newly published file.
struct ModelFile {
  std::string name;
  std::string url;
  int64_t size = -1;
  std::string checksum;
  std::string checksum_type;
};

// One set of files that share a single base URL. A model's full download list
// is one or more of these groups: STT is a single group, plus an optional
// second group for the alphanumeric spelling model (which lives under a
// different CDN path). Each file already carries its full `url`.
//
// ``role`` is empty for ordinary groups. TTS dependency manifests set
// ``role`` to ``"clone_asr"`` for the ZipVoice-owned speech-to-text group
// (local ``name``s are prefixed ``clone_asr/``; ``url``s stay on the STT CDN).
struct ModelDependencyGroup {
  std::string base_url;
  std::vector<ModelFile> files;
  std::string role;
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
// carrying the spelling files is appended. When `include_word_timestamps` is
// true, the optional `*_with_attention.ort` decoder (used only by the
// `word_timestamps` transcriber option, and roughly doubling the download) is
// included for languages that publish it; leave it false to skip that file.
//
// Returns std::nullopt if the language (or the language+arch combination) is
// unknown.
std::optional<ModelDependencies> stt_model_dependencies(
    const std::string& language, std::optional<int32_t> model_arch,
    bool include_spelling, bool include_word_timestamps);

// Returns the download manifest for a text embedding model.
//
// `model_name` is an embedding model id (e.g. "embeddinggemma-300m").
// `variant` is one of the published variants ("q4", "q8", "fp16", "fp32",
// "q4f16"); an empty string selects the model's default variant. Returns
// std::nullopt if the model or variant is unknown.
std::optional<ModelDependencies> embedding_model_dependencies(
    const std::string& model_name, const std::string& variant);

// Returns the download manifest for the speaker diarization models, which back
// the transcriber's `identify_speakers` option. There is only one set, so this
// takes no arguments and always succeeds.
//
// These two files used to be compiled into the library as C arrays. They are
// 8.2 MB together, which every caller paid for whether or not they diarized, so
// they became a download like every other model; see
// docs/diarization-models.md.
ModelDependencies diarization_model_dependencies();

// Canonical filenames the diarization manifest resolves to, in the order the
// loader expects them: segmentation first, then embedding. Exposed so that
// callers assembling an in-memory load (`segmentation.ort` / `embedding.ort`
// keys on moonshine_load_transcriber_from_memory_files) do not have to
// hard-code the names.
std::vector<std::string> diarization_component_files();

// Language codes with at least one registered STT model, in catalog order.
std::vector<std::string> stt_supported_languages();

// Registered embedding model ids, in catalog order.
std::vector<std::string> embedding_supported_models();

// Published variants for an embedding model (empty if the model is unknown).
std::vector<std::string> embedding_supported_variants(
    const std::string& model_name);

// --- Full catalog listings ------------------------------------------------
// These expose the catalog tables themselves (languages, friendly names,
// architectures, variants) so language bindings can present model pickers and
// resolve defaults without maintaining their own duplicate copies.

struct SttCatalogModel {
  int32_t model_arch;  // one of the MOONSHINE_MODEL_ARCH_* constants
  std::string download_url;
  bool is_default;  // true for the language's default (first) model
};

struct SttCatalogLanguage {
  std::string code;          // e.g. "en"
  std::string english_name;  // e.g. "English"
  std::vector<SttCatalogModel> models;
};

// All STT languages and their registered models, in catalog order.
std::vector<SttCatalogLanguage> stt_catalog_listing();

struct EmbeddingCatalogModel {
  std::string name;          // e.g. "embeddinggemma-300m"
  std::string english_name;  // e.g. "Embedding Gemma 300M"
  std::string download_url;
  std::vector<std::string> variants;
  std::string default_variant;
};

// All embedding models, in catalog order.
std::vector<EmbeddingCatalogModel> embedding_catalog_listing();

}  // namespace moonshine

#endif  // MOONSHINE_MODEL_CATALOG_H
