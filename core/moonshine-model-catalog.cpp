#include "moonshine-model-catalog.h"

#include <algorithm>
#include <cctype>

#include "moonshine-c-api.h"
#include "moonshine-model-file-metadata.h"

namespace moonshine {

namespace {

// Builds a dependency group from a base URL and a list of canonical filenames,
// joining in the size/checksum metadata for each file's full download URL.
ModelDependencyGroup make_group(const std::string& base_url,
                                const std::vector<std::string>& names) {
  ModelDependencyGroup group;
  group.base_url = base_url;
  group.files.reserve(names.size());
  for (const std::string& name : names) {
    ModelFile file;
    file.name = name;
    file.url = base_url + "/" + name;
    const ModelFileMetadata meta = find_model_file_metadata(file.url);
    file.size = meta.size;
    file.checksum = meta.checksum;
    file.checksum_type = meta.checksum_type;
    group.files.push_back(std::move(file));
  }
  return group;
}

// Base URL for the model CDN. STT and embedding models live under
// "<kCdnModelBase>/<model>/...", matching the Python catalog's download_url
// values.
constexpr const char* kCdnModelBase = "https://download.moonshine.ai/model";

// Directory holding the current English streaming weights. Re-quantized
// releases go in a new dated directory rather than overwriting the old one, so
// that older library versions keep resolving the weights they were tested
// against and a rollback is a one-line change. Clients key their download cache
// off this URL, so a new directory also guarantees a clean re-fetch instead of
// silently reusing stale files.
constexpr const char* kStreamingQuantizedDir = "/quantized_26_07_30";

struct SttModelEntry {
  int32_t model_arch;
  std::string download_url;
};

struct SttLanguageEntry {
  std::string code;
  std::string english_name;
  std::vector<SttModelEntry> models;
};

struct SpellingModelEntry {
  std::string download_url;
  std::vector<std::string> files;
};

struct EmbeddingModelEntry {
  std::string name;
  std::string english_name;
  std::string download_url;
  std::vector<std::string> variants;
  std::string default_variant;
};

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool is_streaming_arch(int32_t model_arch) {
  return model_arch == MOONSHINE_MODEL_ARCH_TINY_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_BASE_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_SMALL_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
}

// Port of MODEL_INFO from python/src/moonshine_voice/download.py. The first
// model listed for a language is its default.
const std::vector<SttLanguageEntry>& stt_catalog() {
  static const std::vector<SttLanguageEntry> catalog = {
      {"ar",
       "Arabic",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-ar/quantized/base-ar"}}},
      {"es",
       "Spanish",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-es/quantized/base-es"}}},
      {"en",
       "English",
       {
           {MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, std::string(kCdnModelBase) +
                                                       "/medium-streaming-en" +
                                                       kStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_SMALL_STREAMING, std::string(kCdnModelBase) +
                                                      "/small-streaming-en" +
                                                      kStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-en/quantized/base-en"},
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING, std::string(kCdnModelBase) +
                                                     "/tiny-streaming-en" +
                                                     kStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_TINY,
            std::string(kCdnModelBase) + "/tiny-en/quantized/tiny-en"},
       }},
      {"ja",
       "Japanese",
       {
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-ja/quantized/base-ja"},
           {MOONSHINE_MODEL_ARCH_TINY,
            std::string(kCdnModelBase) + "/tiny-ja/quantized/tiny-ja"},
       }},
      // Korean's default model is served from the tiny-ko path with the TINY
      // architecture (mirrors the Python catalog).
      {"ko",
       "Korean",
       {{MOONSHINE_MODEL_ARCH_TINY,
         std::string(kCdnModelBase) + "/tiny-ko/quantized/tiny-ko"}}},
      {"vi",
       "Vietnamese",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-vi/quantized/base-vi"}}},
      {"uk",
       "Ukrainian",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-uk/quantized/base-uk"}}},
      {"zh",
       "Chinese",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-zh/quantized/base-zh"}}},
  };
  return catalog;
}

// Port of SPELLING_MODEL_INFO. Only English ships a spelling model today.
const std::vector<std::pair<std::string, SpellingModelEntry>>&
spelling_catalog() {
  static const std::vector<std::pair<std::string, SpellingModelEntry>> catalog =
      {
          {"en",
           {std::string(kCdnModelBase) + "/spelling-en",
            {"spelling_cnn.ort", "spelling_cnn_meta.json"}}},
      };
  return catalog;
}

// Port of EMBEDDING_MODEL_INFO.
const std::vector<EmbeddingModelEntry>& embedding_catalog() {
  static const std::vector<EmbeddingModelEntry> catalog = {
      {"embeddinggemma-300m",
       "Embedding Gemma 300M",
       std::string(kCdnModelBase) + "/embeddinggemma-300m",
       {"q4", "q8", "fp16", "fp32", "q4f16"},
       "q4"},
  };
  return catalog;
}

// The pyannote community-1 segmentation and speaker-embedding models, as
// converted to ORT format. Both are pinned in one directory because the
// clustering parameters still compiled into the library (the PLDA and x-vector
// arrays in cpp-annote's community1_cpp_annote_embedded.cpp) were fitted
// against this exact pair; a new pair means a new directory and a matching
// library release, not an overwrite.
constexpr const char* kDiarizationDir = "/diarization-community1";

const SttLanguageEntry* find_stt_language(const std::string& language) {
  const std::string wanted = to_lower(language);
  for (const SttLanguageEntry& entry : stt_catalog()) {
    if (entry.code == wanted) {
      return &entry;
    }
  }
  for (const SttLanguageEntry& entry : stt_catalog()) {
    if (to_lower(entry.english_name) == wanted) {
      return &entry;
    }
  }
  return nullptr;
}

std::vector<std::string> stt_component_files(const std::string& language_code,
                                             int32_t model_arch,
                                             bool include_word_timestamps) {
  // The `*_with_attention.ort` decoders are only used to produce word-level
  // timestamps (the `word_timestamps` transcriber option). They roughly double
  // the download, so they are only listed when the caller opts in - matching
  // the option they would pass when constructing the transcriber. Only English
  // publishes an attention decoder today.
  const bool is_english = (language_code == "en");
  if (is_streaming_arch(model_arch)) {
    std::vector<std::string> files = {
        "adapter.ort",   "cross_kv.ort", "decoder_kv.ort",
        "encoder.ort",   "frontend.ort", "streaming_config.json",
        "tokenizer.bin",
    };
    if (is_english && include_word_timestamps) {
      files.push_back("decoder_kv_with_attention.ort");
    }
    return files;
  }
  std::vector<std::string> files = {
      "encoder_model.ort", "decoder_model_merged.ort", "tokenizer.bin"};
  if (is_english && include_word_timestamps) {
    files.push_back("decoder_with_attention.ort");
  }
  return files;
}

const SpellingModelEntry* find_spelling_model(
    const std::string& language_code) {
  for (const auto& [code, entry] : spelling_catalog()) {
    if (code == language_code) {
      return &entry;
    }
  }
  return nullptr;
}

const EmbeddingModelEntry* find_embedding_model(const std::string& model_name) {
  for (const EmbeddingModelEntry& entry : embedding_catalog()) {
    if (entry.name == model_name) {
      return &entry;
    }
  }
  return nullptr;
}

// The C++ embedding loader (gemma-embedding-model.cpp) maps each variant to a
// specific model filename. Note that "q8" resolves to model_quantized, not
// model_q8 (the latter is not published) - this fixes a divergence in the old
// Python table. Each variant now ships as a single all-in-one ``.ort`` file
// (weights embedded inline, no external-data sidecar), so the manifest lists
// only ``model_<variant>.ort`` and ``tokenizer.bin`` - matching how the
// speech-to-text models ship and letting the in-memory loader take a single
// buffer per file. See scripts/export-embedding-model-ort.py for how the
// ``.ort`` files are produced from the published ``.onnx`` + ``.onnx_data``.
std::vector<std::string> embedding_component_files(const std::string& variant) {
  std::string stem;
  if (variant == "fp32") {
    stem = "model";
  } else if (variant == "fp16") {
    stem = "model_fp16";
  } else if (variant == "q8") {
    stem = "model_quantized";
  } else if (variant == "q4") {
    stem = "model_q4";
  } else if (variant == "q4f16") {
    stem = "model_q4f16";
  } else {
    return {};
  }
  return {stem + ".ort", "tokenizer.bin"};
}

}  // namespace

std::optional<ModelDependencies> stt_model_dependencies(
    const std::string& language, std::optional<int32_t> model_arch,
    bool include_spelling, bool include_word_timestamps) {
  const SttLanguageEntry* lang = find_stt_language(language);
  if (lang == nullptr || lang->models.empty()) {
    return std::nullopt;
  }

  const SttModelEntry* model = nullptr;
  if (model_arch.has_value()) {
    for (const SttModelEntry& candidate : lang->models) {
      if (candidate.model_arch == *model_arch) {
        model = &candidate;
        break;
      }
    }
    if (model == nullptr) {
      return std::nullopt;
    }
  } else {
    model = &lang->models.front();
  }

  ModelDependencies deps;
  deps.groups.push_back(make_group(
      model->download_url, stt_component_files(lang->code, model->model_arch,
                                               include_word_timestamps)));

  if (include_spelling) {
    const SpellingModelEntry* spelling = find_spelling_model(lang->code);
    if (spelling != nullptr) {
      deps.groups.push_back(
          make_group(spelling->download_url, spelling->files));
    }
  }
  return deps;
}

std::optional<ModelDependencies> embedding_model_dependencies(
    const std::string& model_name, const std::string& variant) {
  const EmbeddingModelEntry* model = find_embedding_model(model_name);
  if (model == nullptr) {
    return std::nullopt;
  }
  const std::string resolved_variant =
      variant.empty() ? model->default_variant : variant;
  if (std::find(model->variants.begin(), model->variants.end(),
                resolved_variant) == model->variants.end()) {
    return std::nullopt;
  }
  const std::vector<std::string> files =
      embedding_component_files(resolved_variant);
  if (files.empty()) {
    return std::nullopt;
  }
  ModelDependencies deps;
  deps.groups.push_back(make_group(model->download_url, files));
  return deps;
}

std::vector<std::string> diarization_component_files() {
  return {"segmentation.ort", "embedding.ort"};
}

ModelDependencies diarization_model_dependencies() {
  ModelDependencies deps;
  deps.groups.push_back(make_group(std::string(kCdnModelBase) +
                                       kDiarizationDir,
                                   diarization_component_files()));
  return deps;
}

std::vector<std::string> stt_supported_languages() {
  std::vector<std::string> codes;
  for (const SttLanguageEntry& entry : stt_catalog()) {
    codes.push_back(entry.code);
  }
  return codes;
}

std::vector<std::string> embedding_supported_models() {
  std::vector<std::string> names;
  for (const EmbeddingModelEntry& entry : embedding_catalog()) {
    names.push_back(entry.name);
  }
  return names;
}

std::vector<std::string> embedding_supported_variants(
    const std::string& model_name) {
  const EmbeddingModelEntry* model = find_embedding_model(model_name);
  if (model == nullptr) {
    return {};
  }
  return model->variants;
}

std::vector<SttCatalogLanguage> stt_catalog_listing() {
  std::vector<SttCatalogLanguage> out;
  for (const SttLanguageEntry& lang : stt_catalog()) {
    SttCatalogLanguage entry;
    entry.code = lang.code;
    entry.english_name = lang.english_name;
    for (size_t i = 0; i < lang.models.size(); ++i) {
      entry.models.push_back({lang.models[i].model_arch,
                              lang.models[i].download_url,
                              /*is_default=*/i == 0});
    }
    out.push_back(std::move(entry));
  }
  return out;
}

std::vector<EmbeddingCatalogModel> embedding_catalog_listing() {
  std::vector<EmbeddingCatalogModel> out;
  for (const EmbeddingModelEntry& model : embedding_catalog()) {
    out.push_back({model.name, model.english_name, model.download_url,
                   model.variants, model.default_variant});
  }
  return out;
}

}  // namespace moonshine
