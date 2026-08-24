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
constexpr const char* kStreamingQuantizedDir = "/quantized_26_08_21";

// The same scheme for Japanese, which was quantized from different checkpoints
// on a different day. Per-language rather than shared, because a language's
// weights are re-quantized when that language gets a better checkpoint, and one
// shared constant would force every language to move at once.
constexpr const char* kJapaneseStreamingQuantizedDir = "/quantized_26_08_23";

// The six additional languages published as streaming, each with its own
// constant for the reason above. Tagalog in particular is a snapshot of a run
// that was still training, so it is the one most likely to move on its own.
constexpr const char* kGermanStreamingQuantizedDir = "/quantized_26_08_24";
constexpr const char* kSpanishStreamingQuantizedDir = "/quantized_26_08_24";
constexpr const char* kVietnameseStreamingQuantizedDir = "/quantized_26_08_24";
constexpr const char* kArabicStreamingQuantizedDir = "/quantized_26_08_24";
constexpr const char* kChineseStreamingQuantizedDir = "/quantized_26_08_24";
constexpr const char* kTagalogStreamingQuantizedDir = "/quantized_26_08_24";

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

// Port of MODEL_INFO from
// language-bindings/python/src/moonshine_voice/download.py. The first model
// listed for a language is its default.
const std::vector<SttLanguageEntry>& stt_catalog() {
  static const std::vector<SttLanguageEntry> catalog = {
      // Streaming first, so it is the default, mirroring English and Japanese.
      // Tiny streaming measures 15.5 WER across Common Voice and FLEURS on a
      // seeded 400-clip sample at batch 1, as deployed. The older base entry
      // stays listed for callers that ask for that architecture by name; it was
      // never scored on this panel, so this is not a claim that streaming beats
      // it, only that streaming is what we now measure and ship.
      {"ar",
       "Arabic",
       {
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-ar" +
                kArabicStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-ar/quantized/base-ar"},
       }},
      // Small and tiny streaming measure 4.9 and 6.2 WER across FLEURS and MLS
      // on a seeded 400-clip sample at batch 1. As for Arabic, the base entry
      // is
      // kept for callers that name that architecture and was never scored here.
      {"es",
       "Spanish",
       {
           {MOONSHINE_MODEL_ARCH_SMALL_STREAMING,
            std::string(kCdnModelBase) + "/small-streaming-es" +
                kSpanishStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-es" +
                kSpanishStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-es/quantized/base-es"},
       }},
      // German is a new language for the catalog: there is no older base model
      // to fall back to. Small and tiny streaming measure 7.5 and 12.0 WER
      // across FLEURS and MLS on a seeded 400-clip sample at batch 1.
      {"de",
       "German",
       {
           {MOONSHINE_MODEL_ARCH_SMALL_STREAMING,
            std::string(kCdnModelBase) + "/small-streaming-de" +
                kGermanStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-de" +
                kGermanStreamingQuantizedDir},
       }},
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
      // Streaming first, so it is the default, mirroring English. Small and
      // tiny streaming measure 17.2 and 19.7 no-space CER across FLEURS and
      // ReazonSpeech (batch 1, as deployed). The older non-streaming base and
      // tiny entries stay listed for callers that ask for those architectures
      // by name; they were never scored on this panel, so this is not a claim
      // that streaming beats them, only that streaming is what we now measure
      // and ship.
      {"ja",
       "Japanese",
       {
           {MOONSHINE_MODEL_ARCH_SMALL_STREAMING,
            std::string(kCdnModelBase) + "/small-streaming-ja" +
                kJapaneseStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-ja" +
                kJapaneseStreamingQuantizedDir},
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
      // Tiny streaming measures 9.4 WER across FLEURS and LSVSC on a seeded
      // 400-clip sample at batch 1. The base entry is kept for callers that
      // name
      // that architecture and was never scored here.
      {"vi",
       "Vietnamese",
       {
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-vi" +
                kVietnameseStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-vi/quantized/base-vi"},
       }},
      {"uk",
       "Ukrainian",
       {{MOONSHINE_MODEL_ARCH_BASE,
         std::string(kCdnModelBase) + "/base-uk/quantized/base-uk"}}},
      // Mandarin is scored with no-space CER, never WER, for the same reason as
      // Japanese: the language is written without spaces, so word-level
      // alignment measures the tokenizer rather than the model. Tiny streaming
      // measures 16.1 no-space CER across FLEURS and WenetSpeech on a seeded
      // 400-clip sample at batch 1. The base entry is kept for callers that
      // name
      // that architecture and was never scored here.
      {"zh",
       "Chinese",
       {
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-zh" +
                kChineseStreamingQuantizedDir},
           {MOONSHINE_MODEL_ARCH_BASE,
            std::string(kCdnModelBase) + "/base-zh/quantized/base-zh"},
       }},
      // Tagalog is a new language for the catalog, and its entry is a snapshot
      // of a Stage A run that had not finished training when it was taken:
      // 14.9 WER on FLEURS at batch 1, the only Tagalog panel we hold, so this
      // number rests on one read-speech set rather than a macro over two.
      {"tl",
       "Tagalog",
       {
           {MOONSHINE_MODEL_ARCH_TINY_STREAMING,
            std::string(kCdnModelBase) + "/tiny-streaming-tl" +
                kTagalogStreamingQuantizedDir},
       }},
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
       {"q4", "q8"},
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
        "adapter.ort",           "cross_kv.ort",       "decoder_kv.ort",
        "encoder.ort",           "frontend.model.ort", "frontend.weights.ort",
        "streaming_config.json", "tokenizer.bin",
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
  if (variant == "q8") {
    stem = "model_quantized";
  } else if (variant == "q4") {
    stem = "model_q4";
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
  deps.groups.push_back(make_group(std::string(kCdnModelBase) + kDiarizationDir,
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

std::string embedding_variant_unsupported_message(const std::string& variant) {
  if (variant == "fp32" || variant == "fp16" || variant == "q4f16") {
    return "The \"" + variant +
           "\" embedding model variant is no longer supported. "
           "Use \"q4\" (the default) or \"q8\".";
  }
  return {};
}

namespace {

std::string model_arch_label(int32_t model_arch) {
  const char* name = nullptr;
  switch (model_arch) {
    case MOONSHINE_MODEL_ARCH_TINY:
      name = "TINY";
      break;
    case MOONSHINE_MODEL_ARCH_BASE:
      name = "BASE";
      break;
    case MOONSHINE_MODEL_ARCH_TINY_STREAMING:
      name = "TINY_STREAMING";
      break;
    case MOONSHINE_MODEL_ARCH_BASE_STREAMING:
      name = "BASE_STREAMING";
      break;
    case MOONSHINE_MODEL_ARCH_SMALL_STREAMING:
      name = "SMALL_STREAMING";
      break;
    case MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING:
      name = "MEDIUM_STREAMING";
      break;
    default:
      break;
  }
  std::string label = std::to_string(model_arch);
  if (name != nullptr) {
    label += " (";
    label += name;
    label += ")";
  }
  return label;
}

}  // namespace

std::string stt_missing_dependencies_message(
    const std::string& language, std::optional<int32_t> model_arch) {
  const SttLanguageEntry* lang = find_stt_language(language);
  if (lang == nullptr || lang->models.empty()) {
    return "unknown language \"" + language + "\"";
  }
  if (!model_arch.has_value()) {
    return {};
  }
  for (const SttModelEntry& candidate : lang->models) {
    if (candidate.model_arch == *model_arch) {
      return {};
    }
  }
  std::string supported;
  const char* sep = "";
  for (const SttModelEntry& model : lang->models) {
    supported += sep;
    supported += model_arch_label(model.model_arch);
    sep = ", ";
  }
  return "language \"" + language + "\" has no model_arch " +
         model_arch_label(*model_arch) +
         "; supported architectures: " + supported;
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
