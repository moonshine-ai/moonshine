#include "piper-tts.h"

#include "piper-voice-catalog.h"
#include "g2p-path.h"
#include "ort-onnx-external-data.h"
#include "ipa-postprocess.h"
#include "moonshine-g2p.h"
#include "utf8-utils.h"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.h>

extern "C" {
#include <utf8proc.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace moonshine_tts {

namespace {

std::string normalize_lang_key(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == ' ') {
      c = '_';
    } else if (c == '-') {
      c = '_';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

bool py_isspace_utf8_ch(std::string_view ch) {
  if (ch.empty()) {
    return false;
  }
  std::string tmp(ch);
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(tmp, 0, cp, adv) || adv != tmp.size()) {
    return false;
  }
  if (cp < 128) {
    return std::isspace(static_cast<unsigned char>(cp)) != 0;
  }
  const auto cat = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
  return cat == UTF8PROC_CATEGORY_ZS || cat == UTF8PROC_CATEGORY_ZL || cat == UTF8PROC_CATEGORY_ZP;
}

struct PiperLangRow {
  const char* g2p_dialect;
  const char* data_subdir;
  const char* default_onnx;
};

const PiperLangRow* lookup_piper_lang_row(std::string_view k) {
  // Keys are mostly ``ll`` or ``ll_rr`` (underscore). Canonical API tags like ``hi-in`` / ``de-de``
  // normalize to ``hi_in`` / ``de_de``; collapse those to the primary map key when equivalent.
  static const std::unordered_map<std::string, PiperLangRow> m{
      {"en_us", {"en_us", "en_us", "en_US-lessac-medium.onnx"}},
      {"en-us", {"en_us", "en_us", "en_US-lessac-medium.onnx"}},
      {"en", {"en_us", "en_us", "en_US-lessac-medium.onnx"}},
      {"en_gb", {"en_us", "en_gb", "en_GB-cori-medium.onnx"}},
      {"en-gb", {"en_us", "en_gb", "en_GB-cori-medium.onnx"}},
      {"es", {"es-MX", "es_mx", "es_MX-ald-medium.onnx"}},
      {"es_mx", {"es-MX", "es_mx", "es_MX-ald-medium.onnx"}},
      {"es_es", {"es-ES", "es_es", "es_ES-davefx-medium.onnx"}},
      {"es_ar", {"es-AR", "es_ar", "es_AR-daniela-high.onnx"}},
      {"fr", {"fr-FR", "fr", "fr_FR-siwis-medium.onnx"}},
      {"hi", {"hi", "hi", "hi_IN-pratham-medium.onnx"}},
      {"it", {"it-IT", "it", "it_IT-paola-medium.onnx"}},
      {"pt_br", {"pt_br", "pt_br", "pt_BR-cadu-medium.onnx"}},
      {"pt-br", {"pt_br", "pt_br", "pt_BR-cadu-medium.onnx"}},
      {"pt", {"pt_br", "pt_br", "pt_BR-cadu-medium.onnx"}},
      {"pt_pt", {"pt_pt", "pt_pt", "pt_PT-tugão-medium.onnx"}},
      {"pt-pt", {"pt_pt", "pt_pt", "pt_PT-tugão-medium.onnx"}},
      {"zh", {"zh", "zh_hans", "zh_CN-huayan-medium.onnx"}},
      {"zh_hans", {"zh", "zh_hans", "zh_CN-huayan-medium.onnx"}},
      {"zh_cn", {"zh", "zh_hans", "zh_CN-huayan-medium.onnx"}},
      {"zt", {"zh", "zh_hans", "zh_CN-huayan-medium.onnx"}},
      {"ar_msa", {"ar", "ar_msa", "ar_JO-kareem-medium.onnx"}},
      {"ar", {"ar", "ar_msa", "ar_JO-kareem-medium.onnx"}},
      {"de", {"de-DE", "de", "de_DE-thorsten-medium.onnx"}},
      {"nl", {"nl-NL", "nl", "nl_NL-mls-medium.onnx"}},
      {"ru", {"ru-RU", "ru", "ru_RU-denis-medium.onnx"}},
      {"tr", {"tr-TR", "tr", "tr_TR-dfki-medium.onnx"}},
      {"uk", {"uk-UA", "uk", "uk_UA-ukrainian_tts-medium.onnx"}},
      {"vi", {"vi-VN", "vi", "vi_VN-vais1000-medium.onnx"}},
      {"ko", {"ko", "ko", "ko_KR-melotts-medium.onnx"}},
      {"ko_kr", {"ko", "ko", "ko_KR-melotts-medium.onnx"}},
      {"korean", {"ko", "ko", "ko_KR-melotts-medium.onnx"}},
  };
  const std::string key = normalize_lang_key(k);
  if (const auto it = m.find(key); it != m.end()) {
    return &it->second;
  }
  static const std::unordered_map<std::string, std::string> kBcp47UnderscoreToPrimary{
      {"de_de", "de"}, {"fr_fr", "fr"}, {"hi_in", "hi"}, {"it_it", "it"}, {"nl_nl", "nl"},
      {"ru_ru", "ru"}, {"vi_vn", "vi"}, {"uk_ua", "uk"}, {"tr_tr", "tr"},
  };
  if (const auto c = kBcp47UnderscoreToPrimary.find(key); c != kBcp47UnderscoreToPrimary.end()) {
    if (const auto it2 = m.find(c->second); it2 != m.end()) {
      return &it2->second;
    }
  }
  return nullptr;
}

std::string piper_ipa_norm_lang_key(const std::string& lk, std::string_view data_subdir) {
  if (data_subdir == "es_es") {
    return "es_es";
  }
  if (data_subdir == "es_mx") {
    return "es_mx";
  }
  if (data_subdir == "es_ar") {
    return "es_ar";
  }
  return normalize_lang_key(lk);
}

void resolve_piper_lang(const std::string& lk, const MoonshineG2POptions& opt, std::string& g2p_dialect,
                        std::string& data_subdir, std::string& default_onnx) {
  const std::string k = normalize_lang_key(lk);
  if (k == "ja" || k == "jp" || k == "ja_jp") {
    throw std::runtime_error(
        "PiperTTS: Japanese is not available as a Piper ONNX bundle here; use MoonshineTTS (Kokoro).");
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  if (!norm.empty() && dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents)) {
    g2p_dialect = norm;
    if (norm.rfind("es-ES", 0) == 0) {
      data_subdir = "es_es";
      default_onnx = "es_ES-davefx-medium.onnx";
    } else if (norm == "es-AR") {
      data_subdir = "es_ar";
      default_onnx = "es_AR-daniela-high.onnx";
    } else {
      data_subdir = "es_mx";
      default_onnx = "es_MX-ald-medium.onnx";
    }
    return;
  }
  const PiperLangRow* row = lookup_piper_lang_row(k);
  if (row == nullptr) {
    throw std::runtime_error("PiperTTS: unsupported --lang key \"" + lk + "\"");
  }
  g2p_dialect = row->g2p_dialect;
  data_subdir = row->data_subdir;
  default_onnx = row->default_onnx;
}

std::filesystem::path pick_onnx_path(const std::filesystem::path& voices_dir, std::string_view requested,
                                     std::string_view default_basename) {
  if (!std::filesystem::is_directory(voices_dir)) {
    throw std::runtime_error("PiperTTS: voices_dir is not a directory: " + voices_dir.string());
  }
  if (!requested.empty()) {
    std::string name(trim_ascii_ws_copy(requested));
    if (name.size() < 5 || name.substr(name.size() - 5) != ".onnx") {
      name += ".onnx";
    }
    const auto cand = voices_dir / name;
    if (std::filesystem::is_regular_file(cand)) {
      return cand;
    }
  }
  const auto d = voices_dir / std::string(default_basename);
  if (std::filesystem::is_regular_file(d)) {
    return d;
  }
  std::vector<std::filesystem::path> models;
  for (const auto& ent : std::filesystem::directory_iterator(voices_dir)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    const auto& p = ent.path();
    if (p.extension() == ".onnx") {
      models.push_back(p);
    }
  }
  std::sort(models.begin(), models.end());
  if (models.empty()) {
    throw std::runtime_error("PiperTTS: no *.onnx in " + voices_dir.string());
  }
  return models[0];
}

/// Piper pairs ``foo.onnx`` with ``foo.onnx.json``. If ``json_dir`` is empty, that file sits beside ``onnx_path``.
std::filesystem::path piper_model_json_path_for_onnx(const std::filesystem::path& onnx_path,
                                                    const std::filesystem::path& json_dir) {
  if (!json_dir.empty()) {
    return json_dir / (onnx_path.filename().string() + ".json");
  }
  std::filesystem::path p = onnx_path;
  p.replace_extension(".onnx.json");
  return p;
}

void append_phoneme_ids(const std::unordered_map<std::string, std::vector<int64_t>>& id_map,
                        const std::string& key, std::vector<int64_t>& ids) {
  const auto it = id_map.find(key);
  if (it == id_map.end()) {
    return;
  }
  for (int64_t v : it->second) {
    ids.push_back(v);
  }
}

std::vector<int64_t> ipa_utf8_to_piper_ids(const std::string& ipa_nfc,
                                           const std::unordered_map<std::string, std::vector<int64_t>>& id_map) {
  std::string ipa = ipa_nfc;
  repair_ascii_c_combining_cedilla_to_ccedilla_utf8(ipa);
  std::vector<int64_t> ids;
  append_phoneme_ids(id_map, "^", ids);
  append_phoneme_ids(id_map, "_", ids);
  for (const std::string& ch : utf8_split_codepoints(ipa)) {
    if (py_isspace_utf8_ch(ch)) {
      append_phoneme_ids(id_map, " ", ids);
      continue;
    }
    if (id_map.count(ch) != 0) {
      append_phoneme_ids(id_map, ch, ids);
      append_phoneme_ids(id_map, "_", ids);
    }
  }
  append_phoneme_ids(id_map, "$", ids);
  return ids;
}

/// Match ``speak._piper_apply_synthesis_output_effects`` / ``PiperVoice.synthesize`` post-ORT step.
void piper_apply_synthesis_output_effects(std::vector<float>& audio, bool normalize_audio, float volume) {
  if (normalize_audio) {
    float max_val = 0.F;
    for (float x : audio) {
      max_val = std::max(max_val, std::fabs(x));
    }
    if (max_val < 1e-8F) {
      std::fill(audio.begin(), audio.end(), 0.F);
    } else {
      const float inv = 1.F / max_val;
      for (float& x : audio) {
        x *= inv;
      }
    }
  }
  if (volume != 1.F) {
    for (float& x : audio) {
      x *= volume;
    }
  }
  for (float& x : audio) {
    x = std::max(-1.F, std::min(1.F, x));
  }
}

std::vector<float> resample_linear(const std::vector<float>& x, int src_sr, int dst_sr) {
  if (src_sr == dst_sr || x.empty()) {
    return x;
  }
  const double duration = static_cast<double>(x.size() - 1) / static_cast<double>(src_sr);
  const size_t n_out = std::max<size_t>(2, static_cast<size_t>(std::llround(duration * dst_sr)) + 1);
  std::vector<float> y(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    const double t = (n_out <= 1) ? 0.0 : (static_cast<double>(i) * duration / static_cast<double>(n_out - 1));
    const double fidx = t * static_cast<double>(src_sr);
    const size_t i0 = static_cast<size_t>(std::floor(fidx));
    const size_t i1 = std::min(i0 + 1, x.size() - 1);
    const double frac = fidx - static_cast<double>(i0);
    y[i] = static_cast<float>(x[i0] * (1.0 - frac) + x[i1] * frac);
  }
  return y;
}

Ort::SessionOptions make_ort_options(const std::vector<std::string>& names) {
  (void)names;
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(0);
  opts.SetInterOpNumThreads(0);
  return opts;
}

void load_piper_onnx_json(const std::filesystem::path& json_path,
                          std::unordered_map<std::string, std::vector<int64_t>>& phoneme_id_map, int& native_sample_rate,
                          float& noise_scale, float& length_scale_default, float& noise_w, int& num_speakers) {
  std::ifstream jf(json_path);
  if (!jf) {
    throw std::runtime_error("PiperTTS: cannot open config " + json_path.string());
  }
  nlohmann::json j;
  jf >> j;
  if (!j.contains("phoneme_id_map") || !j["phoneme_id_map"].is_object()) {
    throw std::runtime_error("PiperTTS: phoneme_id_map missing in " + json_path.string());
  }
  phoneme_id_map.clear();
  for (auto it = j["phoneme_id_map"].begin(); it != j["phoneme_id_map"].end(); ++it) {
    std::vector<int64_t> seq;
    if (it.value().is_array()) {
      for (const auto& el : it.value()) {
        seq.push_back(el.get<int64_t>());
      }
    }
    phoneme_id_map[it.key()] = std::move(seq);
  }
  native_sample_rate = 22050;
  noise_scale = 0.667F;
  length_scale_default = 1.F;
  noise_w = 0.8F;
  num_speakers = 1;
  if (j.contains("audio") && j["audio"].is_object() && j["audio"].contains("sample_rate")) {
    native_sample_rate = j["audio"]["sample_rate"].get<int>();
  }
  if (j.contains("inference") && j["inference"].is_object()) {
    const auto& inf = j["inference"];
    if (inf.contains("noise_scale")) {
      noise_scale = inf["noise_scale"].get<float>();
    }
    if (inf.contains("length_scale")) {
      length_scale_default = inf["length_scale"].get<float>();
    }
    if (inf.contains("noise_w")) {
      noise_w = inf["noise_w"].get<float>();
    }
  }
  if (j.contains("num_speakers")) {
    num_speakers = j["num_speakers"].get<int>();
  }
}

void load_piper_onnx_json_bytes(const char* data, size_t size, std::string_view ctx,
                                std::unordered_map<std::string, std::vector<int64_t>>& phoneme_id_map,
                                int& native_sample_rate, float& noise_scale, float& length_scale_default,
                                float& noise_w, int& num_speakers) {
  nlohmann::json j = nlohmann::json::parse(data, data + size);
  if (!j.contains("phoneme_id_map") || !j["phoneme_id_map"].is_object()) {
    throw std::runtime_error("PiperTTS: phoneme_id_map missing in piper config (" + std::string(ctx) + ")");
  }
  phoneme_id_map.clear();
  for (auto it = j["phoneme_id_map"].begin(); it != j["phoneme_id_map"].end(); ++it) {
    std::vector<int64_t> seq;
    if (it.value().is_array()) {
      for (const auto& el : it.value()) {
        seq.push_back(el.get<int64_t>());
      }
    }
    phoneme_id_map[it.key()] = std::move(seq);
  }
  native_sample_rate = 22050;
  noise_scale = 0.667F;
  length_scale_default = 1.F;
  noise_w = 0.8F;
  num_speakers = 1;
  if (j.contains("audio") && j["audio"].is_object() && j["audio"].contains("sample_rate")) {
    native_sample_rate = j["audio"]["sample_rate"].get<int>();
  }
  if (j.contains("inference") && j["inference"].is_object()) {
    const auto& inf = j["inference"];
    if (inf.contains("noise_scale")) {
      noise_scale = inf["noise_scale"].get<float>();
    }
    if (inf.contains("length_scale")) {
      length_scale_default = inf["length_scale"].get<float>();
    }
    if (inf.contains("noise_w")) {
      noise_w = inf["noise_w"].get<float>();
    }
  }
  if (j.contains("num_speakers")) {
    num_speakers = j["num_speakers"].get<int>();
  }
}

}  // namespace

struct PiperTTS::Impl {
  std::filesystem::path voices_dir_;
  std::filesystem::path voices_json_dir_;
  std::filesystem::path onnx_path_;
  std::filesystem::path explicit_onnx_json_path_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "piper_tts"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::unordered_map<std::string, std::vector<int64_t>> phoneme_id_map_{};
  std::unordered_set<std::string> phoneme_map_keys_{};
  std::string piper_ipa_lang_key_{};
  int native_sample_rate_ = 22050;
  float noise_scale_ = 0.667F;
  float length_scale_default_ = 1.F;
  float noise_w_ = 0.8F;
  int num_speakers_ = 1;

  double speed_ = 1.0;
  MoonshineG2POptions g2p_opt_{};
  std::unique_ptr<MoonshineG2P> g2p_{};
  std::string g2p_dialect_{};
  std::string data_subdir_{};
  std::string default_onnx_{};
  std::string onnx_basename_request_{};
  bool user_voices_dir_ = false;
  bool explicit_onnx_file_ = false;
  std::vector<std::string> ort_provider_names_{};
  bool piper_normalize_audio_ = true;
  float piper_output_volume_ = 1.F;
  std::optional<float> noise_scale_override_{};
  std::optional<float> noise_w_override_{};
  FileInformationMap tts_asset_files_{};

  ~Impl() {
    for (auto& e : tts_asset_files_.entries) {
      e.second.free();
    }
  }

  std::vector<float> run_ort_from_phoneme_ids(const std::vector<int64_t>& ids) {
    if (ids.size() < 3) {
      throw std::runtime_error("PiperTTS: phoneme id sequence too short");
    }
    const int64_t ntok = static_cast<int64_t>(ids.size());
    int64_t input_len = ntok;
    const std::array<int64_t, 2> shape_in{1, ntok};
    const std::array<int64_t, 1> shape_len{1};
    double sp = speed_;
    if (sp < 0.25) {
      sp = 0.25;
    }
    if (sp > 4.0) {
      sp = 4.0;
    }
    const float length_scale = length_scale_default_ / static_cast<float>(sp);
    float ns = noise_scale_;
    float nw = noise_w_;
    if (noise_scale_override_.has_value()) {
      ns = noise_scale_override_.value();
    }
    if (noise_w_override_.has_value()) {
      nw = noise_w_override_.value();
    }
    std::array<float, 3> scales{ns, length_scale, nw};
    const std::array<int64_t, 1> shape_scales{3};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, const_cast<int64_t*>(ids.data()), ids.size(), shape_in.data(), shape_in.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_, &input_len, 1, shape_len.data(), shape_len.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_, scales.data(), scales.size(), shape_scales.data(),
                                                     shape_scales.size()));

    std::vector<const char*> in_names{"input", "input_lengths", "scales"};
    int64_t sid = 0;
    if (num_speakers_ > 1) {
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_, &sid, 1, shape_len.data(), shape_len.size()));
      in_names.push_back("sid");
    }

    Ort::RunOptions run_opts{nullptr};
    const char* out_names[] = {"output"};
    auto outputs =
        session_.Run(run_opts, in_names.data(), inputs.data(), inputs.size(), out_names, 1);
    const Ort::Value& outv = outputs[0];
    const auto ti = outv.GetTensorTypeAndShapeInfo();
    if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("PiperTTS: ONNX output is not float32");
    }
    const size_t n_el = ti.GetElementCount();
    const float* ptr = outv.GetTensorData<float>();
    std::vector<float> wave(ptr, ptr + n_el);
    piper_apply_synthesis_output_effects(wave, piper_normalize_audio_, piper_output_volume_);
    if (native_sample_rate_ != PiperTTS::kSampleRateHz) {
      wave = resample_linear(wave, native_sample_rate_, PiperTTS::kSampleRateHz);
    }
    return wave;
  }

  void reload_session_from_onnx() {
    static const std::string k_piper_json("piper/onnx.json");
    static const std::string k_piper_onnx("piper/onnx");
    const std::filesystem::path json_disk =
        !explicit_onnx_json_path_.empty()
            ? explicit_onnx_json_path_
            : piper_model_json_path_for_onnx(onnx_path_, voices_json_dir_);

    const auto jit = tts_asset_files_.entries.find(k_piper_json);
    if (jit != tts_asset_files_.entries.end()) {
      FileInformation& jf = jit->second;
      const uint8_t* jb = nullptr;
      size_t jn = 0;
      jf.load(&jb, &jn);
      load_piper_onnx_json_bytes(reinterpret_cast<const char*>(jb), jn, k_piper_json, phoneme_id_map_,
                                 native_sample_rate_, noise_scale_, length_scale_default_, noise_w_, num_speakers_);
      jf.free();
    } else {
      if (!std::filesystem::is_regular_file(json_disk)) {
        throw std::runtime_error("PiperTTS: missing config " + json_disk.string());
      }
      load_piper_onnx_json(json_disk, phoneme_id_map_, native_sample_rate_, noise_scale_, length_scale_default_,
                           noise_w_, num_speakers_);
    }
    phoneme_map_keys_.clear();
    for (const auto& e : phoneme_id_map_) {
      phoneme_map_keys_.insert(e.first);
    }
    Ort::SessionOptions session_opts = make_ort_options(ort_provider_names_);
    const auto oit = tts_asset_files_.entries.find(k_piper_onnx);
    if (oit != tts_asset_files_.entries.end()) {
      FileInformation& of = oit->second;
      const uint8_t* ob = nullptr;
      size_t on = 0;
      of.load(&ob, &on);
      ort_add_external_initializer_files_for_onnx_model_buffer(session_opts, tts_asset_files_, k_piper_onnx);
      session_ = Ort::Session(env_, ob, on, session_opts);
      of.free();
    } else {
#ifdef _WIN32
      const std::wstring wmodel = onnx_path_.wstring();
      session_ = Ort::Session(env_, wmodel.c_str(), session_opts);
#else
      const std::string u8 = onnx_path_.string();
      session_ = Ort::Session(env_, u8.c_str(), session_opts);
#endif
    }
  }

  explicit Impl(const PiperTTSOptions& opt)
      : speed_(opt.speed),
        ort_provider_names_(opt.ort_provider_names),
        piper_normalize_audio_(opt.piper_normalize_audio),
        piper_output_volume_(opt.piper_output_volume),
        noise_scale_override_(opt.piper_noise_scale_override),
        noise_w_override_(opt.piper_noise_w_override),
        tts_asset_files_(opt.tts_asset_files) {
    g2p_opt_ = opt.g2p_options;
    if (g2p_opt_.g2p_root.empty()) {
      g2p_opt_.g2p_root = std::filesystem::current_path();
    }
    explicit_onnx_file_ = !opt.explicit_onnx_path.empty();
    if (!opt.explicit_onnx_json_path.empty()) {
      explicit_onnx_json_path_ = resolve_path_under_root(g2p_opt_.g2p_root, opt.explicit_onnx_json_path);
    }
    if (!opt.voices_json_dir.empty()) {
      voices_json_dir_ = resolve_path_under_root(g2p_opt_.g2p_root, opt.voices_json_dir);
    }
    user_voices_dir_ =
        explicit_onnx_file_ || !opt.voices_dir.empty() || !opt.voices_json_dir.empty();
    resolve_piper_lang(opt.lang, g2p_opt_, g2p_dialect_, data_subdir_, default_onnx_);
    piper_ipa_lang_key_ = piper_ipa_norm_lang_key(opt.lang, data_subdir_);
    if (explicit_onnx_file_) {
      onnx_path_ = resolve_path_under_root(g2p_opt_.g2p_root, opt.explicit_onnx_path);
      voices_dir_ = onnx_path_.parent_path();
    } else if (!opt.voices_dir.empty()) {
      voices_dir_ = std::filesystem::path(opt.voices_dir);
    } else {
      voices_dir_ = g2p_opt_.g2p_root / data_subdir_ / "piper-voices";
    }
    onnx_basename_request_ = opt.onnx_model;
    if (explicit_onnx_file_) {
      // onnx_path_ already set
    } else {
      onnx_path_ = pick_onnx_path(voices_dir_, onnx_basename_request_, default_onnx_);
    }
    reload_session_from_onnx();
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
  }

  void set_speed(double s) {
    if (!(s > 0.0) || !std::isfinite(s)) {
      throw std::runtime_error("PiperTTS: speed must be a positive finite number");
    }
    speed_ = s;
  }

  void set_lang(const std::string& lk) {
    resolve_piper_lang(lk, g2p_opt_, g2p_dialect_, data_subdir_, default_onnx_);
    piper_ipa_lang_key_ = piper_ipa_norm_lang_key(lk, data_subdir_);
    if (!explicit_onnx_file_) {
      if (!user_voices_dir_) {
        voices_dir_ = g2p_opt_.g2p_root / data_subdir_ / "piper-voices";
      }
      onnx_path_ = pick_onnx_path(voices_dir_, onnx_basename_request_, default_onnx_);
    }
    reload_session_from_onnx();
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
  }

  void set_onnx_model(std::string_view stem_or_base) {
    onnx_basename_request_ = std::string(trim_ascii_ws_copy(stem_or_base));
    if (!explicit_onnx_file_) {
      onnx_path_ = pick_onnx_path(voices_dir_, onnx_basename_request_, default_onnx_);
    }
    reload_session_from_onnx();
  }

  std::vector<float> synthesize(std::string_view text) {
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    if (trim_ascii_ws_copy(ipa).empty()) {
      return {};
    }
    const std::string trimmed_ipa = trim_ascii_ws_copy(ipa);
    const std::string ipa_for_piper =
        coerce_unknown_ipa_chars_to_piper_inventory(
            normalize_g2p_ipa_for_piper(trimmed_ipa, piper_ipa_lang_key_), phoneme_map_keys_, true);
    if (trim_ascii_ws_copy(ipa_for_piper).empty()) {
      return {};
    }
    std::vector<int64_t> ids = ipa_utf8_to_piper_ids(ipa_for_piper, phoneme_id_map_);
    if (ids.size() < 3) {
      return {};
    }
    return run_ort_from_phoneme_ids(ids);
  }

  std::vector<float> synthesize_phoneme_ids(const std::vector<int64_t>& phoneme_ids) {
    return run_ort_from_phoneme_ids(phoneme_ids);
  }
};

PiperTTS::PiperTTS(const PiperTTSOptions& opt) : impl_(std::make_unique<Impl>(opt)) {}

PiperTTS::~PiperTTS() = default;

PiperTTS::PiperTTS(PiperTTS&&) noexcept = default;
PiperTTS& PiperTTS::operator=(PiperTTS&&) noexcept = default;

void PiperTTS::set_lang(std::string_view lang_cli) { impl_->set_lang(std::string(lang_cli)); }

void PiperTTS::set_speed(double speed) { impl_->set_speed(speed); }

void PiperTTS::set_onnx_model(std::string_view basename_or_stem) { impl_->set_onnx_model(basename_or_stem); }

std::vector<float> PiperTTS::synthesize(std::string_view text) { return impl_->synthesize(text); }

std::vector<float> PiperTTS::synthesize_phoneme_ids(const std::vector<int64_t>& phoneme_ids) {
  return impl_->synthesize_phoneme_ids(phoneme_ids);
}

std::vector<std::pair<std::string, bool>> piper_list_voices_with_availability(const PiperTTSOptions& opt) {
  static const std::string k_piper_onnx("piper/onnx");
  MoonshineG2POptions g2p_opt = opt.g2p_options;
  if (g2p_opt.g2p_root.empty()) {
    g2p_opt.g2p_root = std::filesystem::current_path();
  }
  const bool explicit_onnx_file = !opt.explicit_onnx_path.empty();
  if (explicit_onnx_file) {
    const std::filesystem::path onnx_path = resolve_path_under_root(g2p_opt.g2p_root, opt.explicit_onnx_path);
    const std::string stem = onnx_path.stem().string();
    if (std::filesystem::is_regular_file(onnx_path)) {
      return {{stem, true}};
    }
    const auto oit = opt.tts_asset_files.entries.find(k_piper_onnx);
    if (oit != opt.tts_asset_files.entries.end() && oit->second.memory != nullptr && oit->second.memory_size > 0) {
      return {{stem, true}};
    }
    return {{stem, false}};
  }
  // No Piper ONNX layout for Japanese (Kokoro-only); avoid resolve_piper_lang throwing when callers
  // merge Piper + Kokoro voice lists (e.g. ``moonshine_get_tts_voices`` for ``ja`` / ``ja-jp``).
  {
    const std::string lk = normalize_lang_key(opt.lang);
    if (lk == "ja" || lk == "jp" || lk == "ja_jp") {
      return {};
    }
  }
  std::filesystem::path voices_dir;
  std::string default_onnx;
  std::string g2p_dialect;
  std::string data_subdir;
  resolve_piper_lang(opt.lang, g2p_opt, g2p_dialect, data_subdir, default_onnx);
  (void)g2p_dialect;
  if (!opt.voices_dir.empty()) {
    voices_dir = opt.voices_dir;
  } else {
    voices_dir = g2p_opt.g2p_root / data_subdir / "piper-voices";
  }
  std::map<std::string, bool> by_id;
  const auto onnx_present = [&](const std::string& stem) {
    return std::filesystem::is_regular_file(voices_dir / (stem + ".onnx"));
  };
  const std::vector<std::string>& bundled = piper_bundled_voice_stems_for_data_subdir(data_subdir);
  if (!bundled.empty()) {
    for (const std::string& stem : bundled) {
      by_id[stem] = onnx_present(stem);
    }
  } else {
    const std::filesystem::path p(default_onnx);
    const std::string def_stem = p.stem().string();
    by_id[def_stem] = onnx_present(def_stem);
  }
  if (std::filesystem::is_directory(voices_dir)) {
    for (const auto& ent : std::filesystem::directory_iterator(voices_dir)) {
      if (!ent.is_regular_file()) {
        continue;
      }
      const std::filesystem::path& fp = ent.path();
      if (fp.extension() != ".onnx") {
        continue;
      }
      const std::string stem = fp.stem().string();
      by_id[stem] = onnx_present(stem);
    }
  }
  std::vector<std::pair<std::string, bool>> out;
  out.reserve(by_id.size());
  for (const auto& pr : by_id) {
    out.emplace_back(pr.first, pr.second);
  }
  return out;
}

bool piper_default_model_bundle_relative_paths(std::string_view lang_cli, const MoonshineG2POptions& opt,
                                               std::string* onnx_relpath_out, std::string* onnx_json_relpath_out,
                                               std::string_view onnx_model_stem) {
  if (onnx_relpath_out == nullptr || onnx_json_relpath_out == nullptr) {
    return false;
  }
  try {
    std::string g2p_dialect;
    std::string data_subdir;
    std::string default_onnx;
    resolve_piper_lang(std::string(lang_cli), opt, g2p_dialect, data_subdir, default_onnx);
    std::string chosen = default_onnx;
    const std::string stem_req = trim_ascii_ws_copy(std::string(onnx_model_stem));
    if (!stem_req.empty()) {
      chosen = stem_req;
      if (chosen.size() < 5 || chosen.compare(chosen.size() - 5, 5, ".onnx") != 0) {
        chosen += ".onnx";
      }
    }
    *onnx_relpath_out = data_subdir + "/piper-voices/" + chosen;
    std::filesystem::path p(chosen);
    p.replace_extension(".onnx.json");
    *onnx_json_relpath_out = data_subdir + "/piper-voices/" + p.filename().string();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace moonshine_tts
