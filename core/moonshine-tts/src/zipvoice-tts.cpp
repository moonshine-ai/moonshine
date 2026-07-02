#include "zipvoice-tts.h"

#include "debug-utils.h"
#include "g2p-path.h"
#include "ipa-postprocess.h"
#include "moonshine-g2p.h"
#include "moonshine-tts-options.h"
#include "ort-onnx-external-data.h"
#include "utf8-utils.h"
#include "zipvoice-custom-ops.h"
#include "zipvoice-mel.h"
#include "zipvoice-voices.h"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace moonshine_tts {

namespace {

std::string normalize_lang_key(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == ' ' || c == '-') {
      c = '_';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

/// English-only for now; structured so more locales can be added. Returns the MoonshineG2P dialect id
/// and the ``normalize_g2p_ipa_for_piper`` language key (ZipVoice tokens are espeak IPA like Piper).
void resolve_zipvoice_lang(const std::string& lang, std::string& g2p_dialect, std::string& ipa_lang_key) {
  const std::string k = normalize_lang_key(lang);
  if (k == "en_gb" || k == "en-gb") {
    g2p_dialect = "en_gb";
  } else {
    g2p_dialect = "en_us";
  }
  ipa_lang_key = "en_us";  // ZipVoice was trained with en-us espeak phonemes.
}

Ort::SessionOptions make_ort_options() {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(0);
  opts.SetInterOpNumThreads(0);
  return opts;
}

std::vector<float> resample_linear(const std::vector<float>& x, int src_sr, int dst_sr) {
  if (src_sr == dst_sr || x.size() < 2) {
    return x;
  }
  const double duration = static_cast<double>(x.size() - 1) / static_cast<double>(src_sr);
  const size_t n_out = std::max<size_t>(2, static_cast<size_t>(std::llround(duration * dst_sr)) + 1);
  std::vector<float> y(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    const double t = static_cast<double>(i) * duration / static_cast<double>(n_out - 1);
    const double fidx = t * static_cast<double>(src_sr);
    const size_t i0 = static_cast<size_t>(std::floor(fidx));
    const size_t i1 = std::min(i0 + 1, x.size() - 1);
    const double frac = fidx - static_cast<double>(i0);
    y[i] = static_cast<float>(x[i0] * (1.0 - frac) + x[i1] * frac);
  }
  return y;
}

/// Approximate ``remove_silence`` edge trimming (pydub ``remove_silence_edges``): drop leading and
/// trailing samples below ``-50`` dBFS, keeping ``100`` ms of margin, then append ``trail_sil_ms`` of
/// silence. A sample-level approximation of pydub's chunk-based detector; adequate for clean clips.
std::vector<float> trim_edge_silence(const std::vector<float>& wav, int sample_rate, int trail_sil_ms) {
  const float thresh = 0.0031622776601683794F;  // 10^(-50/20)
  const int keep = (100 * sample_rate) / 1000;
  size_t begin = 0;
  while (begin < wav.size() && std::fabs(wav[begin]) <= thresh) {
    ++begin;
  }
  size_t start = (begin > static_cast<size_t>(keep)) ? begin - static_cast<size_t>(keep) : 0;
  size_t last = wav.size();
  while (last > start && std::fabs(wav[last - 1]) <= thresh) {
    --last;
  }
  size_t end = std::min(wav.size(), last + static_cast<size_t>(keep));
  if (end <= start) {
    // All silence: keep the original to avoid an empty clone.
    start = 0;
    end = wav.size();
  }
  std::vector<float> out(wav.begin() + static_cast<std::ptrdiff_t>(start),
                         wav.begin() + static_cast<std::ptrdiff_t>(end));
  const int trail = (trail_sil_ms * sample_rate) / 1000;
  if (trail > 0) {
    out.insert(out.end(), static_cast<size_t>(trail), 0.F);
  }
  return out;
}

float rms_of(const std::vector<float>& x) {
  if (x.empty()) {
    return 0.F;
  }
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * static_cast<double>(v);
  }
  return static_cast<float>(std::sqrt(ss / static_cast<double>(x.size())));
}

/// ``get_time_steps``: linspace(0, 1, num_step + 1) then ``t_shift * t / (1 + (t_shift - 1) * t)``.
std::vector<float> get_time_steps(int num_step, float t_shift) {
  std::vector<float> ts(static_cast<size_t>(num_step + 1));
  for (int i = 0; i <= num_step; ++i) {
    const float lin = static_cast<float>(i) / static_cast<float>(num_step);
    ts[static_cast<size_t>(i)] = t_shift * lin / (1.F + (t_shift - 1.F) * lin);
  }
  return ts;
}

std::unordered_map<std::string, int> parse_tokens_txt(const char* data, size_t size) {
  std::unordered_map<std::string, int> token2id;
  size_t i = 0;
  while (i < size) {
    size_t line_end = i;
    while (line_end < size && data[line_end] != '\n') {
      ++line_end;
    }
    size_t content_end = line_end;
    if (content_end > i && data[content_end - 1] == '\r') {
      --content_end;
    }
    // Split on the first tab: "{token}\t{id}". Token may itself be a space.
    size_t tab = i;
    while (tab < content_end && data[tab] != '\t') {
      ++tab;
    }
    if (tab < content_end) {
      const std::string token(data + i, data + tab);
      const std::string id_str(data + tab + 1, data + content_end);
      if (!id_str.empty()) {
        try {
          token2id[token] = std::stoi(id_str);
        } catch (const std::exception&) {
          // skip malformed line
        }
      }
    }
    i = line_end + 1;
  }
  return token2id;
}

}  // namespace

struct ZipVoiceTTS::Impl {
  // Config / controls.
  double speed_ = 1.0;
  bool normalize_audio_ = false;
  float output_volume_ = 1.F;
  int num_step_ = 8;
  float guidance_scale_ = 3.F;
  float t_shift_ = 0.5F;
  float feat_scale_ = 0.1F;
  float target_rms_ = 0.1F;
  unsigned int seed_ = 666U;
  int feat_dim_ = VocosFbank::kNMels;

  MoonshineG2POptions g2p_opt_{};
  std::unique_ptr<MoonshineG2P> g2p_{};
  std::string ipa_lang_key_ = "en_us";

  std::unordered_map<std::string, int> token2id_{};
  std::unordered_set<std::string> token_keys_{};

  FileInformationMap tts_files_{};
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moonshine_zipvoice"};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  Ort::Session text_encoder_{nullptr};
  Ort::Session fm_decoder_{nullptr};
  Ort::Session vocoder_{nullptr};
  std::vector<std::string> te_in_, te_out_, fm_in_, fm_out_, vo_in_, vo_out_;

  // Prepared clone.
  std::vector<float> clone_features_;  // [T_clone * feat_dim] row-major
  int clone_frames_ = 0;
  float clone_rms_ = 0.F;
  std::vector<int64_t> clone_token_ids_{};

  ~Impl() {
    for (auto& e : tts_files_.entries) {
      e.second.free();
    }
  }

  Ort::Session load_session(std::string_view key, bool register_custom_ops,
                            const std::vector<std::string>& providers) {
    (void)providers;
    Ort::SessionOptions opts = make_ort_options();
    if (register_custom_ops) {
      zipvoice_register_custom_ops(opts);
    }
    const std::string k(key);
    const auto it = tts_files_.entries.find(k);
    if (it != tts_files_.entries.end() && it->second.memory != nullptr && it->second.memory_size > 0) {
      const uint8_t* b = nullptr;
      size_t n = 0;
      it->second.load(&b, &n);
      ort_add_external_initializer_files_for_onnx_model_buffer(opts, tts_files_, key);
      Ort::Session s(env_, b, n, opts);
      it->second.free();
      return s;
    }
    std::filesystem::path p = (it != tts_files_.entries.end() && !it->second.path.empty())
                                  ? resolve_path_under_root(g2p_opt_.g2p_root, it->second.path)
                                  : resolve_path_under_root(g2p_opt_.g2p_root, std::filesystem::path(k));
    resolve_disk_model_file_path(p);
    if (!std::filesystem::is_regular_file(p)) {
      throw std::runtime_error("ZipVoiceTTS: missing model file " + p.string());
    }
#ifdef _WIN32
    const std::wstring w = p.wstring();
    return Ort::Session(env_, w.c_str(), opts);
#else
    const std::string u8 = p.string();
    return Ort::Session(env_, u8.c_str(), opts);
#endif
  }

  std::vector<uint8_t> load_asset_bytes(std::string_view key) {
    const std::string k(key);
    const auto it = tts_files_.entries.find(k);
    if (it != tts_files_.entries.end() && it->second.memory != nullptr && it->second.memory_size > 0) {
      const uint8_t* b = nullptr;
      size_t n = 0;
      it->second.load(&b, &n);
      std::vector<uint8_t> out(b, b + n);
      it->second.free();
      return out;
    }
    std::filesystem::path p = (it != tts_files_.entries.end() && !it->second.path.empty())
                                  ? resolve_path_under_root(g2p_opt_.g2p_root, it->second.path)
                                  : resolve_path_under_root(g2p_opt_.g2p_root, std::filesystem::path(k));
    if (!std::filesystem::is_regular_file(p)) {
      throw std::runtime_error("ZipVoiceTTS: missing asset " + p.string());
    }
    std::ifstream f(p, std::ios::binary);
    if (!f) {
      throw std::runtime_error("ZipVoiceTTS: cannot open " + p.string());
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  std::vector<int64_t> ipa_text_to_token_ids(const std::string& text) {
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    const std::string trimmed = trim_ascii_ws_copy(ipa);
    if (trimmed.empty()) {
      return {};
    }
    const std::string ready = coerce_unknown_ipa_chars_to_piper_inventory(
        normalize_g2p_ipa_for_piper(trimmed, ipa_lang_key_), token_keys_, true);
    std::vector<int64_t> ids;
    for (const std::string& ch : utf8_split_codepoints(ready)) {
      const auto it = token2id_.find(ch);
      if (it != token2id_.end()) {
        ids.push_back(it->second);
      }
    }
    return ids;
  }

  explicit Impl(const ZipVoiceTTSOptions& opt) {
    if (!(opt.speed > 0.0) || !std::isfinite(opt.speed)) {
      throw std::runtime_error("ZipVoiceTTS: speed must be a positive finite number");
    }
    speed_ = opt.speed;
    normalize_audio_ = opt.normalize_audio;
    output_volume_ = opt.output_volume;
    t_shift_ = opt.t_shift;
    feat_scale_ = opt.feat_scale;
    target_rms_ = opt.target_rms;
    seed_ = opt.seed;
    num_step_ = opt.num_step > 0 ? opt.num_step : (opt.distill ? 8 : 16);
    guidance_scale_ = opt.guidance_scale >= 0.F ? opt.guidance_scale : (opt.distill ? 3.F : 1.F);

    g2p_opt_ = opt.g2p_options;
    if (g2p_opt_.g2p_root.empty()) {
      g2p_opt_.g2p_root = std::filesystem::current_path();
    }
    tts_files_ = opt.tts_asset_files;

    std::string g2p_dialect;
    resolve_zipvoice_lang(opt.lang, g2p_dialect, ipa_lang_key_);

    // tokens.txt (phoneme -> id) and optional model.json (feat_dim).
    {
      const std::vector<uint8_t> tok = load_asset_bytes(kTtsZipVoiceTokensKey);
      token2id_ = parse_tokens_txt(reinterpret_cast<const char*>(tok.data()), tok.size());
      if (token2id_.empty()) {
        throw std::runtime_error("ZipVoiceTTS: empty or invalid tokens.txt");
      }
      for (const auto& e : token2id_) {
        token_keys_.insert(e.first);
      }
    }
    if (tts_files_.entries.count(std::string(kTtsZipVoiceModelJsonKey)) != 0 ||
        std::filesystem::is_regular_file(
            resolve_path_under_root(g2p_opt_.g2p_root, std::filesystem::path(kTtsZipVoiceModelJsonKey)))) {
      try {
        const std::vector<uint8_t> cfg = load_asset_bytes(kTtsZipVoiceModelJsonKey);
        nlohmann::json j = nlohmann::json::parse(cfg.begin(), cfg.end());
        if (j.contains("model") && j["model"].contains("feat_dim")) {
          feat_dim_ = j["model"]["feat_dim"].get<int>();
        }
      } catch (const std::exception&) {
        feat_dim_ = VocosFbank::kNMels;
      }
    }
    if (feat_dim_ != VocosFbank::kNMels) {
      throw std::runtime_error("ZipVoiceTTS: unsupported feat_dim (expected 100)");
    }

    text_encoder_ = load_session(kTtsZipVoiceTextEncoderKey, /*register_custom_ops=*/false,
                                 opt.ort_provider_names);
    fm_decoder_ = load_session(kTtsZipVoiceFmDecoderKey, /*register_custom_ops=*/true,
                               opt.ort_provider_names);
    vocoder_ = load_session(kTtsZipVoiceVocoderKey, /*register_custom_ops=*/false, opt.ort_provider_names);
    te_in_ = text_encoder_.GetInputNames();
    te_out_ = text_encoder_.GetOutputNames();
    fm_in_ = fm_decoder_.GetInputNames();
    fm_out_ = fm_decoder_.GetOutputNames();
    vo_in_ = vocoder_.GetInputNames();
    vo_out_ = vocoder_.GetOutputNames();
    if (te_in_.size() < 4 || fm_in_.size() < 5 || vo_in_.empty()) {
      throw std::runtime_error("ZipVoiceTTS: unexpected ONNX input signature");
    }

    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect, g2p_opt_);

    // Resolve the reference (clone) clip and its transcript.
    std::vector<float> clone_pcm;
    int clone_sr = opt.clone_sample_rate;
    std::string clone_text = opt.clone_transcript;
    if (!opt.voice_id.empty()) {
      const ZipVoiceBuiltinVoice* v = zipvoice_find_builtin_voice(opt.voice_id);
      if (v == nullptr) {
        throw std::runtime_error("ZipVoiceTTS: unknown built-in voice '" + opt.voice_id + "'");
      }
      clone_pcm = zipvoice_builtin_voice_pcm_to_float(*v);
      clone_sr = static_cast<int>(v->sample_rate);
      clone_text = v->clone_transcript;
    } else {
      clone_pcm = opt.clone_pcm;
    }
    if (clone_pcm.empty()) {
      throw std::runtime_error(
          "ZipVoiceTTS: no reference voice supplied (set a built-in voice id or clone PCM)");
    }
    if (clone_sr != VocosFbank::kSampleRate) {
      clone_pcm = resample_linear(clone_pcm, clone_sr, VocosFbank::kSampleRate);
    }
    clone_pcm = trim_edge_silence(clone_pcm, VocosFbank::kSampleRate, /*trail_sil_ms=*/200);
    clone_rms_ = rms_of(clone_pcm);
    if (clone_rms_ > 0.F && clone_rms_ < target_rms_) {
      const float g = target_rms_ / clone_rms_;
      for (float& s : clone_pcm) {
        s *= g;
      }
    }
    VocosFbank fbank;
    clone_features_ = fbank.extract(clone_pcm, &clone_frames_);
    for (float& v : clone_features_) {
      v *= feat_scale_;
    }
    clone_token_ids_ = ipa_text_to_token_ids(clone_text);
  }

  double speed() const { return speed_; }
  void set_speed(double s) {
    if (!(s > 0.0) || !std::isfinite(s)) {
      throw std::runtime_error("ZipVoiceTTS: speed must be a positive finite number");
    }
    speed_ = s;
  }
  bool normalize_audio() const { return normalize_audio_; }
  void set_normalize_audio(bool on) { normalize_audio_ = on; }
  float output_volume() const { return output_volume_; }
  void set_output_volume(float v) { output_volume_ = v; }

  // Runs the text encoder for one target-token chunk and returns text_condition [frames*feat_dim]
  // (row-major), setting *out_frames.
  std::vector<float> run_text_encoder(const std::vector<int64_t>& tokens, int* out_frames) {
    std::vector<int64_t> tok = tokens;
    std::vector<int64_t> ptok = clone_token_ids_;
    if (tok.empty()) {
      tok.push_back(token2id_.count(" ") ? token2id_.at(" ") : 0);
    }
    if (ptok.empty()) {
      ptok.push_back(token2id_.count(" ") ? token2id_.at(" ") : 0);
    }
    const std::array<int64_t, 2> shape_tok{1, static_cast<int64_t>(tok.size())};
    const std::array<int64_t, 2> shape_ptok{1, static_cast<int64_t>(ptok.size())};
    int64_t pfl = clone_frames_;
    float sp = static_cast<float>(speed_);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_, tok.data(), tok.size(), shape_tok.data(),
                                                       shape_tok.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_, ptok.data(), ptok.size(), shape_ptok.data(),
                                                       shape_ptok.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_, &pfl, 1, nullptr, 0));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_, &sp, 1, nullptr, 0));

    std::array<const char*, 4> in_names{te_in_[0].c_str(), te_in_[1].c_str(), te_in_[2].c_str(),
                                        te_in_[3].c_str()};
    const char* out_names[] = {te_out_[0].c_str()};
    Ort::RunOptions run_opts{nullptr};
    auto outputs =
        text_encoder_.Run(run_opts, in_names.data(), inputs.data(), inputs.size(), out_names, 1);
    const Ort::Value& tc = outputs[0];
    const auto ti = tc.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = ti.GetShape();
    if (shape.size() != 3) {
      throw std::runtime_error("ZipVoiceTTS: text_condition rank != 3");
    }
    const int frames = static_cast<int>(shape[1]);
    if (out_frames != nullptr) {
      *out_frames = frames;
    }
    const size_t n = ti.GetElementCount();
    const float* p = tc.GetTensorData<float>();
    return std::vector<float>(p, p + n);
  }

  // One flow-matching Euler solve for a chunk; returns predicted features [T_gen*feat_dim] (row-major).
  std::vector<float> sample_chunk(const std::vector<int64_t>& tokens, int* out_gen_frames) {
    int frames = 0;
    std::vector<float> text_condition = run_text_encoder(tokens, &frames);
    if (frames <= 0) {
      *out_gen_frames = 0;
      return {};
    }
    const size_t feat = static_cast<size_t>(feat_dim_);
    const size_t total = static_cast<size_t>(frames) * feat;

    // x ~ N(0, 1)
    std::vector<float> x(total);
    std::mt19937 rng(seed_);
    std::normal_distribution<float> dist(0.F, 1.F);
    for (size_t i = 0; i < total; ++i) {
      x[i] = dist(rng);
    }
    // speech_condition = pad(clone_features, to frames) along time.
    std::vector<float> speech_condition(total, 0.F);
    const size_t copy_frames = std::min<size_t>(static_cast<size_t>(clone_frames_),
                                                static_cast<size_t>(frames));
    std::copy(clone_features_.begin(),
              clone_features_.begin() + static_cast<std::ptrdiff_t>(copy_frames * feat),
              speech_condition.begin());

    const std::vector<float> ts = get_time_steps(num_step_, t_shift_);
    const std::array<int64_t, 3> shape3{1, frames, static_cast<int64_t>(feat)};
    float guidance = guidance_scale_;

    std::array<const char*, 5> in_names{fm_in_[0].c_str(), fm_in_[1].c_str(), fm_in_[2].c_str(),
                                        fm_in_[3].c_str(), fm_in_[4].c_str()};
    const char* out_names[] = {fm_out_[0].c_str()};
    Ort::RunOptions run_opts{nullptr};

    for (int step = 0; step < num_step_; ++step) {
      float t_val = ts[static_cast<size_t>(step)];
      std::vector<Ort::Value> inputs;
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, &t_val, 1, nullptr, 0));
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, x.data(), x.size(), shape3.data(),
                                                       shape3.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, text_condition.data(),
                                                       text_condition.size(), shape3.data(),
                                                       shape3.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, speech_condition.data(),
                                                       speech_condition.size(), shape3.data(),
                                                       shape3.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, &guidance, 1, nullptr, 0));
      auto outputs =
          fm_decoder_.Run(run_opts, in_names.data(), inputs.data(), inputs.size(), out_names, 1);
      const Ort::Value& vv = outputs[0];
      const float* vptr = vv.GetTensorData<float>();
      const float dt = ts[static_cast<size_t>(step + 1)] - ts[static_cast<size_t>(step)];
      for (size_t i = 0; i < total; ++i) {
        x[i] += vptr[i] * dt;
      }
    }

    // Trim the clone frames from the front.
    const int gen_frames = frames - clone_frames_;
    *out_gen_frames = gen_frames;
    if (gen_frames <= 0) {
      return {};
    }
    std::vector<float> pred(static_cast<size_t>(gen_frames) * feat);
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(clone_frames_) * feat),
              x.end(), pred.begin());
    return pred;
  }

  // pred [T_gen*feat] row-major -> vocoder -> waveform.
  std::vector<float> run_vocoder(const std::vector<float>& pred, int gen_frames) {
    const size_t feat = static_cast<size_t>(feat_dim_);
    // mel: [1, feat, T_gen], mel[0,c,t] = pred[t,c] / feat_scale.
    std::vector<float> mel(static_cast<size_t>(gen_frames) * feat);
    for (int t = 0; t < gen_frames; ++t) {
      for (size_t c = 0; c < feat; ++c) {
        mel[c * static_cast<size_t>(gen_frames) + static_cast<size_t>(t)] =
            pred[static_cast<size_t>(t) * feat + c] / feat_scale_;
      }
    }
    const std::array<int64_t, 3> shape{1, static_cast<int64_t>(feat), gen_frames};
    std::vector<Ort::Value> inputs;
    inputs.push_back(
        Ort::Value::CreateTensor<float>(mem_, mel.data(), mel.size(), shape.data(), shape.size()));
    const char* in_names[] = {vo_in_[0].c_str()};
    const char* out_names[] = {vo_out_[0].c_str()};
    Ort::RunOptions run_opts{nullptr};
    auto outputs = vocoder_.Run(run_opts, in_names, inputs.data(), inputs.size(), out_names, 1);
    const Ort::Value& w = outputs[0];
    const auto ti = w.GetTensorTypeAndShapeInfo();
    const size_t n = ti.GetElementCount();
    const float* p = w.GetTensorData<float>();
    std::vector<float> wav(n);
    for (size_t i = 0; i < n; ++i) {
      float v = p[i];
      wav[i] = std::max(-1.F, std::min(1.F, v));
    }
    return wav;
  }

  // Split target token ids into chunks near a target size, preferring space-token boundaries.
  std::vector<std::vector<int64_t>> chunk_target_ids(const std::vector<int64_t>& ids) {
    // max_tokens ~ so total (clone + generated) audio stays around 25s (mirrors speak.py).
    int max_tokens = 400;
    if (!clone_token_ids_.empty() && clone_frames_ > 0) {
      const double clone_duration =
          static_cast<double>(clone_frames_) * VocosFbank::kHop / VocosFbank::kSampleRate;
      const double token_duration =
          clone_duration / (static_cast<double>(clone_token_ids_.size()) * speed_);
      if (token_duration > 1e-6) {
        const int m = static_cast<int>((25.0 - clone_duration) / token_duration);
        max_tokens = std::max(1, m);
      }
    }
    std::vector<std::vector<int64_t>> chunks;
    if (static_cast<int>(ids.size()) <= max_tokens) {
      chunks.push_back(ids);
      return chunks;
    }
    const int space_id = token2id_.count(" ") ? token2id_.at(" ") : -1;
    size_t start = 0;
    while (start < ids.size()) {
      size_t end = std::min(ids.size(), start + static_cast<size_t>(max_tokens));
      if (end < ids.size() && space_id >= 0) {
        size_t cut = end;
        while (cut > start && ids[cut] != static_cast<int64_t>(space_id)) {
          --cut;
        }
        if (cut > start) {
          end = cut;
        }
      }
      chunks.emplace_back(ids.begin() + static_cast<std::ptrdiff_t>(start),
                          ids.begin() + static_cast<std::ptrdiff_t>(end));
      start = end;
      while (start < ids.size() && space_id >= 0 && ids[start] == static_cast<int64_t>(space_id)) {
        ++start;
      }
      if (space_id < 0) {
        // no boundary token; already advanced by max_tokens
      }
    }
    return chunks;
  }

  static std::vector<float> cross_fade_concat(const std::vector<std::vector<float>>& chunks,
                                              float fade_seconds, int sample_rate) {
    if (chunks.empty()) {
      return {};
    }
    if (chunks.size() == 1) {
      return chunks[0];
    }
    const int fade = static_cast<int>(fade_seconds * static_cast<float>(sample_rate));
    if (fade <= 0) {
      std::vector<float> out;
      for (const auto& c : chunks) {
        out.insert(out.end(), c.begin(), c.end());
      }
      return out;
    }
    std::vector<float> out = chunks[0];
    for (size_t ci = 1; ci < chunks.size(); ++ci) {
      const std::vector<float>& next = chunks[ci];
      const int k = std::min({fade, static_cast<int>(out.size()), static_cast<int>(next.size())});
      if (k <= 0) {
        out.insert(out.end(), next.begin(), next.end());
        continue;
      }
      const size_t base = out.size() - static_cast<size_t>(k);
      for (int i = 0; i < k; ++i) {
        const float f = 1.F - static_cast<float>(i) / static_cast<float>(k - 1 > 0 ? k - 1 : 1);
        out[base + static_cast<size_t>(i)] =
            out[base + static_cast<size_t>(i)] * f + next[static_cast<size_t>(i)] * (1.F - f);
      }
      out.insert(out.end(), next.begin() + k, next.end());
    }
    return out;
  }

  std::vector<float> synthesize(std::string_view text) {
    std::vector<int64_t> ids = ipa_text_to_token_ids(std::string(text));
    if (ids.empty()) {
      return {};
    }
    const std::vector<std::vector<int64_t>> chunks = chunk_target_ids(ids);
    std::vector<std::vector<float>> wavs;
    wavs.reserve(chunks.size());
    for (const auto& chunk : chunks) {
      int gen_frames = 0;
      std::vector<float> pred = sample_chunk(chunk, &gen_frames);
      if (gen_frames <= 0 || pred.empty()) {
        continue;
      }
      std::vector<float> wav = run_vocoder(pred, gen_frames);
      if (clone_rms_ > 0.F && clone_rms_ < target_rms_) {
        const float g = clone_rms_ / target_rms_;
        for (float& s : wav) {
          s *= g;
        }
      }
      wavs.push_back(std::move(wav));
    }
    std::vector<float> out = cross_fade_concat(wavs, 0.1F, kSampleRateHz);
    out = zipvoice_compress_long_pauses(out, kSampleRateHz);
    out = trim_edge_silence(out, kSampleRateHz, /*trail_sil_ms=*/0);
    apply_synthesis_output_effects(out, normalize_audio_, output_volume_);
    return out;
  }
};

ZipVoiceTTS::ZipVoiceTTS(const ZipVoiceTTSOptions& opt) : impl_(std::make_unique<Impl>(opt)) {}
ZipVoiceTTS::~ZipVoiceTTS() = default;
ZipVoiceTTS::ZipVoiceTTS(ZipVoiceTTS&&) noexcept = default;
ZipVoiceTTS& ZipVoiceTTS::operator=(ZipVoiceTTS&&) noexcept = default;

void ZipVoiceTTS::set_speed(double speed) { impl_->set_speed(speed); }
double ZipVoiceTTS::speed() const { return impl_->speed(); }
bool ZipVoiceTTS::normalize_audio() const { return impl_->normalize_audio(); }
void ZipVoiceTTS::set_normalize_audio(bool on) { impl_->set_normalize_audio(on); }
float ZipVoiceTTS::output_volume() const { return impl_->output_volume(); }
void ZipVoiceTTS::set_output_volume(float volume) { impl_->set_output_volume(volume); }

std::vector<float> ZipVoiceTTS::synthesize(std::string_view text) { return impl_->synthesize(text); }

std::vector<float> zipvoice_compress_long_pauses(const std::vector<float>& wav, int sample_rate,
                                                 float max_silence_ms, float keep_silence_ms,
                                                 float fade_ms) {
  if (wav.size() < 4 || sample_rate <= 0) {
    return wav;
  }

  const int win = std::max(1, (20 * sample_rate) / 1000);
  std::vector<float> env(wav.size(), 0.F);
  double acc = 0.0;
  for (size_t i = 0; i < wav.size(); ++i) {
    acc += std::fabs(wav[i]);
    if (i >= static_cast<size_t>(win)) {
      acc -= std::fabs(wav[i - static_cast<size_t>(win)]);
    }
    const int denom = static_cast<int>(std::min(i + 1, static_cast<size_t>(win)));
    env[i] = static_cast<float>(acc / static_cast<double>(denom));
  }

  float peak = 0.F;
  for (float e : env) {
    peak = std::max(peak, e);
  }
  const float thresh = std::max(0.0031622776601683794F, 0.04F * peak);  // floor -50 dBFS

  const int min_silence = std::max(1, (80 * sample_rate) / 1000);
  const int max_silence = std::max(min_silence + 1, static_cast<int>(max_silence_ms * sample_rate / 1000.F));
  const int keep_silence = std::max(1, static_cast<int>(keep_silence_ms * sample_rate / 1000.F));
  const int fade = std::max(1, static_cast<int>(fade_ms * sample_rate / 1000.F));

  struct Run {
    size_t start = 0;
    size_t end = 0;
  };
  std::vector<Run> runs;
  size_t i = 0;
  while (i < wav.size()) {
    if (env[i] < thresh) {
      size_t j = i;
      while (j < wav.size() && env[j] < thresh) {
        ++j;
      }
      if (static_cast<int>(j - i) >= min_silence) {
        runs.push_back({i, j});
      }
      i = j;
    } else {
      ++i;
    }
  }
  if (runs.empty()) {
    return wav;
  }

  auto fade_out_tail = [&](std::vector<float>& buf, int fade_len) {
    if (fade_len <= 0 || buf.empty()) {
      return;
    }
    fade_len = std::min(fade_len, static_cast<int>(buf.size()));
    for (int k = 0; k < fade_len; ++k) {
      const float g = 1.F - static_cast<float>(k + 1) / static_cast<float>(fade_len);
      buf[buf.size() - static_cast<size_t>(fade_len) + static_cast<size_t>(k)] *= g;
    }
  };

  auto append_fade_in = [&](std::vector<float>& buf, size_t speech_start, size_t speech_end) {
    const int fade_len = std::min(fade, static_cast<int>(speech_end - speech_start));
    if (fade_len <= 0) {
      return;
    }
    for (int k = 0; k < fade_len; ++k) {
      const float g = static_cast<float>(k + 1) / static_cast<float>(fade_len);
      buf.push_back(wav[speech_start + static_cast<size_t>(k)] * g);
    }
    if (speech_start + static_cast<size_t>(fade_len) < speech_end) {
      buf.insert(buf.end(), wav.begin() + static_cast<std::ptrdiff_t>(speech_start + fade_len),
                 wav.begin() + static_cast<std::ptrdiff_t>(speech_end));
    }
  };

  std::vector<float> out;
  out.reserve(wav.size());
  size_t cursor = 0;
  bool need_fade_in = false;
  for (const Run& run : runs) {
    if (run.start > cursor) {
      if (need_fade_in) {
        append_fade_in(out, cursor, run.start);
      } else {
        out.insert(out.end(), wav.begin() + static_cast<std::ptrdiff_t>(cursor),
                   wav.begin() + static_cast<std::ptrdiff_t>(run.start));
      }
      need_fade_in = false;
    }

    const size_t run_len = run.end - run.start;
    if (static_cast<int>(run_len) <= max_silence) {
      out.insert(out.end(), wav.begin() + static_cast<std::ptrdiff_t>(run.start),
                 wav.begin() + static_cast<std::ptrdiff_t>(run.end));
    } else {
      fade_out_tail(out, fade);
      const size_t keep = std::min(static_cast<size_t>(keep_silence), run_len);
      out.insert(out.end(), wav.begin() + static_cast<std::ptrdiff_t>(run.start),
                 wav.begin() + static_cast<std::ptrdiff_t>(run.start + keep));
      need_fade_in = true;
    }
    cursor = run.end;
  }

  if (cursor < wav.size()) {
    if (need_fade_in) {
      append_fade_in(out, cursor, wav.size());
    } else {
      out.insert(out.end(), wav.begin() + static_cast<std::ptrdiff_t>(cursor), wav.end());
    }
  }

  return out;
}

}  // namespace moonshine_tts
