#include "phonon-tts.h"

#include <nlohmann/json.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bin-tokenizer.h"
#include "g2p-path.h"
#include "moonshine-tts-options.h"
#include "ort-session-options.h"
#include "ort-utils-cxx.h"
#include "phonon-text.h"
#include "utf8-utils.h"

namespace moonshine_tts {

namespace {

/// What ``model.json`` says about the exported graphs and the generation
/// defaults. Everything the loop needs that is not a weight lives here, so the
/// engine does not hard-code the shape of one checkpoint.
struct PhononMeta {
  int hidden_size = 512;
  int num_layers = 6;
  int num_heads = 8;
  int head_dim = 64;
  int latent_dim = 32;
  int audio_prompt_dim = 512;
  double frame_rate = 12.5;
  int sample_rate = 24000;
  int frame_size = 1920;
  int mimi_hidden = 512;
  int mimi_heads = 8;
  int mimi_head_dim = 64;
  int mimi_window = 251;
  int mimi_layers = 2;
  int mimi_stride = 16;
  int mimi_context_frames = 2;
  float temperature = 0.3F;
  float eos_threshold = 4.F;
  int min_frames_before_eos = 0;
  int frames_after_eos = 1;
  int frames_after_eos_short = 3;
  int short_text_max_words = 4;
};

template <typename T>
T json_or(const nlohmann::json& j, const char* key, T fallback) {
  if (j.contains(key) && !j[key].is_null()) {
    return j[key].get<T>();
  }
  return fallback;
}

PhononMeta parse_meta(const std::vector<uint8_t>& bytes) {
  nlohmann::json j = nlohmann::json::parse(bytes.begin(), bytes.end());
  PhononMeta m;
  m.hidden_size = json_or(j, "hidden_size", m.hidden_size);
  m.num_layers = json_or(j, "num_hidden_layers", m.num_layers);
  m.num_heads = json_or(j, "num_attention_heads", m.num_heads);
  m.head_dim = json_or(j, "head_dim", m.head_dim);
  m.latent_dim = json_or(j, "latent_dim", m.latent_dim);
  m.audio_prompt_dim = json_or(j, "audio_prompt_dim", m.audio_prompt_dim);
  m.frame_rate = json_or(j, "frame_rate", m.frame_rate);
  m.sample_rate = json_or(j, "sample_rate", m.sample_rate);
  m.frame_size = json_or(j, "frame_size", m.frame_size);
  m.mimi_hidden = json_or(j, "mimi_hidden_size", m.mimi_hidden);
  m.mimi_heads = json_or(j, "mimi_heads", m.mimi_heads);
  m.mimi_head_dim = json_or(j, "mimi_head_dim", m.mimi_head_dim);
  m.mimi_window = json_or(j, "mimi_sliding_window", m.mimi_window);
  m.mimi_layers = json_or(j, "mimi_transformer_layers", m.mimi_layers);
  m.mimi_stride = json_or(j, "mimi_latent_stride", m.mimi_stride);
  m.mimi_context_frames =
      json_or(j, "mimi_seanet_context_frames", m.mimi_context_frames);
  m.temperature = json_or(j, "temperature", m.temperature);
  m.eos_threshold = json_or(j, "eos_threshold", m.eos_threshold);
  m.min_frames_before_eos =
      json_or(j, "min_frames_before_eos", m.min_frames_before_eos);
  m.frames_after_eos = json_or(j, "frames_after_eos", m.frames_after_eos);
  m.frames_after_eos_short =
      json_or(j, "frames_after_eos_short_text", m.frames_after_eos_short);
  m.short_text_max_words =
      json_or(j, "short_text_max_words", m.short_text_max_words);
  if (m.sample_rate != PhononTTS::kSampleRateHz ||
      m.frame_size != PhononTTS::kFrameSamples) {
    throw std::runtime_error(
        "PhononTTS: model.json describes a sample rate or frame size this "
        "build does not support");
  }
  if (m.frame_size % m.mimi_stride != 0) {
    throw std::runtime_error(
        "PhononTTS: frame_size must be a multiple of mimi_latent_stride");
  }
  return m;
}

/// The ``speaker_wavs`` tensor of a Phonon voice file, ``[1, dim, frames]``
/// float32, returned frame-major (``frames * dim``) as the prompt graph takes
/// it. A safetensors file is an 8-byte little-endian header length, a JSON
/// header, then the raw tensor bytes.
std::vector<float> parse_voice_safetensors(const std::vector<uint8_t>& bytes,
                                           int expected_dim, int* out_frames) {
  if (bytes.size() < 8) {
    throw std::runtime_error("PhononTTS: voice file is too short");
  }
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | bytes[static_cast<size_t>(i)];
  }
  if (header_len > bytes.size() - 8) {
    throw std::runtime_error("PhononTTS: voice file header is truncated");
  }
  const auto header_begin = bytes.begin() + 8;
  const auto header_end =
      header_begin + static_cast<std::ptrdiff_t>(header_len);
  nlohmann::json header = nlohmann::json::parse(header_begin, header_end);
  if (!header.contains("speaker_wavs")) {
    throw std::runtime_error(
        "PhononTTS: voice file has no speaker_wavs tensor");
  }
  const nlohmann::json& t = header["speaker_wavs"];
  if (t.value("dtype", "") != "F32") {
    throw std::runtime_error("PhononTTS: speaker_wavs must be float32");
  }
  const std::vector<int64_t> shape = t["shape"].get<std::vector<int64_t>>();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != expected_dim ||
      shape[2] <= 0) {
    throw std::runtime_error("PhononTTS: speaker_wavs must have shape [1, " +
                             std::to_string(expected_dim) + ", frames]");
  }
  const std::vector<uint64_t> offsets =
      t["data_offsets"].get<std::vector<uint64_t>>();
  const int frames = static_cast<int>(shape[2]);
  const size_t count =
      static_cast<size_t>(frames) * static_cast<size_t>(expected_dim);
  const size_t data_start = 8 + static_cast<size_t>(header_len);
  if (offsets.size() != 2 || offsets[1] - offsets[0] != count * sizeof(float) ||
      data_start + offsets[1] > bytes.size()) {
    throw std::runtime_error("PhononTTS: speaker_wavs data is truncated");
  }
  // Stored [dim, frames]; the prompt graph wants [frames, dim].
  std::vector<float> transposed(count);
  const uint8_t* src = bytes.data() + data_start + offsets[0];
  for (int d = 0; d < expected_dim; ++d) {
    for (int f = 0; f < frames; ++f) {
      float v = 0.F;
      std::memcpy(&v,
                  src + (static_cast<size_t>(d) * static_cast<size_t>(frames) +
                         static_cast<size_t>(f)) *
                            sizeof(float),
                  sizeof(float));
      transposed[static_cast<size_t>(f) * static_cast<size_t>(expected_dim) +
                 static_cast<size_t>(d)] = v;
    }
  }
  *out_frames = frames;
  return transposed;
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path& p,
                                     const char* what) {
  if (!std::filesystem::is_regular_file(p)) {
    throw std::runtime_error(std::string("PhononTTS: missing ") + what + " " +
                             p.string());
  }
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    throw std::runtime_error("PhononTTS: cannot open " + p.string());
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

/// ``speed`` as the model's duration conditioning. Measured on the shipped
/// checkpoint, a delta of 0.3 lengthens speech by about a tenth and -0.3
/// shortens it by about an eighth, and the response flattens out past
/// ``|delta| = 1``. A speaking rate of ``speed`` wants a duration of
/// ``1 / speed``, so the delta is the relative change divided by that slope,
/// clamped to the range where it still does something.
float duration_delta_for_speed(double speed) {
  constexpr double kSlope = 0.4;
  const double relative = 1.0 / speed - 1.0;
  const double delta = relative / kSlope;
  return static_cast<float>(std::max(-1.5, std::min(1.5, delta)));
}

std::vector<const char*> c_names(const std::vector<std::string>& names) {
  std::vector<const char*> out;
  out.reserve(names.size());
  for (const std::string& n : names) {
    out.push_back(n.c_str());
  }
  return out;
}

/// A ``[1, heads, 0, head_dim]`` cache: the first step of a sequence.
Ort::Value empty_kv(OrtAllocator* allocator, int heads, int head_dim) {
  const std::array<int64_t, 4> shape{1, heads, 0, head_dim};
  return Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
}

/// Copy a ``[1, heads, P, head_dim]`` tensor's last ``keep`` positions into a
/// fresh tensor owned by ``allocator``.
Ort::Value trim_kv(const Ort::Value& kv, int keep, OrtAllocator* allocator) {
  const std::vector<int64_t> shape = kv.GetTensorTypeAndShapeInfo().GetShape();
  const int64_t heads = shape[1];
  const int64_t positions = shape[2];
  const int64_t dim = shape[3];
  const int64_t kept = std::min<int64_t>(keep, positions);
  const std::array<int64_t, 4> out_shape{1, heads, kept, dim};
  Ort::Value out = Ort::Value::CreateTensor<float>(allocator, out_shape.data(),
                                                   out_shape.size());
  const float* src = kv.GetTensorData<float>();
  float* dst = out.GetTensorMutableData<float>();
  for (int64_t h = 0; h < heads; ++h) {
    const float* from = src + (h * positions + (positions - kept)) * dim;
    std::copy(from, from + kept * dim, dst + h * kept * dim);
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct PhononTTS::Impl {
  double speed_ = 1.0;
  float duration_delta_ = 0.F;
  float temperature_ = 0.3F;
  unsigned int seed_ = 0U;
  bool normalize_audio_ = false;
  float output_volume_ = 1.F;
  bool normalize_text_ = true;

  MoonshineG2POptions g2p_opt_{};
  FileInformationMap tts_files_{};
  std::vector<std::string> ort_provider_names_{};
  std::string coreml_cache_dir_{};
  PhononMeta meta_{};
  mutable std::unique_ptr<BinTokenizer> tokenizer_{};

  Ort::Env env_ = make_ort_env(ORT_LOGGING_LEVEL_WARNING, "moonshine_phonon");
  Ort::MemoryInfo mem_{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  Ort::AllocatorWithDefaultOptions allocator_{};
  Ort::Session embed_prompt_{nullptr};
  Ort::Session embed_latent_{nullptr};
  Ort::Session backbone_{nullptr};
  Ort::Session flow_head_{nullptr};
  Ort::Session mimi_transformer_{nullptr};
  Ort::Session mimi_seanet_{nullptr};
  std::vector<std::string> ep_in_, ep_out_, el_in_, el_out_, bb_in_, bb_out_,
      fh_in_, fh_out_, mt_in_, mt_out_, ms_in_, ms_out_;

  // The voice prompt, frame-major [voice_frames_, audio_prompt_dim].
  std::vector<float> voice_latents_{};
  int voice_frames_ = 0;

  ~Impl() {
    for (auto& e : tts_files_.entries) {
      e.second.free();
    }
  }

  std::filesystem::path asset_path(std::string_view key) const {
    const std::string k(key);
    const auto it = tts_files_.entries.find(k);
    if (it != tts_files_.entries.end() && !it->second.path.empty()) {
      return resolve_path_under_root(g2p_opt_.g2p_root, it->second.path);
    }
    return resolve_path_under_root(g2p_opt_.g2p_root, std::filesystem::path(k));
  }

  bool asset_in_memory(std::string_view key) const {
    const auto it = tts_files_.entries.find(std::string(key));
    return it != tts_files_.entries.end() && it->second.memory != nullptr &&
           it->second.memory_size > 0;
  }

  Ort::Session load_session(std::string_view key) {
    Ort::SessionOptions opts =
        make_ort_session_options(ort_provider_names_, coreml_cache_dir_);
    // Six sessions take turns every 80 ms frame, two of them concurrently.
    // ORT's workers spin for a while after each run by default, so each
    // session's idle pool was stealing cores from the one running: turning
    // that off cuts the frame time by more than half on an M4 Max and is what
    // the Windows hybrid-core policy in ort_configure_cpu_threading does too.
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    const std::string k(key);
    if (asset_in_memory(key)) {
      auto& entry = tts_files_.entries.at(k);
      const uint8_t* b = nullptr;
      size_t n = 0;
      entry.load(&b, &n);
      require_ort_model_bytes(b, n, "Phonon model supplied as " + k);
      Ort::Session s(env_, b, n, opts);
      entry.free();
      return s;
    }
    const std::filesystem::path p = asset_path(key);
    require_ort_model_path(p, "Phonon model file");
    if (!std::filesystem::is_regular_file(p)) {
      throw std::runtime_error("PhononTTS: missing model file " + p.string());
    }
#ifdef _WIN32
    const std::wstring w = p.wstring();
    return Ort::Session(env_, w.c_str(), opts);
#else
    const std::string u8 = p.string();
    return Ort::Session(env_, u8.c_str(), opts);
#endif
  }

  std::vector<uint8_t> load_asset_bytes(std::string_view key,
                                        const char* what) {
    if (asset_in_memory(key)) {
      auto& entry = tts_files_.entries.at(std::string(key));
      const uint8_t* b = nullptr;
      size_t n = 0;
      entry.load(&b, &n);
      std::vector<uint8_t> out(b, b + n);
      entry.free();
      return out;
    }
    return read_file_bytes(asset_path(key), what);
  }

  explicit Impl(const PhononTTSOptions& opt) {
    if (!(opt.speed > 0.0) || !std::isfinite(opt.speed)) {
      throw std::runtime_error(
          "PhononTTS: speed must be a positive finite number");
    }
    g2p_opt_ = opt.g2p_options;
    if (g2p_opt_.g2p_root.empty()) {
      g2p_opt_.g2p_root = std::filesystem::current_path();
    }
    tts_files_ = opt.tts_asset_files;
    ort_provider_names_ = opt.ort_provider_names;
    coreml_cache_dir_ = opt.coreml_cache_dir;
    normalize_audio_ = opt.normalize_audio;
    output_volume_ = opt.output_volume;
    normalize_text_ = opt.normalize_text;
    seed_ = opt.seed;

    meta_ = parse_meta(load_asset_bytes(kTtsPhononModelJsonKey, "model.json"));
    temperature_ = opt.temperature >= 0.F ? opt.temperature : meta_.temperature;
    speed_ = opt.speed;
    duration_delta_ =
        opt.duration_delta != PhononTTSOptions::kUnsetDurationDelta
            ? opt.duration_delta
            : duration_delta_for_speed(speed_);

    {
      const std::vector<uint8_t> tok =
          load_asset_bytes(kTtsPhononTokenizerKey, "tokenizer.bin");
      tokenizer_ = std::make_unique<BinTokenizer>(tok.data(), tok.size());
      if (tokenizer_->encoding_for_test() != BinTokenizerEncoding::kUnigram) {
        throw std::runtime_error(
            "PhononTTS: tokenizer.bin must be a Unigram vocabulary with "
            "scores (regenerate it with scripts/convert_tokenizer.py)");
      }
    }

    embed_prompt_ = load_session(kTtsPhononEmbedPromptKey);
    embed_latent_ = load_session(kTtsPhononEmbedLatentKey);
    backbone_ = load_session(kTtsPhononBackboneKey);
    flow_head_ = load_session(kTtsPhononFlowHeadKey);
    mimi_transformer_ = load_session(kTtsPhononMimiTransformerKey);
    mimi_seanet_ = load_session(kTtsPhononMimiSeanetKey);
    ep_in_ = embed_prompt_.GetInputNames();
    ep_out_ = embed_prompt_.GetOutputNames();
    el_in_ = embed_latent_.GetInputNames();
    el_out_ = embed_latent_.GetOutputNames();
    bb_in_ = backbone_.GetInputNames();
    bb_out_ = backbone_.GetOutputNames();
    fh_in_ = flow_head_.GetInputNames();
    fh_out_ = flow_head_.GetOutputNames();
    mt_in_ = mimi_transformer_.GetInputNames();
    mt_out_ = mimi_transformer_.GetOutputNames();
    ms_in_ = mimi_seanet_.GetInputNames();
    ms_out_ = mimi_seanet_.GetOutputNames();
    const size_t kv = 2 * static_cast<size_t>(meta_.num_layers);
    const size_t mimi_kv = 2 * static_cast<size_t>(meta_.mimi_layers);
    if (ep_in_.size() != 3 || el_in_.size() != 2 || bb_in_.size() != 1 + kv ||
        bb_out_.size() != 2 + kv || fh_in_.size() != 4 ||
        mt_in_.size() != 3 + mimi_kv || mt_out_.size() != 2 + mimi_kv ||
        ms_in_.size() != 1) {
      throw std::runtime_error(
          "PhononTTS: unexpected ONNX signature; the graphs do not match "
          "model.json");
    }

    // The voice.
    if (!opt.voice_latents.empty()) {
      if (opt.voice_frames <= 0 ||
          opt.voice_latents.size() !=
              static_cast<size_t>(opt.voice_frames) *
                  static_cast<size_t>(meta_.audio_prompt_dim)) {
        throw std::runtime_error(
            "PhononTTS: voice_latents must hold voice_frames * " +
            std::to_string(meta_.audio_prompt_dim) + " floats");
      }
      voice_latents_ = opt.voice_latents;
      voice_frames_ = opt.voice_frames;
    } else if (asset_in_memory(kTtsPhononVoiceFileKey) ||
               (opt.voice_id.empty() && tts_files_.entries.count(std::string(
                                            kTtsPhononVoiceFileKey)) != 0)) {
      voice_latents_ = parse_voice_safetensors(
          load_asset_bytes(kTtsPhononVoiceFileKey, "voice file"),
          meta_.audio_prompt_dim, &voice_frames_);
    } else {
      if (opt.voice_id.empty()) {
        throw std::runtime_error(
            "PhononTTS: no voice supplied (set a catalog voice id or a voice "
            "file)");
      }
      const std::vector<PhononVoiceInfo> voices =
          list_voices(g2p_opt_.g2p_root, tts_files_);
      const PhononVoiceInfo* match = nullptr;
      for (const PhononVoiceInfo& v : voices) {
        if (v.id == opt.voice_id) {
          match = &v;
          break;
        }
      }
      // The latent is addressed by its canonical asset key so that a caller
      // who supplied it in memory (the WASM binding has no filesystem) is
      // honoured; on disk the key resolves under the data root as before.
      std::string voice_key;
      if (match != nullptr) {
        // ``file`` is relative to voices.json, e.g. "voices/zoey.safetensors".
        voice_key = std::string(kTtsPhononVoicesJsonKey.substr(
                        0, kTtsPhononVoicesJsonKey.rfind('/') + 1)) +
                    match->file;
      } else {
        // Not in the catalog: try the voices directory by name.
        voice_key = std::string(kTtsPhononVoicesDirKey) + "/" + opt.voice_id +
                    ".safetensors";
      }
      if (!asset_in_memory(voice_key) &&
          !std::filesystem::is_regular_file(asset_path(voice_key))) {
        if (match == nullptr) {
          throw std::runtime_error("PhononTTS: unknown voice '" + opt.voice_id +
                                   "'");
        }
        throw std::runtime_error("PhononTTS: missing voice file " +
                                 asset_path(voice_key).string());
      }
      voice_latents_ =
          parse_voice_safetensors(load_asset_bytes(voice_key, "voice file"),
                                  meta_.audio_prompt_dim, &voice_frames_);
    }
  }

  int count_tokens(std::string_view text) const {
    return static_cast<int>(
        tokenizer_->text_to_tokens<int64_t>(std::string(text)).size());
  }

  std::vector<std::string> plan_prompts(std::string_view text) const {
    const std::string normalized =
        normalize_text_ ? phonon_normalize_text(text) : std::string(text);
    constexpr int kMaxTokensPerChunk = 50;
    constexpr int kMaxFitTokens = 200;
    return phonon_plan_chunks(
        normalized, [this](std::string_view t) { return count_tokens(t); },
        kMaxTokensPerChunk, kMaxFitTokens);
  }

  Ort::Value run_one(Ort::Session& session, const std::vector<std::string>& in,
                     const std::vector<std::string>& out,
                     std::vector<Ort::Value>& inputs) {
    const std::vector<const char*> in_names = c_names(in);
    const char* out_name = out[0].c_str();
    Ort::RunOptions run_opts{nullptr};
    std::vector<Ort::Value> outputs = session.Run(
        run_opts, in_names.data(), inputs.data(), inputs.size(), &out_name, 1);
    return std::move(outputs[0]);
  }
};

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

namespace {

/// The LM half of each frame runs beside the Mimi half on its own thread.
/// Under Emscripten it runs inline instead: the engine may be on the browser's
/// main thread, which cannot block on a worker.
#if defined(__EMSCRIPTEN__)
constexpr std::launch kPipelineLaunch = std::launch::deferred;
#else
constexpr std::launch kPipelineLaunch = std::launch::async;
#endif

}  // namespace

struct PhononTTS::Generation::State {
  Impl* impl = nullptr;
  float duration_delta = 0.F;
  float noise_scale = 0.F;
  std::mt19937 rng;
  std::normal_distribution<float> gauss{0.F, 1.F};

  // Backbone state: the condition for the frame about to be sampled, the EOS
  // logit that came with it, and the KV cache behind them.
  std::vector<float> condition{};
  float eos_logit = 0.F;
  std::vector<Ort::Value> past{};

  // The latent for the frame about to be decoded, once sampled.
  std::vector<float> latent{};
  bool have_latent = false;

  // Where the loop is.
  int frames = 0;
  int max_frames = 0;
  int remaining_after_eos = -1;
  int frames_after_eos = 1;
  bool done = false;

  // Mimi state.
  std::vector<Ort::Value> mimi_past{};
  std::vector<float> prev_hidden{};
  int64_t position = 0;
  /// Transformer output of the last context frames, [mimi_hidden, steps].
  std::vector<float> history{};
  int history_steps = 0;

  /// Feed the backbone one step of embeddings [1, S, hidden] and keep what
  /// comes back.
  void step_backbone(std::vector<float>& embeds, int steps) {
    const PhononMeta& m = impl->meta_;
    const std::array<int64_t, 3> shape{1, steps, m.hidden_size};
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + past.size());
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, embeds.data(), embeds.size(), shape.data(), shape.size()));
    for (Ort::Value& p : past) {
      inputs.push_back(std::move(p));
    }
    const std::vector<const char*> in_names = c_names(impl->bb_in_);
    const std::vector<const char*> out_names = c_names(impl->bb_out_);
    Ort::RunOptions run_opts{nullptr};
    std::vector<Ort::Value> outputs =
        impl->backbone_.Run(run_opts, in_names.data(), inputs.data(),
                            inputs.size(), out_names.data(), out_names.size());
    const float* c = outputs[0].GetTensorData<float>();
    condition.assign(c, c + m.hidden_size);
    eos_logit = outputs[1].GetTensorData<float>()[0];
    past.clear();
    for (size_t i = 2; i < outputs.size(); ++i) {
      past.push_back(std::move(outputs[i]));
    }
  }

  /// One flow step from noise to a latent, conditioned on the backbone.
  std::vector<float> sample_latent() {
    const PhononMeta& m = impl->meta_;
    std::vector<float> noise(static_cast<size_t>(m.latent_dim), 0.F);
    if (noise_scale > 0.F) {
      for (float& v : noise) {
        v = gauss(rng) * noise_scale;
      }
    }
    const std::array<int64_t, 2> latent_shape{1, m.latent_dim};
    const std::array<int64_t, 2> cond_shape{1, m.hidden_size};
    const std::array<int64_t, 1> scalar_shape{1};
    float t_start = 0.F;
    float t_end = 1.F;
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, noise.data(), noise.size(), latent_shape.data(),
        latent_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, condition.data(), condition.size(), cond_shape.data(),
        cond_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, &t_start, 1, scalar_shape.data(), scalar_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, &t_end, 1, scalar_shape.data(), scalar_shape.size()));
    Ort::Value velocity =
        impl->run_one(impl->flow_head_, impl->fh_in_, impl->fh_out_, inputs);
    const float* v = velocity.GetTensorData<float>();
    std::vector<float> latent(static_cast<size_t>(m.latent_dim));
    for (int i = 0; i < m.latent_dim; ++i) {
      latent[static_cast<size_t>(i)] =
          noise[static_cast<size_t>(i)] + v[static_cast<size_t>(i)];
    }
    return latent;
  }

  std::vector<float> embed_latent(std::vector<float>& latent) {
    const PhononMeta& m = impl->meta_;
    const std::array<int64_t, 2> latent_shape{1, m.latent_dim};
    const std::array<int64_t, 1> scalar_shape{1};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, latent.data(), latent.size(), latent_shape.data(),
        latent_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, &duration_delta, 1, scalar_shape.data(),
        scalar_shape.size()));
    Ort::Value out =
        impl->run_one(impl->embed_latent_, impl->el_in_, impl->el_out_, inputs);
    const float* p = out.GetTensorData<float>();
    return std::vector<float>(p, p + m.hidden_size);
  }

  /// Decode one latent frame to ``frame_size`` samples, exactly as a whole
  /// decode would: the transformer keeps a sliding-window KV cache and the
  /// one-frame overlap of its upsampling convolution, and the convolutional
  /// half is re-run over a frame of context whose samples are then dropped.
  std::vector<float> decode_frame(const std::vector<float>& latent) {
    const PhononMeta& m = impl->meta_;
    const int stride = m.mimi_stride;
    const int hidden = m.mimi_hidden;
    const int hop = m.frame_size / stride;

    const std::array<int64_t, 3> latent_shape{1, 1, m.latent_dim};
    const std::array<int64_t, 3> prev_shape{1, hidden, 1};
    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + mimi_past.size());
    std::vector<float> latent_in = latent;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, latent_in.data(), latent_in.size(), latent_shape.data(),
        latent_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, prev_hidden.data(), prev_hidden.size(), prev_shape.data(),
        prev_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(impl->mem_, &position, 1,
                                                       nullptr, 0));
    for (Ort::Value& p : mimi_past) {
      inputs.push_back(std::move(p));
    }
    const std::vector<const char*> in_names = c_names(impl->mt_in_);
    const std::vector<const char*> out_names = c_names(impl->mt_out_);
    Ort::RunOptions run_opts{nullptr};
    std::vector<Ort::Value> outputs = impl->mimi_transformer_.Run(
        run_opts, in_names.data(), inputs.data(), inputs.size(),
        out_names.data(), out_names.size());
    // hidden [1, hidden, stride]
    const float* h = outputs[0].GetTensorData<float>();
    const float* np = outputs[1].GetTensorData<float>();
    prev_hidden.assign(np, np + hidden);
    const int keep = m.mimi_window - 1;
    mimi_past.clear();
    for (size_t i = 2; i < outputs.size(); ++i) {
      const int64_t positions =
          outputs[i].GetTensorTypeAndShapeInfo().GetShape()[2];
      if (positions > keep) {
        mimi_past.push_back(trim_kv(outputs[i], keep, impl->allocator_));
      } else {
        mimi_past.push_back(std::move(outputs[i]));
      }
    }
    position += stride;

    // SEANet over [history | hidden], dropping the history's samples.
    const int ctx_steps = history_steps;
    const int total_steps = ctx_steps + stride;
    std::vector<float> window(static_cast<size_t>(hidden) *
                              static_cast<size_t>(total_steps));
    for (int c = 0; c < hidden; ++c) {
      float* row = window.data() +
                   static_cast<size_t>(c) * static_cast<size_t>(total_steps);
      if (ctx_steps > 0) {
        std::copy(history.data() +
                      static_cast<size_t>(c) * static_cast<size_t>(ctx_steps),
                  history.data() + static_cast<size_t>(c + 1) *
                                       static_cast<size_t>(ctx_steps),
                  row);
      }
      std::copy(h + static_cast<size_t>(c) * static_cast<size_t>(stride),
                h + static_cast<size_t>(c + 1) * static_cast<size_t>(stride),
                row + ctx_steps);
    }
    const std::array<int64_t, 3> window_shape{1, hidden, total_steps};
    std::vector<Ort::Value> se_inputs;
    se_inputs.push_back(Ort::Value::CreateTensor<float>(
        impl->mem_, window.data(), window.size(), window_shape.data(),
        window_shape.size()));
    Ort::Value audio = impl->run_one(impl->mimi_seanet_, impl->ms_in_,
                                     impl->ms_out_, se_inputs);
    const size_t n = audio.GetTensorTypeAndShapeInfo().GetElementCount();
    const size_t skip =
        static_cast<size_t>(ctx_steps) * static_cast<size_t>(hop);
    if (n < skip + static_cast<size_t>(m.frame_size)) {
      throw std::runtime_error("PhononTTS: Mimi returned too few samples");
    }
    const float* a = audio.GetTensorData<float>();
    std::vector<float> out(a + skip, a + skip + m.frame_size);

    // Keep the last context frames of transformer output for the next call.
    const int max_ctx = m.mimi_context_frames * stride;
    const int new_steps = std::min(total_steps, max_ctx);
    std::vector<float> next_history(static_cast<size_t>(hidden) *
                                    static_cast<size_t>(new_steps));
    for (int c = 0; c < hidden; ++c) {
      const float* row =
          window.data() +
          static_cast<size_t>(c) * static_cast<size_t>(total_steps) +
          static_cast<size_t>(total_steps - new_steps);
      std::copy(row, row + new_steps,
                next_history.data() +
                    static_cast<size_t>(c) * static_cast<size_t>(new_steps));
    }
    history = std::move(next_history);
    history_steps = new_steps;
    return out;
  }

  /// The flow LM for the frame after ``current``: embed it, step the
  /// backbone, sample the next latent.
  std::vector<float> advance(std::vector<float> current) {
    std::vector<float> embeds = embed_latent(current);
    step_backbone(embeds, 1);
    return sample_latent();
  }

  /// One frame of raw audio, advancing the loop; empty once done.
  ///
  /// The LM for the next frame needs only this frame's latent, not its
  /// audio, so it runs on a second thread while Mimi decodes and a frame
  /// costs the longer stage rather than the sum. The two halves use disjoint
  /// sessions and touch disjoint members of this state (the LM half:
  /// ``condition``, ``eos_logit``, ``past``, ``rng``; the Mimi half: the
  /// ``mimi_*``, ``prev_hidden``, ``position`` and ``history`` fields).
  std::vector<float> raw_next_frame() {
    if (done) {
      return {};
    }
    const PhononMeta& m = impl->meta_;
    if (!have_latent) {
      latent = sample_latent();
      have_latent = true;
    }
    const std::vector<float> current = std::move(latent);
    have_latent = false;

    // Whether another frame follows is known before decoding: ``eos_logit``
    // came with the condition this latent was sampled from.
    ++frames;
    if (remaining_after_eos < 0 && frames > m.min_frames_before_eos &&
        eos_logit > m.eos_threshold) {
      remaining_after_eos = frames_after_eos;
    }
    const bool more = remaining_after_eos != 0 && frames < max_frames;
    if (more && remaining_after_eos > 0) {
      --remaining_after_eos;
    }

    std::future<std::vector<float>> next;
    if (more) {
      next = std::async(kPipelineLaunch,
                        [this, current]() { return advance(current); });
    }
    std::vector<float> audio = decode_frame(current);
    if (more) {
      latent = next.get();
      have_latent = true;
    } else {
      done = true;
    }
    return audio;
  }
};

PhononTTS::Generation::Generation(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
PhononTTS::Generation::~Generation() = default;

bool PhononTTS::Generation::done() const { return state_->done; }

std::vector<float> PhononTTS::Generation::next_frame() {
  std::vector<float> audio = state_->raw_next_frame();
  if (!audio.empty()) {
    apply_synthesis_output_effects(audio, /*normalize_audio=*/false,
                                   state_->impl->output_volume_);
  }
  return audio;
}

std::unique_ptr<PhononTTS::Generation> PhononTTS::begin(
    std::string_view prompt) {
  Impl& im = *impl_;
  const PhononMeta& m = im.meta_;
  const PhononPrompt prepared = phonon_prepare_prompt(prompt);
  auto state = std::make_unique<Generation::State>();
  state->impl = &im;
  state->duration_delta = im.duration_delta_;
  state->noise_scale = im.temperature_ > 0.F ? std::sqrt(im.temperature_) : 0.F;
  state->rng.seed(im.seed_);
  if (prepared.text.empty()) {
    state->done = true;
    return std::unique_ptr<Generation>(new Generation(std::move(state)));
  }
  std::vector<int64_t> text_ids =
      im.tokenizer_->text_to_tokens<int64_t>(prepared.text);
  if (text_ids.empty()) {
    state->done = true;
    return std::unique_ptr<Generation>(new Generation(std::move(state)));
  }
  state->frames_after_eos = prepared.words <= m.short_text_max_words
                                ? m.frames_after_eos_short
                                : m.frames_after_eos;
  state->max_frames = static_cast<int>(std::ceil(
      (static_cast<double>(text_ids.size()) / 3.0 + 2.0) * m.frame_rate));

  // Prompt embeddings: [audio_prompt x voice][text][audio_bos].
  const std::array<int64_t, 2> ids_shape{1,
                                         static_cast<int64_t>(text_ids.size())};
  const std::array<int64_t, 3> voice_shape{1, im.voice_frames_,
                                           m.audio_prompt_dim};
  const std::array<int64_t, 1> scalar_shape{1};
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      im.mem_, text_ids.data(), text_ids.size(), ids_shape.data(),
      ids_shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<float>(
      im.mem_, im.voice_latents_.data(), im.voice_latents_.size(),
      voice_shape.data(), voice_shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<float>(
      im.mem_, &state->duration_delta, 1, scalar_shape.data(),
      scalar_shape.size()));
  Ort::Value prompt_embeds =
      im.run_one(im.embed_prompt_, im.ep_in_, im.ep_out_, inputs);
  const std::vector<int64_t> pe_shape =
      prompt_embeds.GetTensorTypeAndShapeInfo().GetShape();
  const int steps = static_cast<int>(pe_shape[1]);
  const float* pe = prompt_embeds.GetTensorData<float>();
  std::vector<float> embeds(
      pe, pe + static_cast<size_t>(steps) * static_cast<size_t>(m.hidden_size));

  // Prefill with an empty cache.
  state->past.clear();
  for (int i = 0; i < 2 * m.num_layers; ++i) {
    state->past.push_back(empty_kv(im.allocator_, m.num_heads, m.head_dim));
  }
  state->step_backbone(embeds, steps);

  // Fresh Mimi state.
  state->mimi_past.clear();
  for (int i = 0; i < 2 * m.mimi_layers; ++i) {
    state->mimi_past.push_back(
        empty_kv(im.allocator_, m.mimi_heads, m.mimi_head_dim));
  }
  state->prev_hidden.assign(static_cast<size_t>(m.mimi_hidden), 0.F);
  state->position = 0;
  state->history.clear();
  state->history_steps = 0;
  return std::unique_ptr<Generation>(new Generation(std::move(state)));
}

// ---------------------------------------------------------------------------
// PhononTTS
// ---------------------------------------------------------------------------

PhononTTS::PhononTTS(const PhononTTSOptions& opt)
    : impl_(std::make_unique<Impl>(opt)) {}
PhononTTS::~PhononTTS() = default;
PhononTTS::PhononTTS(PhononTTS&&) noexcept = default;
PhononTTS& PhononTTS::operator=(PhononTTS&&) noexcept = default;

void PhononTTS::set_speed(double speed) {
  if (!(speed > 0.0) || !std::isfinite(speed)) {
    throw std::runtime_error(
        "PhononTTS: speed must be a positive finite number");
  }
  impl_->speed_ = speed;
  impl_->duration_delta_ = duration_delta_for_speed(speed);
}
double PhononTTS::speed() const { return impl_->speed_; }
void PhononTTS::set_duration_delta(float delta) {
  if (!std::isfinite(delta)) {
    throw std::runtime_error("PhononTTS: duration_delta must be finite");
  }
  impl_->duration_delta_ = delta;
}
float PhononTTS::duration_delta() const { return impl_->duration_delta_; }
void PhononTTS::set_temperature(float temperature) {
  if (!(temperature >= 0.F) || !std::isfinite(temperature)) {
    throw std::runtime_error(
        "PhononTTS: temperature must be a non-negative finite number");
  }
  impl_->temperature_ = temperature;
}
float PhononTTS::temperature() const { return impl_->temperature_; }
void PhononTTS::set_seed(unsigned int seed) { impl_->seed_ = seed; }
unsigned int PhononTTS::seed() const { return impl_->seed_; }
bool PhononTTS::normalize_audio() const { return impl_->normalize_audio_; }
void PhononTTS::set_normalize_audio(bool on) { impl_->normalize_audio_ = on; }
float PhononTTS::output_volume() const { return impl_->output_volume_; }
void PhononTTS::set_output_volume(float volume) {
  impl_->output_volume_ = volume;
}

std::vector<std::string> PhononTTS::plan_prompts(std::string_view text) const {
  return impl_->plan_prompts(text);
}

std::vector<float> PhononTTS::synthesize(std::string_view text) {
  std::vector<float> out;
  for (const std::string& prompt : impl_->plan_prompts(text)) {
    std::unique_ptr<Generation> gen = begin(prompt);
    while (!gen->done()) {
      std::vector<float> frame = gen->state_->raw_next_frame();
      out.insert(out.end(), frame.begin(), frame.end());
    }
  }
  apply_synthesis_output_effects(out, impl_->normalize_audio_,
                                 impl_->output_volume_);
  return out;
}

// ---------------------------------------------------------------------------
// PhononChunkSource
// ---------------------------------------------------------------------------

PhononChunkSource::PhononChunkSource(PhononTTS& tts, ChunkPolicyOptions policy)
    : tts_(tts), policy_(policy) {}

void PhononChunkSource::begin(std::string_view text) {
  prompts_ = tts_.plan_prompts(text);
  next_prompt_ = 0;
  gen_.reset();
  chunks_emitted_ = 0;
  advance();
}

void PhononChunkSource::advance() {
  while (!gen_ || gen_->done()) {
    if (next_prompt_ >= prompts_.size()) {
      gen_.reset();
      return;
    }
    gen_ = tts_.begin(prompts_[next_prompt_++]);
  }
}

std::vector<float> PhononChunkSource::next() {
  if (!gen_) {
    return {};
  }
  // The schedule: the first chunk sets time to first audio, later ones grow.
  const double frame_seconds =
      static_cast<double>(PhononTTS::kFrameSamples) / PhononTTS::kSampleRateHz;
  double seconds = std::max(0.F, policy_.first_chunk_seconds);
  for (int i = 0; i < chunks_emitted_; ++i) {
    seconds *= std::max(1.F, policy_.growth);
  }
  const int target_frames =
      std::max(1, static_cast<int>(std::ceil(seconds / frame_seconds)));
  std::vector<float> out;
  out.reserve(static_cast<size_t>(target_frames) *
              static_cast<size_t>(PhononTTS::kFrameSamples));
  int frames = 0;
  while (frames < target_frames && !gen_->done()) {
    const std::vector<float> frame = gen_->next_frame();
    out.insert(out.end(), frame.begin(), frame.end());
    ++frames;
  }
  ++chunks_emitted_;
  // Keep `has_more` exact: step to the next prompt now, so a prompt that has
  // nothing to say is never what the stream is waiting on.
  advance();
  return out;
}

bool PhononChunkSource::has_more() const { return gen_ != nullptr; }

int PhononChunkSource::sample_rate() const { return PhononTTS::kSampleRateHz; }

const std::vector<std::string>& PhononTTS::builtin_voice_ids() {
  // Mirrors core/moonshine-tts/data/phonon/voices.json; phonon_tts_test
  // checks the two agree when the data directory is present.
  static const std::vector<std::string> ids = {
      "alfie",
      "azgorath",
      "bryony",
      "captain_silas_blackwater",
      "delia",
      "easton",
      "edwin",
      "elodie-rose",
      "everly",
      "fairy_godmother",
      "fennick_the_young_elf",
      "freya",
      "grub_the_ogre",
      "hadley",
      "harper",
      "hugh",
      "imogen",
      "juniper",
      "lottie",
      "maeve",
      "master_aldric",
      "micah",
      "miles",
      "morwenna_of_the_deep",
      "old_mortlake",
      "pippa",
      "preston",
      "russell",
      "sir_bramwell",
      "spencer",
      "teddy",
      "the_snow_queen",
      "the_thing_under_the_stairs",
      "toby",
      "trevor",
      "vyrmathrax_the_ancient",
      "zoey",
  };
  return ids;
}

std::vector<PhononVoiceInfo> PhononTTS::list_voices(
    const std::filesystem::path& root, const FileInformationMap& files) {
  std::vector<uint8_t> bytes;
  const std::string key(kTtsPhononVoicesJsonKey);
  const auto it = files.entries.find(key);
  if (it != files.entries.end() && it->second.memory != nullptr &&
      it->second.memory_size > 0) {
    bytes.assign(it->second.memory, it->second.memory + it->second.memory_size);
  } else {
    const std::filesystem::path p =
        (it != files.entries.end() && !it->second.path.empty())
            ? resolve_path_under_root(root, it->second.path)
            : resolve_path_under_root(root, std::filesystem::path(key));
    if (!std::filesystem::is_regular_file(p)) {
      return {};
    }
    bytes = read_file_bytes(p, "voices.json");
  }
  std::vector<PhononVoiceInfo> out;
  try {
    nlohmann::json j = nlohmann::json::parse(bytes.begin(), bytes.end());
    const nlohmann::json& rows = j.contains("voices") ? j["voices"] : j;
    // Catalog fields can be null (characters have no gender), so this cannot
    // be ``value()``, which would throw on them.
    const auto str_or = [](const nlohmann::json& row, const char* key) {
      return row.contains(key) && row[key].is_string()
                 ? row[key].get<std::string>()
                 : std::string();
    };
    for (const nlohmann::json& row : rows) {
      PhononVoiceInfo v;
      v.id = str_or(row, "id");
      v.name = str_or(row, "name");
      v.locale = str_or(row, "locale");
      v.gender = str_or(row, "gender");
      v.description = str_or(row, "description");
      v.file = str_or(row, "file");
      if (!v.id.empty() && !v.file.empty()) {
        out.push_back(std::move(v));
      }
    }
  } catch (const std::exception&) {
    return {};
  }
  return out;
}

}  // namespace moonshine_tts
