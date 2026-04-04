#include "onnx-g2p-models.h"

#include "constants.h"
#include "utf8-utils.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <nlohmann/json.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

namespace {

Ort::SessionOptions make_session_options(bool use_cuda) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  (void)use_cuda;
  // Optional CUDA: link a GPU ORT build and append execution provider here if needed.
  return session_options;
}

Ort::Session open_session(Ort::Env& env, const std::filesystem::path& model_path, bool use_cuda) {
#ifdef _WIN32
  const std::wstring w = model_path.wstring();
  return Ort::Session(env, w.c_str(), make_session_options(use_cuda));
#else
  const std::string u8 = model_path.string();
  return Ort::Session(env, u8.c_str(), make_session_options(use_cuda));
#endif
}

Ort::Session open_session_memory(Ort::Env& env, const void* data, size_t len, bool use_cuda) {
  return Ort::Session(env, data, len, make_session_options(use_cuda));
}

std::vector<int64_t> encode_chars_for_model(
    const std::string& text,
    const std::unordered_map<std::string, int64_t>& char_stoi) {
  const int64_t unk = char_stoi.at(std::string(kSpecialUnk));
  std::vector<int64_t> ids;
  for (const auto& ch : utf8_split_codepoints(text)) {
    const auto it = char_stoi.find(ch);
    ids.push_back(it != char_stoi.end() ? it->second : unk);
  }
  return ids;
}

void decoder_io_padded(const std::vector<int64_t>& cur,
                       int max_phoneme_len,
                       int64_t pad_token_id,
                       std::vector<int64_t>& dec_row,
                       std::vector<int64_t>& dec_mask,
                       int& L) {
  L = static_cast<int>(cur.size());
  if (L > max_phoneme_len) {
    throw std::runtime_error("decoder length > max_phoneme_len");
  }
  dec_row = cur;
  dec_row.resize(static_cast<size_t>(max_phoneme_len), pad_token_id);
  dec_mask.assign(static_cast<size_t>(max_phoneme_len), 0);
  for (int i = 0; i < L; ++i) {
    dec_mask[static_cast<size_t>(i)] = 1;
  }
}

int argmax_vocab_row(const float* logits, int64_t vocab, int time_index) {
  const size_t base = static_cast<size_t>(time_index) * static_cast<size_t>(vocab);
  int best = 0;
  float best_v = logits[base];
  for (int64_t k = 1; k < vocab; ++k) {
    const float v = logits[base + static_cast<size_t>(k)];
    if (v > best_v) {
      best_v = v;
      best = static_cast<int>(k);
    }
  }
  return best;
}

}  // namespace

OnnxOovG2p::OnnxOovG2p(Ort::Env& env, const std::filesystem::path& model_onnx, bool use_cuda)
    : tab_(load_oov_tables(model_onnx)), session_(open_session(env, model_onnx, use_cuda)) {}

OnnxOovG2p::OnnxOovG2p(Ort::Env& env, const void* model_onnx_bytes, size_t model_onnx_size,
                       const nlohmann::json& onnx_config, bool use_cuda)
    : tab_(load_oov_tables_from_json(onnx_config, "onnx-config (memory)")),
      session_(open_session_memory(env, model_onnx_bytes, model_onnx_size, use_cuda)) {}

std::vector<std::string> OnnxOovG2p::predict_phonemes(const std::string& word) {
  if (word.empty()) {
    return {};
  }
  std::vector<int64_t> ids = encode_chars_for_model(word, tab_.char_stoi);
  if (static_cast<int>(ids.size()) > tab_.max_seq_len) {
    ids.resize(static_cast<size_t>(tab_.max_seq_len));
  }
  const int enc_len = static_cast<int>(ids.size());
  std::vector<int64_t> enc_ids(static_cast<size_t>(tab_.max_seq_len), tab_.pad_id);
  std::vector<int64_t> enc_mask(static_cast<size_t>(tab_.max_seq_len), 0);
  for (int i = 0; i < enc_len; ++i) {
    enc_ids[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)];
    enc_mask[static_cast<size_t>(i)] = 1;
  }

  const std::array<int64_t, 2> enc_shape{1, tab_.max_seq_len};

  std::vector<int64_t> cur;
  cur.push_back(tab_.bos);

  const char* in_names[] = {"encoder_input_ids", "encoder_attention_mask", "decoder_input_ids",
                            "decoder_attention_mask"};
  const char* out_names[] = {"logits"};

  for (int step = 0; step < tab_.max_phoneme_len; ++step) {
    (void)step;
    std::vector<int64_t> dec_row;
    std::vector<int64_t> dec_mask;
    int L = 0;
    decoder_io_padded(cur, tab_.max_phoneme_len, tab_.phon_pad, dec_row, dec_mask, L);

    const std::array<int64_t, 2> dec_shape{1, tab_.max_phoneme_len};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, enc_ids.data(), enc_ids.size(), enc_shape.data(), enc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, enc_mask.data(), enc_mask.size(), enc_shape.data(), enc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, dec_row.data(), dec_row.size(), dec_shape.data(), dec_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, dec_mask.data(), dec_mask.size(), dec_shape.data(), dec_shape.size()));

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(),
                                out_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 3) {
      throw std::runtime_error("unexpected logits rank");
    }
    const int64_t vocab = shape[2];
    const int nxt = argmax_vocab_row(logits, vocab, L - 1);
    if (nxt == static_cast<int>(tab_.eos) || nxt == static_cast<int>(tab_.phon_pad)) {
      break;
    }
    cur.push_back(nxt);
    if (static_cast<int>(cur.size()) >= tab_.max_phoneme_len) {
      break;
    }
  }

  std::vector<std::string> out;
  for (size_t i = 1; i < cur.size(); ++i) {
    const int64_t tid = cur[i];
    if (tid == tab_.eos) {
      break;
    }
    if (tid >= 0 && static_cast<size_t>(tid) < tab_.phoneme_itos.size()) {
      const std::string& tok = tab_.phoneme_itos[static_cast<size_t>(tid)];
      if (tok == kPhonPad || tok == kPhonBos || tok == kPhonEos) {
        continue;
      }
      out.push_back(tok);
    }
  }
  return out;
}

}  // namespace moonshine_tts
