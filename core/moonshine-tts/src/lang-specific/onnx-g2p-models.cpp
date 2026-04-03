#include "onnx-g2p-models.h"

#include "constants.h"
#include "heteronym-context.h"
#include "ipa-postprocess.h"
#include "utf8-utils.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

bool heteronym_debug_env_enabled() {
  const char* v = std::getenv("MOONSHINE_TTS_DEBUG_HET");
  if (v == nullptr || v[0] == '\0') {
    return false;
  }
  const std::string s(v);
  return s != "0" && s != "false" && s != "FALSE";
}

void log_once_onnxruntime_version() {
  static bool done = false;
  if (done || !heteronym_debug_env_enabled()) {
    return;
  }
  done = true;
  std::cerr << "moonshine_tts: heteronym debug: onnxruntime " << Ort::GetVersionString() << '\n';
}

}  // namespace

OnnxOovG2p::OnnxOovG2p(Ort::Env& env, const std::filesystem::path& model_onnx, bool use_cuda)
    : tab_(load_oov_tables(model_onnx)), session_(open_session(env, model_onnx, use_cuda)) {}

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

OnnxHeteronymG2p::OnnxHeteronymG2p(Ort::Env& env, const std::filesystem::path& model_onnx,
                                   bool use_cuda)
    : tab_(load_heteronym_tables(model_onnx)), session_(open_session(env, model_onnx, use_cuda)) {}

std::string OnnxHeteronymG2p::disambiguate_ipa(const std::string& full_text,
                                               int span_s,
                                               int span_e,
                                               const std::string& lookup_key,
                                               const std::vector<std::string>& cmudict_alternatives) {
  const bool dbg = heteronym_debug_env_enabled();
  if (dbg) {
    log_once_onnxruntime_version();
  }

  if (cmudict_alternatives.size() <= 1) {
    if (dbg) {
      std::cerr << "moonshine_tts: heteronym debug: skip (<=1 CMU alt) lookup_key=" << lookup_key
                << " n_alts=" << cmudict_alternatives.size() << '\n';
    }
    return cmudict_alternatives.empty() ? "" : cmudict_alternatives[0];
  }

  std::string gkey;
  if (tab_.group_key == "lower") {
    gkey = lookup_key;
  } else {
    const auto cps = utf8_split_codepoints(full_text);
    for (int i = span_s; i < span_e && i < static_cast<int>(cps.size()); ++i) {
      gkey += cps[static_cast<size_t>(i)];
    }
  }
  if (tab_.ordered_candidates.find(gkey) == tab_.ordered_candidates.end()) {
    if (dbg) {
      std::cerr << "moonshine_tts: heteronym debug: fallback (gkey not in homograph_index) gkey="
                << std::quoted(gkey) << " group_key=" << std::quoted(tab_.group_key)
                << " first_alt=" << std::quoted(cmudict_alternatives[0]) << '\n';
    }
    return cmudict_alternatives[0];
  }

  const auto full_cells = utf8_split_codepoints(full_text);
  const auto win = heteronym_centered_context_window_cells(full_cells, span_s, span_e,
                                                             kHeteronymContextMaxChars);
  if (!win) {
    if (dbg) {
      std::cerr << "moonshine_tts: heteronym debug: fallback (no context window) span=[" << span_s
                << ',' << span_e << ") full_cp=" << full_cells.size() << '\n';
    }
    return cmudict_alternatives[0];
  }
  const auto& [window_text, ws, we] = *win;

  std::vector<int64_t> ids = encode_chars_for_model(window_text, tab_.char_stoi);
  std::vector<float> span(static_cast<size_t>(ids.size()), 0.0F);
  for (int j = ws; j < we && j < static_cast<int>(span.size()); ++j) {
    span[static_cast<size_t>(j)] = 1.0F;
  }

  while (static_cast<int>(ids.size()) < tab_.max_seq_len) {
    ids.push_back(tab_.pad_id);
    span.push_back(0.0F);
  }
  ids.resize(static_cast<size_t>(tab_.max_seq_len));
  span.resize(static_cast<size_t>(tab_.max_seq_len));

  std::vector<int64_t> attn_1d(static_cast<size_t>(tab_.max_seq_len));
  for (int i = 0; i < tab_.max_seq_len; ++i) {
    attn_1d[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)] != tab_.pad_id ? 1 : 0;
  }

  float span_sum = 0.F;
  for (float v : span) {
    span_sum += v;
  }
  if (span_sum < 1.0F) {
    if (dbg) {
      std::cerr << "moonshine_tts: heteronym debug: fallback (span_mask sum < 1) span_sum="
                << span_sum << '\n';
    }
    return cmudict_alternatives[0];
  }

  if (dbg) {
    std::cerr << "moonshine_tts: heteronym debug: lookup_key=" << std::quoted(lookup_key)
              << " gkey=" << std::quoted(gkey) << " span_cp=[" << span_s << ',' << span_e
              << ") window_cp_len=" << utf8_split_codepoints(window_text).size() << " ws=" << ws
              << " we=" << we << " window=" << std::quoted(window_text) << '\n';
    std::cerr << "moonshine_tts: heteronym debug: homograph ordered IPA (training order):";
    const auto& oc = tab_.ordered_candidates.at(gkey);
    for (size_t i = 0; i < oc.size(); ++i) {
      std::cerr << " [" << i << "]=" << std::quoted(oc[i]);
    }
    std::cerr << '\n';
    std::cerr << "moonshine_tts: heteronym debug: CMU dict alts (lookup order, may match homograph):";
    for (size_t i = 0; i < cmudict_alternatives.size(); ++i) {
      std::cerr << " [" << i << "]=" << std::quoted(cmudict_alternatives[i]);
    }
    std::cerr << '\n';
    std::cerr << "moonshine_tts: heteronym debug: encoder_input_ids[0..min(31)]:";
    for (int i = 0; i < tab_.max_seq_len && i < 32; ++i) {
      std::cerr << ' ' << ids[static_cast<size_t>(i)];
    }
    std::cerr << '\n';
  }

  const std::array<int64_t, 2> enc_shape{1, tab_.max_seq_len};

  std::vector<int64_t> cur;
  cur.push_back(tab_.bos);

  const char* in_names[] = {"encoder_input_ids", "encoder_attention_mask", "span_mask",
                            "decoder_input_ids", "decoder_attention_mask"};
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
        mem_, ids.data(), ids.size(), enc_shape.data(), enc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, attn_1d.data(), attn_1d.size(), enc_shape.data(), enc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, span.data(), span.size(), enc_shape.data(), enc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, dec_row.data(), dec_row.size(), dec_shape.data(), dec_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, dec_mask.data(), dec_mask.size(), dec_shape.data(), dec_shape.size()));

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(),
                                out_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
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

  std::vector<std::string> pred_tokens;
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
      pred_tokens.push_back(tok);
    }
  }

  const std::string raw =
      pick_closest_cmudict_ipa(pred_tokens, cmudict_alternatives, tab_.levenshtein_extra);
  const auto matched = match_prediction_to_cmudict_ipa(raw, cmudict_alternatives);

  if (dbg) {
    std::ostringstream pred_joined;
    for (size_t i = 0; i < pred_tokens.size(); ++i) {
      pred_joined << pred_tokens[i];
    }
    std::cerr << "moonshine_tts: heteronym debug: pred_phoneme_toks(n=" << pred_tokens.size()
              << "):";
    for (size_t i = 0; i < pred_tokens.size() && i < 48; ++i) {
      std::cerr << ' ' << std::quoted(pred_tokens[i]);
    }
    if (pred_tokens.size() > 48) {
      std::cerr << " ...";
    }
    std::cerr << "\nmoonshine_tts: heteronym debug: pred_joined=" << std::quoted(pred_joined.str())
              << '\n';
    const int n = static_cast<int>(cmudict_alternatives.size());
    int best_i = 0;
    int best_d = std::numeric_limits<int>::max();
    std::cerr << "moonshine_tts: heteronym debug: levenshtein (extra_phonemes="
              << tab_.levenshtein_extra << "):";
    for (int i = 0; i < n; ++i) {
      const auto cand = ipa_string_to_phoneme_tokens(cmudict_alternatives[static_cast<size_t>(i)]);
      const int lim =
          static_cast<int>(cand.size()) + std::max(0, static_cast<int>(tab_.levenshtein_extra));
      const int take = std::min(lim, static_cast<int>(pred_tokens.size()));
      std::vector<std::string> prefix(pred_tokens.begin(),
                                      pred_tokens.begin() + static_cast<ptrdiff_t>(take));
      const int d = levenshtein_distance(cand, prefix);
      std::cerr << " d[" << i << "]=" << d;
      if (d < best_d) {
        best_d = d;
        best_i = i;
      }
    }
    std::cerr << " -> pick_i=" << best_i << " pick_closest_cmudict_ipa=" << std::quoted(raw)
              << '\n';
    std::cerr << "moonshine_tts: heteronym debug: match_prediction_to_cmudict_ipa=";
    if (matched.has_value()) {
      std::cerr << std::quoted(*matched);
    } else {
      std::cerr << "nullopt";
    }
    std::cerr << " return="
              << std::quoted(matched.has_value() ? *matched : cmudict_alternatives[0]) << '\n';
  }

  return matched ? *matched : cmudict_alternatives[0];
}

}  // namespace moonshine_tts
