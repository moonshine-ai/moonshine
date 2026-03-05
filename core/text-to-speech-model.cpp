#include "text-to-speech-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "string-utils.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

/* ============================================================================
 * Helpers – minimal vocab.json parsing
 *
 * The vocab.json produced by the training pipeline looks like:
 *
 *   {
 *     "pad": "_",
 *     "punctuation": ";:,.!?...",
 *     "letters": "ABCabc...",
 *     "letters_ipa": "ɑɐɒ..."
 *   }
 *
 * We only need the four string values.  A full JSON library is overkill, so we
 * reuse the same style of hand-rolled extraction that the streaming model uses
 * for its config.
 * ========================================================================= */

namespace {

/** Read the contents of a file into a std::string. */
static std::string read_file_contents(const char *path) {
  std::ifstream f(path);
  if (!f.good()) return "";
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

/**
 * Extract a JSON string value for a given key.
 * E.g. for key "pad" in {"pad": "_", ...} this returns "_".
 * Handles escaped characters (\", \\, \n, \t, \uXXXX basic-BMP).
 */
static std::string json_get_string(const std::string &json, const char *key) {
  std::string search = std::string("\"") + key + "\"";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return "";

  // Skip past the key, colon, and whitespace to the opening quote.
  pos += search.length();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                json[pos] == ':' || json[pos] == '\n' ||
                                json[pos] == '\r'))
    pos++;

  if (pos >= json.size() || json[pos] != '"') return "";
  pos++;  // skip opening quote

  std::string result;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      pos++;
      switch (json[pos]) {
        case '"':
          result += '"';
          break;
        case '\\':
          result += '\\';
          break;
        case 'n':
          result += '\n';
          break;
        case 't':
          result += '\t';
          break;
        case 'u': {
          // Basic \uXXXX → UTF-8 (BMP only, sufficient for IPA symbols).
          if (pos + 4 < json.size()) {
            char hex[5] = {json[pos + 1], json[pos + 2], json[pos + 3],
                           json[pos + 4], '\0'};
            uint32_t cp = static_cast<uint32_t>(strtoul(hex, nullptr, 16));
            pos += 4;
            if (cp < 0x80) {
              result += static_cast<char>(cp);
            } else if (cp < 0x800) {
              result += static_cast<char>(0xC0 | (cp >> 6));
              result += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              result += static_cast<char>(0xE0 | (cp >> 12));
              result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (cp & 0x3F));
            }
          }
          break;
        }
        default:
          result += json[pos];
          break;
      }
    } else {
      result += json[pos];
    }
    pos++;
  }
  return result;
}

/**
 * Iterate over a UTF-8 string one code-point at a time, calling `fn` with each
 * code-point as a std::string.
 */
template <typename Fn>
static void for_each_utf8_char(const std::string &s, Fn fn) {
  size_t i = 0;
  while (i < s.size()) {
    size_t len = 1;
    uint8_t c = static_cast<uint8_t>(s[i]);
    if ((c & 0x80) == 0)
      len = 1;
    else if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    fn(s.substr(i, len));
    i += len;
  }
}

}  // namespace

/* ============================================================================
 * TextToSpeechModel implementation
 * ========================================================================= */

TextToSpeechModel::TextToSpeechModel()
    : ort_api_(nullptr),
      ort_env_(nullptr),
      ort_session_options_(nullptr),
      ort_memory_info_(nullptr),
      ort_allocator_(nullptr),
      session_(nullptr),
      mmapped_data_(nullptr),
      mmapped_data_size_(0) {
  ort_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                    "TextToSpeechModel", &ort_env_));

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info_));

  ort_allocator_ = new MoonshineOrtAllocator(ort_memory_info_);

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateSessionOptions(&ort_session_options_));
  LOG_ORT_ERROR(ort_api_, ort_api_->SetSessionGraphOptimizationLevel(
                              ort_session_options_, ORT_ENABLE_ALL));
  LOG_ORT_ERROR(ort_api_,
                ort_api_->SetIntraOpNumThreads(ort_session_options_, 1));
}

TextToSpeechModel::~TextToSpeechModel() {
  if (session_) ort_api_->ReleaseSession(session_);
  if (ort_session_options_)
    ort_api_->ReleaseSessionOptions(ort_session_options_);
  if (ort_memory_info_) ort_api_->ReleaseMemoryInfo(ort_memory_info_);
  if (ort_env_) ort_api_->ReleaseEnv(ort_env_);
  delete ort_allocator_;

#ifndef _WIN32
  if (mmapped_data_) {
    munmap(const_cast<char *>(mmapped_data_), mmapped_data_size_);
  }
#endif
}

/* --------------------------------------------------------------------------
 * Loading
 * ---------------------------------------------------------------------- */

int TextToSpeechModel::load(const char *model_dir) {
  if (model_dir == nullptr) {
    LOG("Model directory is null\n");
    return 1;
  }

  std::string model_path = append_path_component(model_dir, "model.onnx");
  std::string vocab_path = append_path_component(model_dir, "vocab.json");

  // Load ONNX session
  RETURN_ON_ERROR(ort_session_from_path(ort_api_, ort_env_, ort_session_options_,
                                        model_path.c_str(), &session_,
                                        &mmapped_data_, &mmapped_data_size_));
  RETURN_ON_NULL(session_);

  // Query the model's output name.
  {
    OrtAllocator *allocator = nullptr;
    OrtStatus *s = ort_api_->GetAllocatorWithDefaultOptions(&allocator);
    if (s != nullptr) {
      LOG_ORT_ERROR(ort_api_, s);
      return 1;
    }
    char *name = nullptr;
    s = ort_api_->SessionGetOutputName(session_, 0, allocator, &name);
    if (s != nullptr) {
      LOG_ORT_ERROR(ort_api_, s);
      return 1;
    }
    output_name_ = name;
    allocator->Free(allocator, name);
    LOGF("TTS model output name: %s\n", output_name_.c_str());
  }

  // Load and parse vocab.json
  std::string vocab_json = read_file_contents(vocab_path.c_str());
  if (vocab_json.empty()) {
    LOGF("Failed to read vocab file: %s\n", vocab_path.c_str());
    return 1;
  }
  RETURN_ON_ERROR(parse_vocab(vocab_json));

  return 0;
}

int TextToSpeechModel::load_from_memory(const uint8_t *model_data,
                                        size_t model_data_size,
                                        const char *vocab_json,
                                        size_t vocab_json_size) {
  if (model_data == nullptr || model_data_size == 0) {
    LOG("Model data is null or empty\n");
    return 1;
  }
  if (vocab_json == nullptr || vocab_json_size == 0) {
    LOG("Vocab JSON data is null or empty\n");
    return 1;
  }

  RETURN_ON_ERROR(ort_session_from_memory(ort_api_, ort_env_,
                                          ort_session_options_, model_data,
                                          model_data_size, &session_));
  RETURN_ON_NULL(session_);

  // Query the model's output name.
  {
    OrtAllocator *allocator = nullptr;
    OrtStatus *s = ort_api_->GetAllocatorWithDefaultOptions(&allocator);
    if (s != nullptr) {
      LOG_ORT_ERROR(ort_api_, s);
      return 1;
    }
    char *name = nullptr;
    s = ort_api_->SessionGetOutputName(session_, 0, allocator, &name);
    if (s != nullptr) {
      LOG_ORT_ERROR(ort_api_, s);
      return 1;
    }
    output_name_ = name;
    allocator->Free(allocator, name);
  }

  std::string json_str(vocab_json, vocab_json_size);
  RETURN_ON_ERROR(parse_vocab(json_str));

  return 0;
}

/* --------------------------------------------------------------------------
 * Vocab parsing
 *
 * The symbol list is constructed by concatenating
 * [pad] + list(punctuation) + list(letters) + list(letters_ipa),
 * and each symbol is assigned an index equal to its position.
 * ---------------------------------------------------------------------- */

int TextToSpeechModel::parse_vocab(const std::string &json) {
  std::string pad = json_get_string(json, "pad");
  std::string punctuation = json_get_string(json, "punctuation");
  std::string letters = json_get_string(json, "letters");
  std::string letters_ipa = json_get_string(json, "letters_ipa");

  if (pad.empty()) {
    LOG("vocab.json: missing or empty 'pad' field\n");
    return 1;
  }

  // Build symbol list:
  //   symbols = [pad] + list(punctuation) + list(letters) + list(letters_ipa)
  char_to_index_.clear();
  int64_t index = 0;

  // pad (single character)
  for_each_utf8_char(pad, [&](const std::string &ch) {
    char_to_index_[ch] = index++;
  });

  // punctuation characters
  for_each_utf8_char(punctuation, [&](const std::string &ch) {
    char_to_index_[ch] = index++;
  });

  // letters
  for_each_utf8_char(letters, [&](const std::string &ch) {
    char_to_index_[ch] = index++;
  });

  // IPA letters
  for_each_utf8_char(letters_ipa, [&](const std::string &ch) {
    char_to_index_[ch] = index++;
  });

  LOGF("TTS vocab loaded: %zu symbols\n",
       static_cast<size_t>(char_to_index_.size()));
  return 0;
}

/* --------------------------------------------------------------------------
 * Tokenization
 *
 * Token sequence layout:
 *   tokens = text_cleaner(phonemes)            # char → index
 *   texts  = zeros([1, len(tokens) + 2])       # pad on both sides
 *   texts[0][1 : len(tokens)+1] = tokens
 *   text_lengths = [len(tokens) + 2]
 * ---------------------------------------------------------------------- */

void TextToSpeechModel::tokenize(const std::string &phonemes,
                                 std::vector<int64_t> &out_tokens,
                                 int64_t &out_length) {
  // Convert each character to its vocab index.
  std::vector<int64_t> raw_tokens;
  for_each_utf8_char(phonemes, [&](const std::string &ch) {
    auto it = char_to_index_.find(ch);
    if (it != char_to_index_.end()) {
      raw_tokens.push_back(it->second);
    } else {
      LOGF("TTS tokenizer: skipping unknown character '%s'\n", ch.c_str());
    }
  });

  // Wrap with leading and trailing pad (index 0).
  int64_t n = static_cast<int64_t>(raw_tokens.size());
  out_length = n + 2;
  out_tokens.assign(static_cast<size_t>(out_length), 0);
  for (int64_t i = 0; i < n; i++) {
    out_tokens[static_cast<size_t>(i + 1)] = raw_tokens[static_cast<size_t>(i)];
  }
}

/* --------------------------------------------------------------------------
 * Inference
 *
 * The ONNX model expects:
 *   inputs  = { "texts": int64[1, N+2], "text_lengths": int64[1] }
 *   outputs = session.run(inputs)[0]   → float waveform
 * ---------------------------------------------------------------------- */

std::vector<float> TextToSpeechModel::run_inference(
    const std::vector<int64_t> &tokens, int64_t text_length) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!session_) {
    LOG("TTS model not loaded\n");
    return {};
  }

  // Shapes
  int64_t batch = 1;
  int64_t seq_len = text_length;
  std::vector<int64_t> texts_shape = {batch, seq_len};
  std::vector<int64_t> lengths_shape = {1};

  // Create input tensors
  OrtValue *texts_tensor = nullptr;
  OrtStatus *status = ort_api_->CreateTensorWithDataAsOrtValue(
      ort_memory_info_, const_cast<int64_t *>(tokens.data()),
      tokens.size() * sizeof(int64_t), texts_shape.data(), texts_shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &texts_tensor);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    return {};
  }

  int64_t length_val = text_length;
  OrtValue *lengths_tensor = nullptr;
  status = ort_api_->CreateTensorWithDataAsOrtValue(
      ort_memory_info_, &length_val, sizeof(int64_t), lengths_shape.data(),
      lengths_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      &lengths_tensor);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(texts_tensor);
    return {};
  }

  // Input / output names (must match the ONNX graph)
  const char *input_names[] = {"texts", "text_lengths"};
  const char *out_name = output_name_.c_str();
  const char *output_names[] = {out_name};

  OrtValue *inputs[] = {texts_tensor, lengths_tensor};
  OrtValue *outputs[] = {nullptr};

  // Run
  status = ort_api_->Run(session_, nullptr, input_names, inputs, 2,
                         output_names, 1, outputs);

  ort_api_->ReleaseValue(texts_tensor);
  ort_api_->ReleaseValue(lengths_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    return {};
  }

  // Read output shape to determine waveform length.
  OrtTensorTypeAndShapeInfo *out_info = nullptr;
  status = ort_api_->GetTensorTypeAndShape(outputs[0], &out_info);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }

  size_t num_dims = 0;
  status = ort_api_->GetDimensionsCount(out_info, &num_dims);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseTensorTypeAndShapeInfo(out_info);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }
  std::vector<int64_t> out_shape(num_dims);
  status = ort_api_->GetDimensions(out_info, out_shape.data(), num_dims);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseTensorTypeAndShapeInfo(out_info);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }
  ort_api_->ReleaseTensorTypeAndShapeInfo(out_info);

  size_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= static_cast<size_t>(d);
  }

  float *raw_output = nullptr;
  status =
      ort_api_->GetTensorMutableData(outputs[0], (void **)&raw_output);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }

  std::vector<float> waveform(raw_output, raw_output + total_elements);
  ort_api_->ReleaseValue(outputs[0]);

  return waveform;
}

/* --------------------------------------------------------------------------
 * PCM conversion
 *   waveform * 32767 → int16
 * ---------------------------------------------------------------------- */

std::vector<int16_t> TextToSpeechModel::to_pcm16(
    const std::vector<float> &waveform) {
  std::vector<int16_t> pcm(waveform.size());
  for (size_t i = 0; i < waveform.size(); i++) {
    float sample = waveform[i] * 32767.0f;
    // Clamp to int16 range.
    sample = std::max(-32768.0f, std::min(32767.0f, sample));
    pcm[i] = static_cast<int16_t>(sample);
  }
  return pcm;
}

/* --------------------------------------------------------------------------
 * Public generate()
 *   phonemes → tokenize → run model → convert to PCM
 * ---------------------------------------------------------------------- */

int TextToSpeechModel::generate(const std::string &phonemes,
                                TextToSpeechResult &result) {
  if (!is_loaded()) {
    LOG("TTS model is not loaded\n");
    return 1;
  }

  // Tokenize
  std::vector<int64_t> tokens;
  int64_t text_length = 0;
  tokenize(phonemes, tokens, text_length);

  if (text_length <= 2) {
    LOG("TTS: empty phoneme input\n");
    return 1;
  }

  // Run model
  std::vector<float> waveform = run_inference(tokens, text_length);
  if (waveform.empty()) {
    LOG("TTS: model produced empty waveform\n");
    return 1;
  }

  // Convert to PCM
  result.audio_data = to_pcm16(waveform);
  result.sample_rate = 24000;

  return 0;
}

bool TextToSpeechModel::is_loaded() const {
  return session_ != nullptr && !char_to_index_.empty();
}
