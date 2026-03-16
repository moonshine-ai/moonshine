#ifndef TEXT_TO_SPEECH_MODEL_H
#define TEXT_TO_SPEECH_MODEL_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "moonshine-ort-allocator.h"
#include "onnxruntime_c_api.h"

// Forward-declare the Phonemis pipeline to avoid exposing the third-party
// header to consumers of this header.
namespace phonemis {
class Pipeline;
}  // namespace phonemis

/**
 * Result of a text-to-speech generation call.
 * The audio data is 24 kHz signed 16-bit PCM.
 */
struct TextToSpeechResult {
  std::vector<int16_t> audio_data;
  int32_t sample_rate = 24000;
};

/**
 * Text-to-speech model that converts plain text into audio waveforms.
 *
 * The model handles grapheme-to-phoneme conversion internally using the
 * Phonemis library, then tokenizes the phonemes and runs an ONNX model
 * to produce a waveform.
 *
 * Required model files:
 *   - model.onnx   (the TTS ONNX model)
 *   - vocab.json   (character-to-index mapping for tokenization)
 */
class TextToSpeechModel {
 public:
  TextToSpeechModel();
  ~TextToSpeechModel();

  /**
   * Load the model from a directory on disk.
   * The directory must contain:
   *   - model.onnx   (the TTS ONNX model)
   *   - vocab.json    (character-to-index mapping)
   * @param model_dir  Path to the model directory.
   * @return 0 on success, non-zero on failure.
   */
  int load(const char *model_dir);

  /**
   * Load the model from in-memory buffers.
   * @param model_data       Pointer to ONNX model bytes.
   * @param model_data_size  Size of model data in bytes.
   * @param vocab_json       Pointer to vocab.json contents (UTF-8 string).
   * @param vocab_json_size  Size of vocab JSON data in bytes.
   * @return 0 on success, non-zero on failure.
   */
  int load_from_memory(const uint8_t *model_data, size_t model_data_size,
                       const char *vocab_json, size_t vocab_json_size);

  /**
   * Generate speech audio from plain text.
   * The text is internally converted to phonemes using the Phonemis G2P
   * pipeline, then tokenized and fed to the ONNX model.
   * @param text    Plain text string (UTF-8).
   * @param result  Output – populated with PCM audio data and sample rate.
   * @return 0 on success, non-zero on failure.
   */
  int generate(const std::string &text, TextToSpeechResult &result);

  /** @return true if the model and vocab have been loaded successfully. */
  bool is_loaded() const;

 private:
  /* ---- Vocab / tokenization ---- */

  /**
   * Parse a vocab.json string and build the character-to-index map.
   * The JSON is expected to contain "pad", "punctuation", "letters", and
   * "letters_ipa" string fields.
   */
  int parse_vocab(const std::string &json);

  /**
   * Convert plain text to an IPA phoneme string using the Phonemis pipeline.
   */
  std::string phonemize(const std::string &text);

  /**
   * Tokenize a phoneme string using the loaded vocabulary.
   * Wraps the token sequence with leading and trailing pad tokens.
   * @param phonemes      Input phoneme string.
   * @param out_tokens    Output token array (shape: [1, N+2]).
   * @param out_length    Output length scalar (N+2).
   */
  void tokenize(const std::string &phonemes, std::vector<int64_t> &out_tokens,
                int64_t &out_length);

  /**
   * Run the ONNX model and return the raw float waveform.
   */
  std::vector<float> run_inference(const std::vector<int64_t> &tokens,
                                   int64_t text_length);

  /**
   * Convert a float waveform (±1.0 range) to signed 16-bit PCM.
   */
  static std::vector<int16_t> to_pcm16(const std::vector<float> &waveform);

  /* ---- ORT resources ---- */
  const OrtApi *ort_api_;
  OrtEnv *ort_env_;
  OrtSessionOptions *ort_session_options_;
  OrtMemoryInfo *ort_memory_info_;
  MoonshineOrtAllocator *ort_allocator_;
  OrtSession *session_;

  const char *mmapped_data_;
  size_t mmapped_data_size_;

  /* ---- Model metadata ---- */
  std::string output_name_;

  /* ---- Vocab ---- */
  std::map<std::string, int64_t> char_to_index_;

  /* ---- G2P pipeline ---- */
  std::unique_ptr<phonemis::Pipeline> g2p_pipeline_;

  /* ---- Thread safety ---- */
  std::mutex mutex_;
};

#endif
