#ifndef SPELLING_MODEL_H
#define SPELLING_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include <mutex>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"
#include "spelling-fusion.h"

/* Wraps the SpellingCNN ``.ort`` model. Mirrors the construction /
   loading shape of ``SpeakerEmbeddingModel``: an instance is created
   without a session, then ``load`` or ``load_from_memory`` is called.

   All metadata defaults (sample rate, clip seconds, class list,
   input/output tensor names) live in ``spelling-fusion-data.cpp`` and
   are baked into the binary. ``load*`` overrides the defaults from the
   model's embedded ``custom_metadata_map`` when present so a caller
   that points at a re-trained model gets the right config without
   shipping a sidecar JSON file. */

class SpellingModel {
 public:
  static constexpr size_t default_target_samples = 16000;

  explicit SpellingModel(bool log_ort_run = false);
  ~SpellingModel();

  SpellingModel(const SpellingModel &) = delete;
  SpellingModel &operator=(const SpellingModel &) = delete;

  // Load from a ``.ort`` file on disk. Returns 0 on success, non-zero
  // ORT error code on failure.
  int load(const char *model_path);

  // Load from a memory buffer. The buffer must remain valid for the
  // lifetime of the model (it is not copied).
  int load_from_memory(const uint8_t *model_data, size_t model_data_size);

  // Run the model on a single audio clip and write the top-1
  // prediction to ``out_prediction``. Returns 0 on success.
  //
  // ``audio`` is 1-D float PCM at ``sample_rate`` Hz; the wrapper
  // pads / truncates to ``clip_seconds`` worth of ``sample_rate``
  // samples internally. Any sample-rate mismatch is rejected with
  // an error code.
  int predict(const float *audio, size_t audio_size, int32_t sample_rate,
              SpellingPrediction *out_prediction);

  // Accessors.
  int32_t sample_rate() const { return sample_rate_; }
  float clip_seconds() const { return clip_seconds_; }
  const std::vector<std::string> &classes() const { return classes_; }

 private:
  void initialize_session_options();
  int populate_metadata_from_session();
  void apply_default_metadata();

  const OrtApi *ort_api_ = nullptr;
  OrtEnv *ort_env_ = nullptr;
  OrtSessionOptions *ort_session_options_ = nullptr;
  OrtMemoryInfo *ort_memory_info_ = nullptr;
  OrtSession *ort_session_ = nullptr;
  // Named without trailing underscore so ORT_RUN's ``this->log_ort_run``
  // expansion compiles.
  bool log_ort_run = false;
  std::mutex processing_mutex_;

  const char *mmapped_data_ = nullptr;
  size_t mmapped_data_size_ = 0;

  // Metadata, defaulted from spelling-fusion-data and overridden by
  // values found in the loaded model's ``custom_metadata_map``.
  int32_t sample_rate_ = 16000;
  float clip_seconds_ = 1.0f;
  std::string input_name_ = "waveform";
  std::string output_name_ = "logits";
  std::vector<std::string> classes_;
  size_t target_samples_ = 16000;
};

#endif  // SPELLING_MODEL_H
