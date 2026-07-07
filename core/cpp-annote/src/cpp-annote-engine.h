// SPDX-License-Identifier: MIT
// Internal engine class for segmentation ORT + embedding ORT + VBx (PLDA).
// Not part of the public API — use CppAnnote (cpp-annote.h) instead.

#ifndef CPP_ANNOTE_ENGINE_H_
#define CPP_ANNOTE_ENGINE_H_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clustering_vbx.h"
#include "cpp-annote.h"
#include "plda_vbx.h"

namespace cppannote {

struct DiarizationProfile {
  int total_chunks = 0;
  int num_frames = 0;
  int num_classes = 0;
  double segmentation_ort_sec = 0.;
  double embedding_ort_sec = 0.;
  double clustering_vbx_sec = 0.;
  double reconstruct_sec = 0.;
  double total_sec = 0.;

  void print(std::ostream& os, const char* prefix = "  ") const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s%d chunks, %d frames, %d classes\n"
                  "%ssegmentation_ort: %.3fs\n"
                  "%sembedding_ort:    %.3fs\n"
                  "%sclustering_vbx:   %.3fs\n"
                  "%sreconstruct:      %.3fs\n"
                  "%stotal:            %.3fs\n",
                  prefix, total_chunks, num_frames, num_classes, prefix,
                  segmentation_ort_sec, prefix, embedding_ort_sec, prefix,
                  clustering_vbx_sec, prefix, reconstruct_sec, prefix,
                  total_sec);
    os << buf;
  }

  void accumulate(const DiarizationProfile& o) {
    segmentation_ort_sec += o.segmentation_ort_sec;
    embedding_ort_sec += o.embedding_ort_sec;
    clustering_vbx_sec += o.clustering_vbx_sec;
    reconstruct_sec += o.reconstruct_sec;
    total_sec += o.total_sec;
  }
};

class CppAnnoteEngine {
 public:
  /// Construct from compiled-in ORT model data (community-1 defaults).
  CppAnnoteEngine();

  /// Construct with optional file-based ONNX models.  Pass an empty string
  /// to use the compiled-in default for that model.
  CppAnnoteEngine(const std::string& segmentation_onnx_path,
                   const std::string& embedding_onnx_path);

  CppAnnoteEngine(const CppAnnoteEngine&) = delete;
  CppAnnoteEngine& operator=(const CppAnnoteEngine&) = delete;
  CppAnnoteEngine(CppAnnoteEngine&&) = delete;
  CppAnnoteEngine& operator=(CppAnnoteEngine&&) = delete;

  static std::vector<float> extract_chunk_audio(const float* audio,
                                                int64_t num_samples,
                                                int64_t offset,
                                                int chunk_num_samples,
                                                int num_channels);

  std::vector<float> run_segmentation_ort_single(const float* chunk_buf);

  std::vector<float> run_embedding_ort_single(const float* chunk_mono,
                                              const float* seg_binarized);

  std::vector<DiarizationTurn> cluster_and_decode(
      const std::vector<float>& seg_out, const std::vector<float>& emb, int C,
      DiarizationProfile& profile, double chunk_step_sec_override = 0.0);

  int segmentation_model_sample_rate() const { return cfg_.sr_model; }
  int segmentation_num_channels() const { return cfg_.num_channels; }
  int segmentation_chunk_num_samples() const { return cfg_.chunk_num_samples; }
  double segmentation_chunk_step_sec() const { return cfg_.chunk_step_sec; }
  double segmentation_chunk_duration_sec() const { return cfg_.chunk_dur_sec; }
  int seg_frames_per_chunk() const { return seg_F_; }
  int seg_classes() const { return seg_K_; }
  int embedding_dimension() const { return embed_dim_; }

 private:
  struct SegConfig {
    int sr_model = 0;
    int num_channels = 0;
    int chunk_num_samples = 0;
    bool multilabel_export = false;
    double chunk_step_sec = 0.;
    double chunk_dur_sec = 0.;
  };

  std::string golden_bounds_body_;

  SegConfig cfg_{};
  int seg_F_ = 0;
  int seg_K_ = 0;
  double rf_dur_ = 0.;
  double rf_step_ = 0.;
  double min_off_ = 0.;
  double min_on_ = 0.;

  Ort::Env ort_env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::MemoryInfo mem_;
  Ort::AllocatorWithDefaultOptions alloc_;
  Ort::AllocatedStringPtr in_name_;
  Ort::AllocatedStringPtr out_name_;

  int embed_sr_ = 16000;
  int embed_mel_bins_ = 80;
  float embed_frame_length_ms_ = 25.f;
  float embed_frame_shift_ms_ = 10.f;
  int embed_dim_ = 256;
  bool embedding_exclude_overlap_ = false;
  int min_num_samples_ = 800;

  std::unique_ptr<Ort::Session> embed_session_;
  bool embed_inputs_fbank_then_weights_ = true;

  std::unique_ptr<plda_vbx::PldaModel> plda_model_;
  clustering_vbx::VbxClusteringParams vbx_params_{};

  void init_config_and_models(const std::string& embedding_onnx_path);
};

}  // namespace cppannote

#endif  // CPP_ANNOTE_ENGINE_H_
