// SPDX-License-Identifier: MIT

#include "cpp-annote.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "annotation_support.h"
#include "clustering_vbx.h"
#include "community1_cpp_annote_embedded.h"
#include "community1_ort_embedded.h"
#include "compute_fbank.h"
#include "cpp-annote-engine.h"
#include "cpp-annote-streaming.h"
#include "embedding_ort_infer.h"
#include "parity_log.h"
#include "plda_vbx.h"
#include "wav_pcm_float32.h"

namespace cppannote {
namespace {

double json_double(const std::string &json, const char *key) {
  const std::string pat = std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("json missing \"") + key + "\"");
  }
  return std::stod(m[1].str());
}

bool json_bool(const std::string &json, const char *key) {
  const std::string pat = std::string("\"") + key + "\"\\s*:\\s*(true|false)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("json missing bool \"") + key + "\"");
  }
  return m[1].str() == "true";
}

int closest_frame(double t, double sw_start, double sw_duration,
                  double sw_step) {
  const double x = (t - sw_start - 0.5 * sw_duration) / sw_step;
  return static_cast<int>(std::lrint(x));
}

void trim_warmup_inplace(std::vector<float> &data, size_t num_chunks,
                         size_t &num_frames, size_t num_classes, double warm0,
                         double warm1, double &chunk_start,
                         double &chunk_duration) {
  const size_t n_left =
      static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm0));
  const size_t n_right =
      static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm1));
  if (n_left + n_right >= num_frames) {
    throw std::runtime_error("trim: warm_up removes all frames");
  }
  const size_t new_frames = num_frames - n_left - n_right;
  std::vector<float> out(num_chunks * new_frames * num_classes);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < new_frames; ++f) {
      for (size_t k = 0; k < num_classes; ++k) {
        out[(c * new_frames + f) * num_classes + k] =
            data[(c * num_frames + (f + n_left)) * num_classes + k];
      }
    }
  }
  data.swap(out);
  num_frames = new_frames;
  chunk_start += warm0 * chunk_duration;
  chunk_duration *= (1.0 - warm0 - warm1);
}

void inference_aggregate(const std::vector<float> &scores, size_t num_chunks,
                         size_t num_frames_per_chunk, size_t num_classes,
                         double chunks_start, double chunks_duration,
                         double chunks_step, double out_duration,
                         double out_step, bool skip_average, float epsilon,
                         float missing, std::vector<float> &out_avg,
                         int &num_out_frames) {
  const double out_sw_start = chunks_start;
  const double out_sw_duration = out_duration;
  const double out_sw_step = out_step;
  const double end_t = chunks_start + chunks_duration +
                       static_cast<double>(num_chunks - 1) * chunks_step +
                       0.5 * out_sw_duration;
  num_out_frames =
      closest_frame(end_t, out_sw_start, out_sw_duration, out_sw_step) + 1;
  if (num_out_frames <= 0) {
    throw std::runtime_error("aggregate: non-positive num_out_frames");
  }
  const int nf = static_cast<int>(num_frames_per_chunk);
  const int nc = static_cast<int>(num_classes);
  std::vector<float> agg(static_cast<size_t>(num_out_frames) * num_classes,
                         0.f);
  std::vector<float> occ(static_cast<size_t>(num_out_frames) * num_classes,
                         0.f);
  std::vector<float> mask_max(static_cast<size_t>(num_out_frames) * num_classes,
                              0.f);
  for (size_t ci = 0; ci < num_chunks; ++ci) {
    const double chunk_start =
        chunks_start + static_cast<double>(ci) * chunks_step;
    const int start_frame =
        closest_frame(chunk_start + 0.5 * out_sw_duration, out_sw_start,
                      out_sw_duration, out_sw_step);
    for (int j = 0; j < nf; ++j) {
      for (int k = 0; k < nc; ++k) {
        const size_t idx =
            (ci * static_cast<size_t>(nf) + static_cast<size_t>(j)) *
                num_classes +
            static_cast<size_t>(k);
        const float raw = scores[idx];
        const float mask = std::isnan(raw) ? 0.f : 1.f;
        float score = raw;
        if (std::isnan(score)) {
          score = 0.f;
        }
        const int fi = start_frame + j;
        if (fi < 0 || fi >= num_out_frames) {
          continue;
        }
        const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) +
                         static_cast<size_t>(k);
        agg[o] += score * mask;
        occ[o] += mask;
        mask_max[o] = std::max(mask_max[o], mask);
      }
    }
  }
  out_avg.resize(static_cast<size_t>(num_out_frames) * num_classes);
  for (int fi = 0; fi < num_out_frames; ++fi) {
    for (int k = 0; k < nc; ++k) {
      const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) +
                       static_cast<size_t>(k);
      if (skip_average) {
        out_avg[o] = agg[o];
      } else {
        out_avg[o] = agg[o] / std::max(occ[o], epsilon);
      }
      if (mask_max[o] == 0.f) {
        out_avg[o] = missing;
      }
    }
  }
}

std::vector<std::uint8_t> speaker_count_initial_uint8(
    std::vector<float> binarized, size_t num_chunks, size_t num_frames,
    size_t num_classes, double &chunk_start, double chunk_step,
    double &chunk_duration, double rf_dur, double rf_step,
    int &num_out_frames) {
  size_t nf = num_frames;
  trim_warmup_inplace(binarized, num_chunks, nf, num_classes, 0.0, 0.0,
                      chunk_start, chunk_duration);
  std::vector<float> summed(num_chunks * nf * 1);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < nf; ++f) {
      float s = 0.f;
      for (size_t k = 0; k < num_classes; ++k) {
        s += binarized[(c * nf + f) * num_classes + k];
      }
      summed[c * nf + f] = s;
    }
  }
  std::vector<float> avg;
  inference_aggregate(summed, num_chunks, nf, 1, chunk_start, chunk_duration,
                      chunk_step, rf_dur, rf_step, false, 1e-12f, 0.f, avg,
                      num_out_frames);
  std::vector<std::uint8_t> out(static_cast<size_t>(num_out_frames));
  for (int i = 0; i < num_out_frames; ++i) {
    const double r =
        std::rint(static_cast<double>(avg[static_cast<size_t>(i)]));
    out[static_cast<size_t>(i)] =
        static_cast<std::uint8_t>(std::max(0.0, std::min(255.0, r)));
  }
  return out;
}

std::vector<std::int8_t> cap_count(const std::vector<std::uint8_t> &u8,
                                   int max_cap) {
  std::vector<std::int8_t> out(u8.size());
  for (size_t i = 0; i < u8.size(); ++i) {
    const int m = std::min(static_cast<int>(u8[i]), max_cap);
    out[i] = static_cast<std::int8_t>(static_cast<std::uint8_t>(m));
  }
  return out;
}

void crop_loose_frame_range(double focus_start, double focus_end,
                            double sw_start, double sw_duration, double sw_step,
                            int &out_i0, int &out_i1_exclusive) {
  const double i_ = (focus_start - sw_duration - sw_start) / sw_step;
  int i = static_cast<int>(std::ceil(i_));
  const double j_ = (focus_end - sw_start) / sw_step;
  int j = static_cast<int>(std::floor(j_));
  out_i0 = i;
  out_i1_exclusive = j + 1;
}

void extent_of_frames(double sw_start, double sw_step, size_t n_rows,
                      double &seg_start, double &seg_end) {
  seg_start = sw_start;
  seg_end = sw_start + static_cast<double>(n_rows) * sw_step;
}

void crop_feature_loose(const std::vector<float> &data, int n_samples,
                        int n_cols, double sw_start, double sw_duration,
                        double sw_step, double focus_start, double focus_end,
                        std::vector<float> &out_data, int &out_rows,
                        double &new_sw_start) {
  int i0 = 0;
  int i1 = 0;
  crop_loose_frame_range(focus_start, focus_end, sw_start, sw_duration, sw_step,
                         i0, i1);
  const int clipped0 = std::max(0, i0);
  const int clipped1 = std::min(n_samples, i1);
  if (clipped0 >= clipped1) {
    out_rows = 0;
    out_data.clear();
    new_sw_start = sw_start;
    return;
  }
  out_rows = clipped1 - clipped0;
  out_data.resize(static_cast<size_t>(out_rows) * static_cast<size_t>(n_cols));
  for (int r = 0; r < out_rows; ++r) {
    const int src_row = clipped0 + r;
    for (int c = 0; c < n_cols; ++c) {
      out_data[static_cast<size_t>(r) * static_cast<size_t>(n_cols) +
               static_cast<size_t>(c)] =
          data[static_cast<size_t>(src_row) * static_cast<size_t>(n_cols) +
               static_cast<size_t>(c)];
    }
  }
  new_sw_start = sw_start + static_cast<double>(clipped0) * sw_step;
}

std::vector<int> argsort_desc_stable(const float *row, int k) {
  std::vector<int> idx(static_cast<size_t>(k));
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [row](int a, int b) {
    if (row[a] != row[b]) {
      return row[a] > row[b];
    }
    return a < b;
  });
  return idx;
}

std::vector<float> reconstruct_to_diarization(
    const std::vector<float> &segmentations, int C, int F, int L, double seg_ss,
    double seg_sd, double seg_st, const std::int8_t *hard_clusters,
    const std::vector<std::int8_t> &count_flat, double cnt_ss, double cnt_sd,
    double cnt_st, int &out_num_speaker_cols) {
  int max_clu = -3;
  for (int c = 0; c < C; ++c) {
    for (int j = 0; j < L; ++j) {
      max_clu = std::max(max_clu, static_cast<int>(hard_clusters[c * L + j]));
    }
  }
  const int num_clusters = max_clu + 1;
  if (num_clusters <= 0) {
    throw std::runtime_error("reconstruct: no positive cluster ids");
  }
  std::vector<float> clustered(static_cast<size_t>(C) * static_cast<size_t>(F) *
                                   static_cast<size_t>(num_clusters),
                               std::numeric_limits<float>::quiet_NaN());
  for (int c = 0; c < C; ++c) {
    const float *segm =
        &segmentations[static_cast<size_t>(c) * static_cast<size_t>(F) *
                       static_cast<size_t>(L)];
    const std::int8_t *cluster = &hard_clusters[c * L];
    std::vector<char> seen_k(static_cast<size_t>(num_clusters), 0);
    for (int j = 0; j < L; ++j) {
      const int k = static_cast<int>(cluster[j]);
      if (k == -2) {
        continue;
      }
      if (k >= 0 && k < num_clusters) {
        seen_k[static_cast<size_t>(k)] = 1;
      }
    }
    for (int k = 0; k < num_clusters; ++k) {
      if (!seen_k[static_cast<size_t>(k)]) {
        continue;
      }
      for (int f = 0; f < F; ++f) {
        bool any_finite = false;
        float m = 0.f;
        for (int j = 0; j < L; ++j) {
          if (static_cast<int>(cluster[j]) != k) {
            continue;
          }
          const float v = segm[static_cast<size_t>(f) * static_cast<size_t>(L) +
                               static_cast<size_t>(j)];
          if (std::isnan(v)) {
            continue;
          }
          if (!any_finite) {
            m = v;
            any_finite = true;
          } else {
            m = std::max(m, v);
          }
        }
        const size_t dst = (static_cast<size_t>(c) * static_cast<size_t>(F) +
                            static_cast<size_t>(f)) *
                               static_cast<size_t>(num_clusters) +
                           static_cast<size_t>(k);
        clustered[dst] =
            any_finite ? m : std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  std::vector<float> activations;
  int T = 0;
  inference_aggregate(clustered, static_cast<size_t>(C), static_cast<size_t>(F),
                      static_cast<size_t>(num_clusters), seg_ss, seg_sd, seg_st,
                      cnt_sd, cnt_st, true, 1e-12f, 0.f, activations, T);
  int K = num_clusters;
  int max_spf = 0;
  for (size_t t = 0; t < count_flat.size(); ++t) {
    max_spf = std::max(max_spf, static_cast<int>(count_flat[t]));
  }
  max_spf = std::max(0, max_spf);
  if (K < max_spf) {
    std::vector<float> padded(
        static_cast<size_t>(T) * static_cast<size_t>(max_spf), 0.f);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        padded[static_cast<size_t>(t) * static_cast<size_t>(max_spf) +
               static_cast<size_t>(k)] =
            activations[static_cast<size_t>(t) * static_cast<size_t>(K) +
                        static_cast<size_t>(k)];
      }
    }
    activations.swap(padded);
    K = max_spf;
  }
  double act_s = 0, act_e = 0;
  extent_of_frames(cnt_ss, cnt_st, static_cast<size_t>(T), act_s, act_e);
  const int Tcnt = static_cast<int>(count_flat.size());
  double cnt_s = 0, cnt_e = 0;
  extent_of_frames(cnt_ss, cnt_st, static_cast<size_t>(Tcnt), cnt_s, cnt_e);
  const double inter_s = std::max(act_s, cnt_s);
  const double inter_e = std::min(act_e, cnt_e);
  std::vector<float> act_cropped;
  int act_rows = 0;
  double tmp0 = 0.;
  crop_feature_loose(activations, T, K, cnt_ss, cnt_sd, cnt_st, inter_s,
                     inter_e, act_cropped, act_rows, tmp0);
  std::vector<float> cnt_2d(static_cast<size_t>(Tcnt));
  for (int t = 0; t < Tcnt; ++t) {
    cnt_2d[static_cast<size_t>(t)] =
        static_cast<float>(count_flat[static_cast<size_t>(t)]);
  }
  std::vector<float> cnt_cropped;
  int cnt_rows = 0;
  double tmp1 = 0.;
  crop_feature_loose(cnt_2d, Tcnt, 1, cnt_ss, cnt_sd, cnt_st, inter_s, inter_e,
                     cnt_cropped, cnt_rows, tmp1);
  if (act_rows != cnt_rows) {
    throw std::runtime_error("crop row mismatch");
  }
  std::vector<float> binary(
      static_cast<size_t>(act_rows) * static_cast<size_t>(K), 0.f);
  for (int t = 0; t < act_rows; ++t) {
    const float *arow =
        &act_cropped[static_cast<size_t>(t) * static_cast<size_t>(K)];
    std::vector<int> order = argsort_desc_stable(arow, K);
    const int c = static_cast<int>(
        std::lrint(static_cast<double>(cnt_cropped[static_cast<size_t>(t)])));
    const int c_use = std::max(0, std::min(c, K));
    for (int i = 0; i < c_use; ++i) {
      binary[static_cast<size_t>(t) * static_cast<size_t>(K) +
             static_cast<size_t>(order[static_cast<size_t>(i)])] = 1.f;
    }
  }
  out_num_speaker_cols = K;
  return binary;
}

void binarize_column(const float *k_scores, int num_frames, double sw_start,
                     double sw_dur, double sw_step, double onset, double offset,
                     double pad_onset, double pad_offset,
                     std::vector<std::pair<double, double>> &regions_out) {
  if (num_frames <= 0) {
    return;
  }
  std::vector<double> ts(static_cast<size_t>(num_frames));
  for (int i = 0; i < num_frames; ++i) {
    ts[static_cast<size_t>(i)] =
        sw_start + static_cast<double>(i) * sw_step + 0.5 * sw_dur;
  }
  double start = ts[0];
  bool is_active = static_cast<double>(k_scores[0]) > onset;
  for (int i = 1; i < num_frames; ++i) {
    const double t = ts[static_cast<size_t>(i)];
    const double y = static_cast<double>(k_scores[i]);
    if (is_active) {
      if (y < offset) {
        regions_out.emplace_back(start - pad_onset, t + pad_offset);
        start = t;
        is_active = false;
      }
    } else {
      if (y > onset) {
        start = t;
        is_active = true;
      }
    }
  }
  if (is_active) {
    const double t_end = ts[static_cast<size_t>(num_frames - 1)];
    regions_out.emplace_back(start - pad_onset, t_end + pad_offset);
  }
}

bool try_regex_double(const std::string &json, const std::string &key_esc,
                      double &out) {
  const std::string pat = "\"" + key_esc + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = std::stod(m[1].str());
  return true;
}

void filter_min_duration_on(std::vector<std::pair<double, double>> &regs,
                            double min_on) {
  if (min_on <= 0.0) {
    return;
  }
  std::vector<std::pair<double, double>> kept;
  for (const auto &pr : regs) {
    if (pr.second - pr.first >= min_on - 1e-12) {
      kept.push_back(pr);
    }
  }
  regs.swap(kept);
}

bool try_json_bool_field(const std::string &json, const char *key_esc,
                         bool &out) {
  const std::string pat =
      std::string("\"") + key_esc + "\"\\s*:\\s*(true|false)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = (m[1].str() == "true");
  return true;
}

bool try_regex_int(const std::string &json, const std::string &key_esc,
                   int &out) {
  const std::string pat = "\"" + key_esc + "\"\\s*:\\s*([-+0-9]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = std::stoi(m[1].str());
  return true;
}

}  // namespace

Ort::Session make_segmentation_session(Ort::Env &env,
                                       Ort::SessionOptions &opts,
                                       const std::string &path) {
  if (!path.empty()) {
    return Ort::Session(env, path.c_str(), opts);
  }
  return Ort::Session(
      env, cppannote::embedded_community1::segmentation_ort_data,
      cppannote::embedded_community1::segmentation_ort_data_size, opts);
}

std::unique_ptr<Ort::Session> make_embedding_session(
    Ort::Env &env, Ort::SessionOptions &opts, const std::string &path) {
  if (!path.empty()) {
    return std::make_unique<Ort::Session>(env, path.c_str(), opts);
  }
  return std::make_unique<Ort::Session>(
      env, cppannote::embedded_community1::embedding_ort_data,
      cppannote::embedded_community1::embedding_ort_data_size, opts);
}

void CppAnnoteEngine::init_config_and_models(
    const std::string &embedding_onnx_path) {
  const std::string seg_json(
      cppannote::embedded_community1::segmentation_json,
      cppannote::embedded_community1::segmentation_json_size);
  cfg_.sr_model = static_cast<int>(json_double(seg_json, "sample_rate"));
  cfg_.num_channels = static_cast<int>(json_double(seg_json, "num_channels"));
  cfg_.chunk_num_samples =
      static_cast<int>(json_double(seg_json, "chunk_num_samples"));
  cfg_.multilabel_export =
      json_bool(seg_json, "export_includes_powerset_to_multilabel");
  cfg_.chunk_step_sec = 0.1 * json_double(seg_json, "chunk_duration_sec");
  {
    std::regex re("\"chunk_step_sec\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (std::regex_search(seg_json, m, re)) {
      cfg_.chunk_step_sec = std::stod(m[1].str());
    }
  }
  cfg_.chunk_dur_sec = json_double(seg_json, "chunk_duration_sec");

  const std::string rf_txt(
      cppannote::embedded_community1::receptive_field_json,
      cppannote::embedded_community1::receptive_field_json_size);
  rf_dur_ = json_double(rf_txt, "duration");
  rf_step_ = json_double(rf_txt, "step");

  {
    const std::string sj(
        cppannote::embedded_community1::pipeline_snapshot_json,
        cppannote::embedded_community1::pipeline_snapshot_json_size);
    try_regex_double(sj, "segmentation\\.min_duration_off", min_off_);
    try_regex_double(sj, "segmentation\\.min_duration_on", min_on_);
    try_json_bool_field(sj, "embedding_exclude_overlap",
                        embedding_exclude_overlap_);
    try_regex_double(sj, "clustering\\.threshold", vbx_params_.threshold);
    try_regex_double(sj, "clustering\\.Fa", vbx_params_.Fa);
    try_regex_double(sj, "clustering\\.Fb", vbx_params_.Fb);
    {
      int lda_dim = vbx_params_.lda_dimension;
      if (try_regex_int(sj, "clustering\\.lda_dimension", lda_dim)) {
        vbx_params_.lda_dimension = lda_dim;
      }
    }
  }

  golden_bounds_body_ = std::string(
      cppannote::embedded_community1::golden_speaker_bounds_json,
      cppannote::embedded_community1::golden_speaker_bounds_json_size);

  {
    const std::string emb_json(
        cppannote::embedded_community1::embedding_json,
        cppannote::embedded_community1::embedding_json_size);
    embed_sr_ = static_cast<int>(json_double(emb_json, "sample_rate"));
    embed_mel_bins_ = static_cast<int>(json_double(emb_json, "num_mel_bins"));
    embed_frame_length_ms_ =
        static_cast<float>(json_double(emb_json, "frame_length_ms"));
    embed_frame_shift_ms_ =
        static_cast<float>(json_double(emb_json, "frame_shift_ms"));
    embed_dim_ = static_cast<int>(json_double(emb_json, "embedding_dim"));
    embed_inputs_fbank_then_weights_ =
        embedding_ort::embedding_json_inputs_fbank_first(emb_json);
    embed_session_ =
        make_embedding_session(ort_env_, session_options_, embedding_onnx_path);
    min_num_samples_ = embedding_ort::discover_min_num_samples_embedding(
        *embed_session_, mem_, alloc_, embed_inputs_fbank_then_weights_,
        embed_sr_, embed_mel_bins_, embed_frame_length_ms_,
        embed_frame_shift_ms_, embed_dim_);
    plda_model_ = std::make_unique<plda_vbx::PldaModel>();
    plda_model_->load_from_arrays(cppannote::embedded_community1::xvec_mean1,
                                  cppannote::embedded_community1::kEmbeddingDim,
                                  cppannote::embedded_community1::xvec_mean2,
                                  cppannote::embedded_community1::kLdaOutDim,
                                  cppannote::embedded_community1::xvec_lda,
                                  cppannote::embedded_community1::kEmbeddingDim,
                                  cppannote::embedded_community1::kLdaOutDim,
                                  cppannote::embedded_community1::plda_mu,
                                  cppannote::embedded_community1::kPldaDim,
                                  cppannote::embedded_community1::plda_tr,
                                  cppannote::embedded_community1::kPldaDim,
                                  cppannote::embedded_community1::plda_psi,
                                  cppannote::embedded_community1::kPldaDim,
                                  vbx_params_.lda_dimension);
    vbx_params_.min_clusters = 1;
    vbx_params_.max_clusters = 1000000000;
    vbx_params_.num_clusters = -1;
  }
}

CppAnnoteEngine::CppAnnoteEngine()
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "cppannote"),
      session_options_{},
      session_(make_segmentation_session(ort_env_, session_options_, "")),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      alloc_{},
      in_name_(session_.GetInputNameAllocated(0, alloc_)),
      out_name_(session_.GetOutputNameAllocated(0, alloc_)) {
  init_config_and_models("");
}

CppAnnoteEngine::CppAnnoteEngine(const std::string &segmentation_onnx_path,
                                   const std::string &embedding_onnx_path)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "cppannote"),
      session_options_{},
      session_(make_segmentation_session(ort_env_, session_options_,
                                         segmentation_onnx_path)),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      alloc_{},
      in_name_(session_.GetInputNameAllocated(0, alloc_)),
      out_name_(session_.GetOutputNameAllocated(0, alloc_)) {
  init_config_and_models(embedding_onnx_path);
}

// ---------------------------------------------------------------------------
// Per-chunk building blocks
// ---------------------------------------------------------------------------

std::vector<float> CppAnnoteEngine::extract_chunk_audio(const float *audio,
                                                        int64_t num_samples,
                                                        int64_t offset,
                                                        int chunk_num_samples,
                                                        int num_channels) {
  std::vector<float> buf(static_cast<size_t>(num_channels) *
                             static_cast<size_t>(chunk_num_samples),
                         0.f);
  for (int ch = 0; ch < num_channels; ++ch) {
    for (int i = 0; i < chunk_num_samples; ++i) {
      const int64_t si = offset + i;
      if (si >= 0 && si < num_samples) {
        buf[static_cast<size_t>(ch) * static_cast<size_t>(chunk_num_samples) +
            static_cast<size_t>(i)] = audio[static_cast<size_t>(si)];
      }
    }
  }
  return buf;
}

std::vector<float> CppAnnoteEngine::run_segmentation_ort_single(
    const float *chunk_buf) {
  const int num_channels = cfg_.num_channels;
  const int chunk_num_samples = cfg_.chunk_num_samples;
  const std::array<int64_t, 3> in_shape{1, num_channels, chunk_num_samples};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem_, const_cast<float *>(chunk_buf),
      static_cast<size_t>(1 * num_channels * chunk_num_samples),
      in_shape.data(), in_shape.size());
  const char *in_names[] = {in_name_.get()};
  const char *out_names[] = {out_name_.get()};
  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                           out_names, 1);
  float *op = outs[0].GetTensorMutableData<float>();
  auto sh = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  if (sh.size() != 3 || sh[0] != 1) {
    throw std::runtime_error("unexpected ORT output rank");
  }
  const int F = static_cast<int>(sh[1]);
  const int K = static_cast<int>(sh[2]);
  if (seg_F_ == 0) {
    seg_F_ = F;
    seg_K_ = K;
  } else if (F != seg_F_ || K != seg_K_) {
    throw std::runtime_error("ORT output shape changed across chunks");
  }
  return std::vector<float>(op, op + F * K);
}

std::vector<float> CppAnnoteEngine::run_embedding_ort_single(
    const float *chunk_mono, const float *seg_binarized) {
  const int F = seg_F_;
  const int K = seg_K_;
  const int dim = embed_dim_;
  const int chunk_num_samples = cfg_.chunk_num_samples;
  const int sr_model = cfg_.sr_model;
  if (F <= 0 || K <= 0) {
    throw std::runtime_error(
        "run_embedding_ort_single: call run_segmentation_ort_single first");
  }

  std::vector<float> chunk_for_fbank(chunk_mono,
                                     chunk_mono + chunk_num_samples);
  int wav_sr_use = sr_model;
  if (sr_model != embed_sr_) {
    chunk_for_fbank =
        wav_pcm::linear_resample(chunk_for_fbank, sr_model, embed_sr_);
    wav_sr_use = embed_sr_;
  }
  const int chunk_embed_samples = static_cast<int>(chunk_for_fbank.size());
  const int min_nf_seg =
      embedding_exclude_overlap_
          ? static_cast<int>(std::ceil(
                static_cast<double>(F) * static_cast<double>(min_num_samples_) /
                static_cast<double>(chunk_embed_samples)))
          : -1;

  int Tf = 0;
  int Mfb = 0;
  std::vector<float> fbank_all;
  cppannote::fbank::wespeaker_like_fbank(
      static_cast<float>(wav_sr_use), embed_mel_bins_, embed_frame_length_ms_,
      embed_frame_shift_ms_, chunk_for_fbank.data(),
      static_cast<int>(chunk_for_fbank.size()), fbank_all, Tf, Mfb);
  if (Tf < 1 || Mfb != embed_mel_bins_) {
    throw std::runtime_error(
        "embedding fbank: unexpected frames or mel dimension");
  }

  std::vector<float> result(static_cast<size_t>(K) * static_cast<size_t>(dim));
  for (int sp = 0; sp < K; ++sp) {
    std::vector<float> clean_col(static_cast<size_t>(F), 0.f);
    std::vector<float> full_col(static_cast<size_t>(F), 0.f);
    for (int f = 0; f < F; ++f) {
      float rowsum = 0.f;
      for (int j = 0; j < K; ++j) {
        rowsum += seg_binarized[f * K + j];
      }
      const float overlap_ok = (rowsum < 2.f - 1e-5f) ? 1.f : 0.f;
      const float v = seg_binarized[f * K + sp];
      full_col[static_cast<size_t>(f)] = v;
      clean_col[static_cast<size_t>(f)] = v * overlap_ok;
    }
    float sum_clean = 0.f;
    for (int f = 0; f < F; ++f) {
      if (clean_col[static_cast<size_t>(f)] > 0.5f) {
        sum_clean += 1.f;
      }
    }
    const bool prefer_clean =
        (min_nf_seg < 0) ? true : (sum_clean > static_cast<float>(min_nf_seg));
    const std::vector<float> &src = prefer_clean ? clean_col : full_col;
    float *dst = &result[static_cast<size_t>(sp) * static_cast<size_t>(dim)];
    embedding_ort::run_embedding_ort(
        *embed_session_, mem_, alloc_, embed_inputs_fbank_then_weights_,
        fbank_all.data(), Tf, Mfb, src.data(), F, dst, dim);
  }
  return result;
}

std::vector<DiarizationTurn> CppAnnoteEngine::cluster_and_decode(
    const std::vector<float> &seg_out, const std::vector<float> &emb, int C,
    DiarizationProfile &profile, double chunk_step_sec_override) {
  using Clock = std::chrono::steady_clock;
  const int F = seg_F_;
  const int Kcls = seg_K_;
  const int dim = embed_dim_;
  const double chunk_step_sec =
      (chunk_step_sec_override > 0.0) ? chunk_step_sec_override : cfg_.chunk_step_sec;
  const double chunk_dur_sec = cfg_.chunk_dur_sec;
  if (F <= 0 || Kcls <= 0) {
    throw std::runtime_error(
        "cluster_and_decode: segmentation dimensions not yet discovered");
  }
  if (!cfg_.multilabel_export) {
    throw std::runtime_error(
        "CppAnnoteEngine expects export_includes_powerset_to_multilabel=true");
  }

  std::vector<float> binarized = seg_out;

  const auto t_vbx_start = Clock::now();

  std::vector<std::int8_t> hard_clusters_row;
  clustering_vbx::vbx_clustering_hard(*plda_model_, vbx_params_, C, F, Kcls,
                                      dim, emb.data(), binarized.data(),
                                      hard_clusters_row);
  const std::int8_t *hptr = hard_clusters_row.data();

  const auto t_after_vbx = Clock::now();

  double seg_ss = 0.;
  double seg_sd = chunk_dur_sec;
  double seg_st = chunk_step_sec;

  int Tcnt = 0;
  std::vector<std::uint8_t> cnt_u8 = speaker_count_initial_uint8(
      binarized, static_cast<size_t>(C), static_cast<size_t>(F),
      static_cast<size_t>(Kcls), seg_ss, seg_st, seg_sd, rf_dur_, rf_step_,
      Tcnt);
  std::uint8_t max_cnt = 0;
  for (std::uint8_t v : cnt_u8) {
    max_cnt = std::max(max_cnt, v);
  }
  if (max_cnt == 0) {
    profile.clustering_vbx_sec =
        std::chrono::duration<double>(t_after_vbx - t_vbx_start).count();
    profile.reconstruct_sec = 0.;
    profile.total_chunks = C;
    profile.num_frames = F;
    profile.num_classes = Kcls;
    return {};
  }

  int max_cap = INT_MAX;
  if (!golden_bounds_body_.empty()) {
    const std::string &bj = golden_bounds_body_;
    std::regex re("\"max_speakers\"\\s*:\\s*(null|[-+0-9]+)");
    std::smatch m;
    if (std::regex_search(bj, m, re) && m[1].str() != "null") {
      max_cap = std::stoi(m[1].str());
    }
  }
  std::vector<std::int8_t> count_i8 = cap_count(cnt_u8, max_cap);

  int max_cluster_id = -1;
  for (int ci = 0; ci < C; ++ci) {
    for (int j = 0; j < Kcls; ++j) {
      const int hv = static_cast<int>(hptr[static_cast<size_t>(ci * Kcls + j)]);
      if (hv >= 0) {
        max_cluster_id = std::max(max_cluster_id, hv);
      }
    }
  }
  if (max_cluster_id < 0) {
    throw std::runtime_error("hard_clusters: no cluster id >= 0");
  }
  const int num_detected_speakers = max_cluster_id + 1;
  for (size_t t = 0; t < count_i8.size(); ++t) {
    const int v = static_cast<int>(count_i8[t]);
    count_i8[t] = static_cast<std::int8_t>(
        std::max(0, std::min(v, num_detected_speakers)));
  }

  const double onset = 0.5;
  const double offset = 0.5;
  const double pad_onset = 0.0;
  const double pad_offset = 0.0;
  const bool apply_annotation_support =
      (min_off_ > 0.0 || pad_onset > 0.0 || pad_offset > 0.0);

  int K_di = 0;
  const std::vector<float> discrete =
      reconstruct_to_diarization(seg_out, C, F, Kcls, seg_ss, seg_sd, seg_st,
                                 hptr, count_i8, 0.0, rf_dur_, rf_step_, K_di);
  if (K_di <= 0) {
    throw std::runtime_error(
        "reconstruct returned non-positive speaker column count");
  }
  const int rows =
      static_cast<int>(discrete.size() / static_cast<size_t>(K_di));

  std::map<int, std::vector<cppannote::Segment>> by_label;
  for (int k = 0; k < K_di; ++k) {
    std::vector<float> col(static_cast<size_t>(rows));
    for (int t = 0; t < rows; ++t) {
      col[static_cast<size_t>(t)] =
          discrete[static_cast<size_t>(t) * static_cast<size_t>(K_di) +
                   static_cast<size_t>(k)];
    }
    std::vector<std::pair<double, double>> regs;
    binarize_column(col.data(), rows, 0.0, rf_dur_, rf_step_, onset, offset,
                    pad_onset, pad_offset, regs);
    if (!apply_annotation_support) {
      filter_min_duration_on(regs, min_on_);
    }
    for (const auto &pr : regs) {
      by_label[k].push_back(cppannote::Segment{pr.first, pr.second});
    }
  }

  std::vector<DiarizationTurn> turns;
  if (apply_annotation_support) {
    const auto merged = cppannote::annotation_support(by_label, min_off_);
    for (const auto &pr : merged) {
      const cppannote::Segment &seg = pr.second;
      if (seg.duration() < min_on_ - 1e-12) {
        continue;
      }
      turns.push_back({seg.start, seg.end, static_cast<int32_t>(pr.first)});
    }
  } else {
    for (int k = 0; k < K_di; ++k) {
      for (const cppannote::Segment &seg : by_label[k]) {
        turns.push_back({seg.start, seg.end, static_cast<int32_t>(k)});
      }
    }
  }
  std::sort(turns.begin(), turns.end(),
            [](const DiarizationTurn &a, const DiarizationTurn &b) {
              if (a.start != b.start) return a.start < b.start;
              if (a.end != b.end) return a.end < b.end;
              return a.speaker < b.speaker;
            });

  const auto t_end = Clock::now();
  profile.clustering_vbx_sec =
      std::chrono::duration<double>(t_after_vbx - t_vbx_start).count();
  profile.reconstruct_sec =
      std::chrono::duration<double>(t_end - t_after_vbx).count();
  profile.total_chunks = C;
  profile.num_frames = F;
  profile.num_classes = Kcls;
  return turns;
}

// ---------------------------------------------------------------------------
// DiarizationResults::write_json
// ---------------------------------------------------------------------------

void DiarizationResults::write_json(std::ostream &os) const {
  os << std::setprecision(17);
  os << "{\n";
  os << "  \"turns\": [\n";
  for (size_t i = 0; i < turns.size(); ++i) {
    const DiarizationTurn &t = turns[i];
    os << "    {\"start\": " << t.start << ", \"end\": " << t.end
       << ", \"speaker\": " << t.speaker << "}";
    if (i + 1 < turns.size()) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n}\n";
}

void DiarizationResults::write_json(const std::string &path) const {
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("DiarizationResults::write_json: open failed: " +
                             path);
  }
  write_json(f);
}

// ---------------------------------------------------------------------------
// CppAnnote pimpl: Impl definition + forwarding methods
// ---------------------------------------------------------------------------

struct CppAnnote::Impl {
  CppAnnoteEngine engine;
  std::map<int32_t, std::unique_ptr<StreamingDiarizationSession>> streams;
  std::map<int32_t, StreamingDiarizationConfig> stream_configs;
  int32_t next_stream_id = 1;

  Impl() : engine() {}
  Impl(const std::string& seg_path, const std::string& emb_path)
      : engine(seg_path, emb_path) {}

  StreamingDiarizationSession &get_stream(int32_t id) {
    auto it = streams.find(id);
    if (it == streams.end()) {
      throw std::runtime_error("CppAnnote: invalid stream_id " +
                               std::to_string(id));
    }
    return *it->second;
  }

  static DiarizationResults to_results(
      const StreamingDiarizationSnapshot &snap) {
    DiarizationResults r;
    r.turns.reserve(snap.turns.size());
    for (const auto &t : snap.turns) {
      r.turns.push_back({t.start, t.end, t.speaker});
    }
    return r;
  }
};

CppAnnote::CppAnnote() : impl_(std::make_unique<Impl>()) {}

CppAnnote::CppAnnote(const std::string& segmentation_onnx_path,
                      const std::string& embedding_onnx_path)
    : impl_(std::make_unique<Impl>(segmentation_onnx_path,
                                   embedding_onnx_path)) {}

CppAnnote::~CppAnnote() = default;
CppAnnote::CppAnnote(CppAnnote &&) noexcept = default;
CppAnnote &CppAnnote::operator=(CppAnnote &&) noexcept = default;

int32_t CppAnnote::create_stream(double cluster_cadence,
                                 double analyze_cadence) {
  const int32_t id = impl_->next_stream_id++;
  StreamingDiarizationConfig cfg;
  cfg.cluster_cadence = cluster_cadence;
  cfg.analyze_cadence = analyze_cadence;
  impl_->stream_configs[id] = cfg;
  impl_->streams[id] =
      std::make_unique<StreamingDiarizationSession>(impl_->engine, cfg);
  return id;
}

void CppAnnote::free_stream(int32_t stream_id) {
  impl_->streams.erase(stream_id);
  impl_->stream_configs.erase(stream_id);
}

void CppAnnote::start_stream(int32_t stream_id) {
  impl_->get_stream(stream_id).start_session();
}

DiarizationResults CppAnnote::stop_stream(int32_t stream_id) {
  auto snap = impl_->get_stream(stream_id).end_session();
  return Impl::to_results(snap);
}

void CppAnnote::add_audio_to_stream(int32_t stream_id, const float *audio_data,
                                    uint64_t audio_length,
                                    int32_t sample_rate) {
  impl_->get_stream(stream_id).add_audio_chunk(
      audio_data, static_cast<std::size_t>(audio_length),
      static_cast<int>(sample_rate));
}

DiarizationResults CppAnnote::diarize(const float *audio_data,
                                      uint64_t audio_length,
                                      int32_t sample_rate) {
  constexpr double kNeverRefresh = 1e18;
  StreamingDiarizationConfig cfg;
  cfg.cluster_cadence = kNeverRefresh;
  StreamingDiarizationSession sess(impl_->engine, cfg);
  sess.start_session();
  sess.add_audio_chunk(audio_data, static_cast<std::size_t>(audio_length),
                       static_cast<int>(sample_rate));
  auto snap = sess.end_session();
  return Impl::to_results(snap);
}

DiarizationResults CppAnnote::diarize_stream(int32_t stream_id) {
  auto snap = impl_->get_stream(stream_id).refresh_and_snapshot();
  return Impl::to_results(snap);
}

}  // namespace cppannote
