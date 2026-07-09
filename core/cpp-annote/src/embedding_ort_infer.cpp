// SPDX-License-Identifier: MIT

#include "embedding_ort_infer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "compute_fbank.h"

namespace cppannote::embedding_ort {
namespace {

bool any_non_finite_embedding(const float* e, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(e[i])) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool embedding_json_inputs_fbank_first(const std::string& emb_json) {
  const std::size_t pos = emb_json.find("\"input_names\"");
  if (pos == std::string::npos) {
    return true;
  }
  const std::size_t f = emb_json.find("\"fbank\"", pos);
  const std::size_t w = emb_json.find("\"weights\"", pos);
  if (f == std::string::npos || w == std::string::npos) {
    return true;
  }
  return f < w;
}

void run_embedding_ort(Ort::Session& sess, Ort::MemoryInfo& mem,
                       Ort::AllocatorWithDefaultOptions& alloc,
                       bool fbank_first, const float* fbank_rowmajor,
                       int fbank_num_frames, int M, const float* weights,
                       int weight_num_frames, float* out, int dim) {
  Ort::AllocatedStringPtr in0 = sess.GetInputNameAllocated(0, alloc);
  Ort::AllocatedStringPtr in1 = sess.GetInputNameAllocated(1, alloc);
  Ort::AllocatedStringPtr on0 = sess.GetOutputNameAllocated(0, alloc);
  std::array<int64_t, 3> shf{1, static_cast<int64_t>(fbank_num_frames),
                             static_cast<int64_t>(M)};
  std::array<int64_t, 2> shw{1, static_cast<int64_t>(weight_num_frames)};
  Ort::Value fb = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(fbank_rowmajor),
      static_cast<size_t>(fbank_num_frames) * static_cast<size_t>(M),
      shf.data(), shf.size());
  Ort::Value wt = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(weights), static_cast<size_t>(weight_num_frames),
      shw.data(), shw.size());
  Ort::Value inputs[2];
  const char* in_names[2];
  if (fbank_first) {
    inputs[0] = std::move(fb);
    inputs[1] = std::move(wt);
    in_names[0] = in0.get();
    in_names[1] = in1.get();
  } else {
    inputs[0] = std::move(wt);
    inputs[1] = std::move(fb);
    in_names[0] = in0.get();
    in_names[1] = in1.get();
  }
  const char* out_names[] = {on0.get()};
  auto outs =
      sess.Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);
  float* op = outs[0].GetTensorMutableData<float>();
  std::memcpy(out, op, static_cast<size_t>(dim) * sizeof(float));
}

int discover_min_num_samples_embedding(Ort::Session& sess, Ort::MemoryInfo& mem,
                                       Ort::AllocatorWithDefaultOptions& alloc,
                                       bool fbank_first, int embed_sr,
                                       int mel_bins, float fl_ms, float fs_ms,
                                       int embed_dim) {
  int lower = 2;
  int upper = std::max(3, embed_sr / 2);
  while (lower + 1 < upper) {
    const int middle = (lower + upper) / 2;
    std::vector<float> noise(static_cast<size_t>(middle));
    for (int i = 0; i < middle; ++i) {
      noise[static_cast<size_t>(i)] =
          (static_cast<float>((i * 7919) % 13) - 6.f) * 0.01f;
    }
    int Tf = 0;
    int Mdim = 0;
    std::vector<float> fb;
    cppannote::fbank::wespeaker_like_fbank(static_cast<float>(embed_sr),
                                           mel_bins, fl_ms, fs_ms, noise.data(),
                                           middle, fb, Tf, Mdim);
    if (Tf < 1 || Mdim < 1) {
      lower = middle;
      continue;
    }
    std::vector<float> wts(static_cast<size_t>(Tf), 1.f);
    std::vector<float> emb(static_cast<size_t>(embed_dim));
    run_embedding_ort(sess, mem, alloc, fbank_first, fb.data(), Tf, Mdim,
                      wts.data(), Tf, emb.data(), embed_dim);
    if (any_non_finite_embedding(emb.data(), embed_dim)) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return upper;
}

int fbank_num_frames_for_samples(int embed_sr, int mel_bins, float fl_ms,
                                 float fs_ms, int num_samples) {
  if (num_samples < 1) {
    return 0;
  }
  std::vector<float> noise(static_cast<size_t>(num_samples), 0.01f);
  int Tf = 0;
  int Mdim = 0;
  std::vector<float> fb;
  cppannote::fbank::wespeaker_like_fbank(static_cast<float>(embed_sr), mel_bins,
                                         fl_ms, fs_ms, noise.data(),
                                         num_samples, fb, Tf, Mdim);
  return Tf;
}

int seg_to_fbank_nearest_index(int tf, int num_seg_frames,
                               int num_fbank_frames) {
  if (num_fbank_frames <= 0 || num_seg_frames <= 0) {
    return 0;
  }
  // Match ``F.interpolate(..., mode="nearest")`` on ``(B,1,F)`` → length
  // ``Tf``:
  // ``at::native::nearest_neighbor_compute_source_index`` (PyTorch / OpenCV
  // BC).
  const float scale =
      static_cast<float>(num_seg_frames) / static_cast<float>(num_fbank_frames);
  int si = static_cast<int>(std::floor(static_cast<float>(tf) * scale));
  if (si < 0) {
    si = 0;
  }
  if (si >= num_seg_frames) {
    si = num_seg_frames - 1;
  }
  return si;
}

}  // namespace cppannote::embedding_ort
