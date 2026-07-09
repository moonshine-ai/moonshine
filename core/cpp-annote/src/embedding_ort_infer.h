// SPDX-License-Identifier: MIT
// ONNX embedding runtime helpers shared by ``cpp-annote.cpp`` and
// ``embedding_golden_test``.

#ifndef EMBEDDING_ORT_INFER_H_
#define EMBEDDING_ORT_INFER_H_

#include <onnxruntime_cxx_api.h>

#include <string>

namespace cppannote::embedding_ort {

bool embedding_json_inputs_fbank_first(const std::string& emb_json);

void run_embedding_ort(Ort::Session& sess, Ort::MemoryInfo& mem,
                       Ort::AllocatorWithDefaultOptions& alloc,
                       bool fbank_first, const float* fbank_rowmajor,
                       int fbank_num_frames, int M, const float* weights,
                       int weight_num_frames, float* out, int dim);

int discover_min_num_samples_embedding(Ort::Session& sess, Ort::MemoryInfo& mem,
                                       Ort::AllocatorWithDefaultOptions& alloc,
                                       bool fbank_first, int embed_sr,
                                       int mel_bins, float fl_ms, float fs_ms,
                                       int embed_dim);

int fbank_num_frames_for_samples(int embed_sr, int mel_bins, float fl_ms,
                                 float fs_ms, int num_samples);

int seg_to_fbank_nearest_index(int tf, int num_seg_frames,
                               int num_fbank_frames);

}  // namespace cppannote::embedding_ort

#endif  // EMBEDDING_ORT_INFER_H_
