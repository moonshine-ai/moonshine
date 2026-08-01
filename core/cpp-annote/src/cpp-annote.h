// SPDX-License-Identifier: MIT
// Public diarization API using the pointer-to-implementation (pimpl) pattern.
// All internal state (ONNX Runtime sessions, clustering parameters, etc.) is
// hidden behind CppAnnote::Impl, defined in the .cpp translation unit.

#ifndef CPP_ANNOTE_H_
#define CPP_ANNOTE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cppannote {

/// Where one model's bytes come from: a filesystem path, or a buffer the
/// caller owns. When `data` is non-null and `size` is non-zero the buffer is
/// used and `path` is ignored. The buffer is not copied — ONNX Runtime reads
/// out of it for the life of the session — so it must outlive the engine.
struct ModelSource {
  std::string path;
  const uint8_t *data = nullptr;
  size_t size = 0;

  bool empty() const {
    return path.empty() && (data == nullptr || size == 0);
  }
};

/// The two models the pipeline needs. Both are required: they used to be
/// compiled into the library, and are now downloaded (see
/// core/moonshine-model-catalog.h, `diarization_model_dependencies`).
struct ModelSources {
  ModelSource segmentation;
  ModelSource embedding;
};

struct DiarizationTurn {
  double start = 0.;
  double end = 0.;
  int32_t speaker = 0;
};

struct DiarizationResults {
  std::vector<DiarizationTurn> turns;

  void write_json(const std::string &path) const;
  void write_json(std::ostream &os) const;
};

/// Loads the segmentation and embedding ORT models and manages streaming
/// diarization sessions.  All heavy implementation details (ORT sessions, PLDA
/// model, VBx clustering) are hidden behind the pimpl firewall.
///
/// The clustering parameters are still compiled in (see
/// community1_cpp_annote_embedded.h), but the two ORT models are not: supply
/// them from the community-1 files, whichever way suits the platform.
class CppAnnote {
 public:
  /// Construct from files or from caller-owned buffers.  Throws
  /// std::runtime_error if either model is missing.
  explicit CppAnnote(const ModelSources& models);

  /// Convenience overload for the common file-based case.
  CppAnnote(const std::string& segmentation_model_path,
            const std::string& embedding_model_path);

  ~CppAnnote();

  CppAnnote(const CppAnnote &) = delete;
  CppAnnote &operator=(const CppAnnote &) = delete;
  CppAnnote(CppAnnote &&) noexcept;
  CppAnnote &operator=(CppAnnote &&) noexcept;

  /// Diarize an entire buffer of mono PCM audio in one shot.
  DiarizationResults diarize(const float *audio_data, uint64_t audio_length,
                             int32_t sample_rate = 16000);

  /// Allocate a new streaming diarization session and return its handle.
  /// ``cluster_cadence`` controls how often VBx re-clustering runs (seconds).
  /// ``analyze_cadence`` controls the step between segmentation+embedding model
  /// runs (seconds, must be >0 and <=10; 0 means use the model default).
  int32_t create_stream(double cluster_cadence = 2.0,
                        double analyze_cadence = 0.0);

  /// Release a stream and all associated resources.
  void free_stream(int32_t stream_id);

  /// Initialize a stream, clearing any buffered audio and cached results.
  void start_stream(int32_t stream_id);

  /// Finalize the stream (forces a last clustering pass) and return
  /// diarization.
  DiarizationResults stop_stream(int32_t stream_id);

  /// Append PCM audio to a stream.  Resampling to the model rate is handled
  /// internally; ``sample_rate`` is the rate of the supplied buffer.
  void add_audio_to_stream(int32_t stream_id, const float *audio_data,
                           uint64_t audio_length, int32_t sample_rate);

  /// Force a clustering refresh and return the current diarization snapshot
  /// without stopping the stream.
  DiarizationResults diarize_stream(int32_t stream_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cppannote

#endif  // CPP_ANNOTE_H_
