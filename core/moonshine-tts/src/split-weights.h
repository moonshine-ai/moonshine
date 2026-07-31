#ifndef MOONSHINE_TTS_SPLIT_WEIGHTS_H
#define MOONSHINE_TTS_SPLIT_WEIGHTS_H

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace moonshine_tts {

/// A float32 weight tensor that lives outside its model and is supplied as a
/// graph input on every inference.
///
/// An ORT-format model has its graph optimizations baked in at conversion time
/// and ORT does not re-apply them at load (``inference_session.cc`` registers
/// graph transformers only when not loading ORT format). For a model whose
/// weights are stored as int8 and dequantized by a ``Cast -> Mul -> Add``
/// chain, that forces a choice between folding the chain at conversion and
/// storing float32 weights (~4x the file) or running the chain on every
/// inference. Splitting the model avoids both: ``<stem>.model.ort`` is the
/// fused graph with its weights declared as inputs and carries no weight data,
/// and ``<stem>.weights.ort`` holds the int8 data plus the dequantize chains
/// and is run exactly once, at load.
///
/// See ``scripts/split-model-weights.py``, which produces the pair.
struct SplitWeight {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// Sequence length below which a split model's MatMul falls off a fast path.
///
/// ORT rearranges a constant MatMul operand into a blocked layout when it loads
/// the model, and cannot do so once that operand is a graph input. Above this
/// many rows it packs the operand per call and amortises the cost; below it the
/// unpacked kernel is used and inference costs several times more, worsening as
/// the sequence grows until the threshold is crossed. Callers that feed short
/// sequences should pad up to this length and mask the padding off.
inline constexpr int64_t kSplitWeightsMinSequenceLength = 32;

/// Runs a ``<stem>.weights.ort`` model once and returns its outputs.
///
/// The weights session (and the int8 data it holds) is released before
/// returning, so only the float32 results stay resident.
std::vector<SplitWeight> run_split_weights_model(
    Ort::Env& env, const std::filesystem::path& weights_path,
    const Ort::SessionOptions& session_options);

/// Memory-backed overload, for assets loaded from a bundle rather than disk.
std::vector<SplitWeight> run_split_weights_model(
    Ort::Env& env, const void* data, size_t length,
    const Ort::SessionOptions& session_options);

/// Appends each weight to *inputs* / *input_names* as a tensor view.
///
/// The tensors alias the buffers in *weights*, so no data is copied and the
/// caller must keep *weights* alive for the duration of the run.
void append_split_weight_inputs(const std::vector<SplitWeight>& weights,
                                const Ort::MemoryInfo& mem,
                                std::vector<Ort::Value>& inputs,
                                std::vector<const char*>& input_names);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_SPLIT_WEIGHTS_H
