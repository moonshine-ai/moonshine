#include "split-weights.h"

#include <stdexcept>
#include <utility>

namespace moonshine_tts {

namespace {

std::vector<SplitWeight> collect_outputs(Ort::Session& weights_session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = weights_session.GetOutputCount();

  std::vector<std::string> names;
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    names.emplace_back(
        weights_session.GetOutputNameAllocated(i, allocator).get());
  }
  std::vector<const char*> name_ptrs;
  name_ptrs.reserve(count);
  for (const std::string& name : names) {
    name_ptrs.push_back(name.c_str());
  }

  Ort::RunOptions run_opts{nullptr};
  std::vector<Ort::Value> values = weights_session.Run(
      run_opts, nullptr, nullptr, 0, name_ptrs.data(), name_ptrs.size());
  if (values.size() != count) {
    throw std::runtime_error("split weights model returned " +
                             std::to_string(values.size()) + " of " +
                             std::to_string(count) + " tensors");
  }

  std::vector<SplitWeight> weights;
  weights.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto info = values[i].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("split weight " + names[i] + " is not float32");
    }
    const size_t elements = info.GetElementCount();
    const float* src = values[i].GetTensorData<float>();
    SplitWeight weight;
    weight.name = names[i];
    weight.shape = info.GetShape();
    weight.data.assign(src, src + elements);
    weights.push_back(std::move(weight));
  }
  return weights;
}

}  // namespace

std::vector<SplitWeight> run_split_weights_model(
    Ort::Env& env, const std::filesystem::path& weights_path,
    const Ort::SessionOptions& session_options) {
#ifdef _WIN32
  const std::wstring path = weights_path.wstring();
#else
  const std::string path = weights_path.string();
#endif
  Ort::Session weights_session(env, path.c_str(), session_options);
  return collect_outputs(weights_session);
}

std::vector<SplitWeight> run_split_weights_model(
    Ort::Env& env, const void* data, size_t length,
    const Ort::SessionOptions& session_options) {
  Ort::Session weights_session(env, data, length, session_options);
  return collect_outputs(weights_session);
}

void append_split_weight_inputs(const std::vector<SplitWeight>& weights,
                                const Ort::MemoryInfo& mem,
                                std::vector<Ort::Value>& inputs,
                                std::vector<const char*>& input_names) {
  for (const SplitWeight& weight : weights) {
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(weight.data.data()), weight.data.size(),
        weight.shape.data(), weight.shape.size()));
    input_names.push_back(weight.name.c_str());
  }
}

}  // namespace moonshine_tts
