// TFLM-based on-device VAD inference. See vad.h for the contract.
//
// Op set is identical to the SpellingCNN's (the TinyVadCNN reuses the same
// MobileNetV2 blocks): PAD, DEPTHWISE_CONV_2D, CONV_2D, ADD, SUM,
// FULLY_CONNECTED, RESHAPE. If the model is re-exported with new ops, expand
// the MicroMutableOpResolver<7> below to match.

#include "vad/vad.h"

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace spelling {

namespace {

inline int8_t Saturate8(float v) {
  if (v >= 127.0f) return 127;
  if (v <= -128.0f) return -128;
  const long r = lroundf(v);
  return static_cast<int8_t>(r);
}

}  // namespace

struct Vad::Impl {
  tflite::MicroMutableOpResolver<8>* resolver = nullptr;
  tflite::MicroInterpreter* interp = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
};

Vad::Vad(const unsigned char* model_data, unsigned int /*model_size*/,
         uint8_t* tensor_arena, std::size_t tensor_arena_size,
         int expected_n_mels, int expected_window_frames,
         tflite::MicroProfilerInterface* profiler) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("VAD model schema version %d != supported %d",
                static_cast<int>(model->version()),
                static_cast<int>(TFLITE_SCHEMA_VERSION));
    while (true) { /* halt */
    }
  }

  constexpr std::size_t kStaticsReservation = 1024;
  if (tensor_arena_size < kStaticsReservation + 1024) {
    MicroPrintf("VAD tensor arena %zu bytes too small", tensor_arena_size);
    while (true) { /* halt */
    }
  }
  auto place_at = [&](std::size_t offset, std::size_t size) -> void* {
    const std::size_t aligned = (offset + 7u) & ~static_cast<std::size_t>(7u);
    if (aligned + size > kStaticsReservation) {
      MicroPrintf("VAD static reservation overflow at %zu+%zu", aligned, size);
      while (true) { /* halt */
      }
    }
    return tensor_arena + aligned;
  };

  std::size_t off = 0;
  impl_ = new (place_at(off, sizeof(Impl))) Impl();
  off += sizeof(Impl);
  impl_->resolver =
      new (place_at(off, sizeof(tflite::MicroMutableOpResolver<8>)))
          tflite::MicroMutableOpResolver<8>();
  off += sizeof(tflite::MicroMutableOpResolver<8>);

  // 7 ops for the int8 body + QUANTIZE for the head16 int8->int16 requantize
  // before the int16 head (head_conv / pool / FC). QUANTIZE is harmless/unused
  // for a pure-int8 model, so one resolver serves both export variants.
  if (impl_->resolver->AddConv2D() != kTfLiteOk ||
      impl_->resolver->AddDepthwiseConv2D() != kTfLiteOk ||
      impl_->resolver->AddPad() != kTfLiteOk ||
      impl_->resolver->AddAdd() != kTfLiteOk ||
      impl_->resolver->AddSum() != kTfLiteOk ||
      impl_->resolver->AddFullyConnected() != kTfLiteOk ||
      impl_->resolver->AddReshape() != kTfLiteOk ||
      impl_->resolver->AddQuantize() != kTfLiteOk) {
    MicroPrintf("VAD: failed to register required ops");
    while (true) { /* halt */
    }
  }

  uint8_t* working_arena = tensor_arena + kStaticsReservation;
  std::size_t working_size = tensor_arena_size - kStaticsReservation;

  impl_->interp = new (place_at(off, sizeof(tflite::MicroInterpreter)))
      tflite::MicroInterpreter(model, *impl_->resolver, working_arena,
                               working_size,
                               /*resource_variables=*/nullptr, profiler);

  if (impl_->interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("VAD AllocateTensors() failed -- arena too small?");
    while (true) { /* halt */
    }
  }
  arena_used_bytes_ = impl_->interp->arena_used_bytes();
  impl_->input = impl_->interp->input(0);
  impl_->output = impl_->interp->output(0);

  // Input is always int8 (the feature quantization path is unchanged). The
  // output logit is int8 for the pure-int8 model, or int16 for the head16
  // mixed-precision model (int16 head). Accept either.
  if (impl_->input->type != kTfLiteInt8) {
    MicroPrintf("VAD expected int8 input; got %d", impl_->input->type);
    while (true) { /* halt */
    }
  }
  if (impl_->output->type != kTfLiteInt8 &&
      impl_->output->type != kTfLiteInt16) {
    MicroPrintf("VAD expected int8/int16 output; got %d", impl_->output->type);
    while (true) { /* halt */
    }
  }
  const std::size_t in_count = static_cast<std::size_t>(expected_n_mels) *
                               static_cast<std::size_t>(expected_window_frames);
  if (impl_->input->bytes != in_count) {
    MicroPrintf("VAD input %u bytes; expected %u (%d x %d)",
                static_cast<unsigned>(impl_->input->bytes),
                static_cast<unsigned>(in_count), expected_n_mels,
                expected_window_frames);
    while (true) { /* halt */
    }
  }
  // Single logit: 1 byte (int8) or 2 bytes (int16).
  const std::size_t out_elem_bytes =
      (impl_->output->type == kTfLiteInt16) ? 2u : 1u;
  if (impl_->output->bytes != out_elem_bytes) {
    MicroPrintf("VAD output %u bytes; expected %u (single logit)",
                static_cast<unsigned>(impl_->output->bytes),
                static_cast<unsigned>(out_elem_bytes));
    while (true) { /* halt */
    }
  }

  input_quant_.scale = impl_->input->params.scale;
  input_quant_.zero_point = impl_->input->params.zero_point;
  output_quant_.scale = impl_->output->params.scale;
  output_quant_.zero_point = impl_->output->params.zero_point;
  input_count_ = in_count;

  // Borrow fp32 feature scratch from the bottom of the activation overlay,
  // shifting past the int8 input slot if they would alias (same reasoning as
  // classifier.cc).
  uint8_t* overlay = working_arena;
  const std::size_t feat_bytes = input_count_ * sizeof(float);
  const std::size_t in_bytes = input_count_;  // int8
  const std::size_t in_off = static_cast<std::size_t>(
      reinterpret_cast<uint8_t*>(impl_->input->data.int8) - overlay);
  std::size_t scratch_off = 0;
  const bool overlaps_input =
      (scratch_off < in_off + in_bytes) && (in_off < scratch_off + feat_bytes);
  if (overlaps_input) {
    scratch_off = (in_off + in_bytes + 15u) & ~static_cast<std::size_t>(15u);
  }
  if (scratch_off + feat_bytes > working_size) {
    MicroPrintf("VAD feature scratch overflows arena overlay");
    while (true) { /* halt */
    }
  }
  feature_scratch_ = reinterpret_cast<float*>(overlay + scratch_off);
}

float Vad::Predict(const float* features) const {
  const float inv_scale = 1.0f / input_quant_.scale;
  const int zp = input_quant_.zero_point;
  int8_t* in_buf = impl_->input->data.int8;
  for (std::size_t i = 0; i < input_count_; ++i) {
    in_buf[i] = Saturate8(features[i] * inv_scale + static_cast<float>(zp));
  }
  if (impl_->interp->Invoke() != kTfLiteOk) {
    MicroPrintf("VAD Invoke() failed");
    while (true) { /* halt */
    }
  }
  // Dequantize the single logit. int16 for the head16 model, int8 otherwise.
  const float raw = (impl_->output->type == kTfLiteInt16)
                        ? static_cast<float>(impl_->output->data.i16[0])
                        : static_cast<float>(impl_->output->data.int8[0]);
  const float logit = (raw - static_cast<float>(output_quant_.zero_point)) *
                      output_quant_.scale;
  return 1.0f / (1.0f + std::exp(-logit));  // sigmoid -> speech probability
}

}  // namespace spelling
