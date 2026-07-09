// TFLM-based on-device classifier. See classifier.h for the contract.
//
// Op set is locked to exactly what spelling_cnn.mel.int8.tflite uses
// today (see generate_embedded_data.py / the upstream export script):
//     PAD, DEPTHWISE_CONV_2D, CONV_2D, ADD, SUM, FULLY_CONNECTED, RESHAPE
// Anything not on that list will fail at AllocateTensors() time with a
// clear "Op not found" error. If the model is re-exported with new
// ops, expand `MicroMutableOpResolver<7>` below to match -- the
// template parameter is the EXACT op count, not a max.

#include <cstdint>
#include <cstring>

#include "stt/stt.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace spelling {

namespace {

// Saturating round-and-cast for the input-quantization step. Same formula as
// the host reference classifier so both agree at the int8 input boundary.
inline int8_t Saturate8(float v) {
  if (v >= 127.0f) return 127;
  if (v <= -128.0f) return -128;
  // lroundf is OK on -fno-exceptions builds; it has the same
  // round-half-away-from-zero semantics as Python's round() on int8
  // boundaries.
  const long r = lroundf(v);
  return static_cast<int8_t>(r);
}

}  // namespace

// The MicroInterpreter / op resolver are heavy objects we don't want
// to expose through the header. They live here behind an opaque Impl
// pointer that we allocate via placement-new directly into the
// user-supplied tensor arena: this keeps everything in a single
// contiguous block of memory, with no malloc on the data path.
struct Classifier::Impl {
  // The op resolver template parameter is the EXACT op count. Our
  // current model uses 7 distinct op codes (PAD, DEPTHWISE_CONV_2D,
  // CONV_2D, ADD, SUM, FULLY_CONNECTED, RESHAPE).
  tflite::MicroMutableOpResolver<7>* resolver = nullptr;
  tflite::MicroInterpreter* interp = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
};

Classifier::Classifier(const unsigned char* model_data,
                       unsigned int /*model_size*/, uint8_t* tensor_arena,
                       std::size_t tensor_arena_size, int expected_n_mels,
                       int expected_target_frames, int expected_n_classes,
                       tflite::MicroProfilerInterface* profiler) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d != supported %d",
                static_cast<int>(model->version()),
                static_cast<int>(TFLITE_SCHEMA_VERSION));
    while (true) { /* halt */
    }
  }

  // Place all heavy TFLM objects (op resolver, interpreter, our Impl)
  // inside the user-supplied tensor arena. TFLM does the same for its
  // own internal allocations, so co-locating ours keeps the working
  // set in one contiguous chunk and avoids any hidden malloc.
  //
  // We reserve the FIRST few hundred bytes of the arena for these
  // statics, then hand the rest to MicroInterpreter as its working
  // arena. The reservation is intentionally generous (1 KB); the
  // interpreter reports the true arena usage via arena_used_bytes().
  constexpr std::size_t kStaticsReservation = 1024;
  if (tensor_arena_size < kStaticsReservation + 1024) {
    MicroPrintf("Tensor arena %zu bytes too small (need >= %zu)",
                tensor_arena_size, kStaticsReservation + 1024);
    while (true) { /* halt */
    }
  }
  auto place_at = [&](std::size_t offset, std::size_t size) -> void* {
    // 8-byte-align manually; TFLM's interpreter prefers aligned starts.
    const std::size_t aligned = (offset + 7u) & ~static_cast<std::size_t>(7u);
    if (aligned + size > kStaticsReservation) {
      MicroPrintf("Static reservation overflow at offset %zu+%zu", aligned,
                  size);
      while (true) { /* halt */
      }
    }
    return tensor_arena + aligned;
  };

  // Roll our own placement-new with the tiny offset bookkeeping.
  std::size_t off = 0;
  impl_ = new (place_at(off, sizeof(Impl))) Impl();
  off += sizeof(Impl);
  impl_->resolver =
      new (place_at(off, sizeof(tflite::MicroMutableOpResolver<7>)))
          tflite::MicroMutableOpResolver<7>();
  off += sizeof(tflite::MicroMutableOpResolver<7>);

  // Register exactly the ops the model uses. The order matches the
  // model's op-code table; mismatches would surface as "Op not found".
  if (impl_->resolver->AddConv2D() != kTfLiteOk ||
      impl_->resolver->AddDepthwiseConv2D() != kTfLiteOk ||
      impl_->resolver->AddPad() != kTfLiteOk ||
      impl_->resolver->AddAdd() != kTfLiteOk ||
      impl_->resolver->AddSum() != kTfLiteOk ||
      impl_->resolver->AddFullyConnected() != kTfLiteOk ||
      impl_->resolver->AddReshape() != kTfLiteOk) {
    MicroPrintf("Failed to register required ops");
    while (true) { /* halt */
    }
  }

  uint8_t* working_arena = tensor_arena + kStaticsReservation;
  std::size_t working_size = tensor_arena_size - kStaticsReservation;

  // The 5th ctor arg is MicroResourceVariables* (we have none); the 6th
  // is the optional profiler. Passing it here makes the interpreter wrap
  // every op node in a ScopedMicroProfiler.
  impl_->interp = new (place_at(off, sizeof(tflite::MicroInterpreter)))
      tflite::MicroInterpreter(model, *impl_->resolver, working_arena,
                               working_size,
                               /*resource_variables=*/nullptr, profiler);

  if (impl_->interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf(
        "AllocateTensors() failed -- arena too small? "
        "Try bumping kTensorArenaSize in main.cc.");
    while (true) { /* halt */
    }
  }
  arena_used_bytes_ = impl_->interp->arena_used_bytes();

  impl_->input = impl_->interp->input(0);
  impl_->output = impl_->interp->output(0);

  // Shape sanity: catch model/code drift up front instead of silently
  // running on the wrong tensor layout.
  if (impl_->input->type != kTfLiteInt8 || impl_->output->type != kTfLiteInt8) {
    MicroPrintf("Expected int8 I/O; got input=%d output=%d", impl_->input->type,
                impl_->output->type);
    while (true) { /* halt */
    }
  }
  const std::size_t in_count = static_cast<std::size_t>(expected_n_mels) *
                               static_cast<std::size_t>(expected_target_frames);
  // input->bytes is the int8 element count for int8 tensors.
  if (impl_->input->bytes != in_count) {
    MicroPrintf("Input tensor has %u bytes; expected %u (%d x %d)",
                static_cast<unsigned>(impl_->input->bytes),
                static_cast<unsigned>(in_count), expected_n_mels,
                expected_target_frames);
    while (true) { /* halt */
    }
  }
  if (impl_->output->bytes != static_cast<std::size_t>(expected_n_classes)) {
    MicroPrintf("Output tensor has %u bytes; expected %d",
                static_cast<unsigned>(impl_->output->bytes),
                expected_n_classes);
    while (true) { /* halt */
    }
  }

  // Cache quantization parameters and counts so Run() is hot-path free.
  input_quant_.scale = impl_->input->params.scale;
  input_quant_.zero_point = impl_->input->params.zero_point;
  output_quant_.scale = impl_->output->params.scale;
  output_quant_.zero_point = impl_->output->params.zero_point;
  input_count_ = in_count;
  n_classes_ = expected_n_classes;

  // Borrow the fp32 feature scratch from TFLM's activation overlay.
  //
  // The log-mel features (input_count_ floats) are a transient: produced
  // by the front-end, consumed by Run()'s quantize loop, then dead the
  // instant Invoke() starts. A dedicated static buffer for them would
  // sit idle through every inference. Instead we reuse the arena.
  //
  // Verified layout of this vendored TFLM (single_arena_buffer_allocator):
  //   * The activation overlay (the "resizable buffer") begins at the
  //     LOW end of the working arena (working_arena, offset 0) and grows
  //     upward. Every planned activation tensor -- including our int8
  //     input -- lives somewhere in [0, head_high_water).
  //   * Persistent allocations (interpreter bookkeeping, quant params,
  //     scratch-buffer handles) grow DOWNWARD from the HIGH end (the
  //     tail). They sit far above anything we touch here.
  //   * Nothing in the overlay is valid until Invoke() executes op 0;
  //     before that it is uninitialised scratch we may freely use.
  //
  // We therefore place the feature scratch at the very bottom of the
  // overlay. The single hazard is aliasing the int8 input tensor's own
  // slot: Run() reads the fp32 scratch while writing the int8 input, so
  // an overlap would corrupt features mid-quantize. If the input lands
  // in the bottom feat_bytes, we shift the scratch up to just past it.
  // Either placement stays in the low overlay (tens of KB at most), far
  // below the persistent tail, and is harmlessly overwritten by
  // activations once Invoke() runs.
  uint8_t* overlay = working_arena;
  const std::size_t feat_bytes = input_count_ * sizeof(float);
  const std::size_t in_bytes = input_count_;  // int8 -> 1 byte/elem
  const std::size_t in_off = static_cast<std::size_t>(
      reinterpret_cast<uint8_t*>(impl_->input->data.int8) - overlay);

  std::size_t scratch_off = 0;
  const bool overlaps_input =
      (scratch_off < in_off + in_bytes) && (in_off < scratch_off + feat_bytes);
  if (overlaps_input) {
    // 16-byte-align the shifted start so the fp32 view stays aligned.
    scratch_off = (in_off + in_bytes + 15u) & ~static_cast<std::size_t>(15u);
  }
  if (scratch_off + feat_bytes > working_size) {
    MicroPrintf("Feature scratch (%u B @ off %u) overflows arena overlay %u",
                static_cast<unsigned>(feat_bytes),
                static_cast<unsigned>(scratch_off),
                static_cast<unsigned>(working_size));
    while (true) { /* halt */
    }
  }
  feature_scratch_ = reinterpret_cast<float*>(overlay + scratch_off);
}

void Classifier::Run(const float* features, float* logits_out) const {
  // 1. Quantize fp32 -> int8 directly into the model's input tensor.
  const float inv_scale = 1.0f / input_quant_.scale;
  const int zp = input_quant_.zero_point;
  int8_t* in_buf = impl_->input->data.int8;
  for (std::size_t i = 0; i < input_count_; ++i) {
    in_buf[i] = Saturate8(features[i] * inv_scale + static_cast<float>(zp));
  }

  // 2. Run inference. If this returns non-OK something's gone very
  //    wrong (the arena was sized at AllocateTensors() time and we
  //    haven't reshaped anything), so we halt loudly.
  if (impl_->interp->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    while (true) { /* halt */
    }
  }

  // 3. Dequantize int8 logits -> fp32.
  const float out_scale = output_quant_.scale;
  const int out_zp = output_quant_.zero_point;
  const int8_t* out_buf = impl_->output->data.int8;
  for (int i = 0; i < n_classes_; ++i) {
    logits_out[i] =
        (static_cast<float>(out_buf[i]) - static_cast<float>(out_zp)) *
        out_scale;
  }
}

}  // namespace spelling
