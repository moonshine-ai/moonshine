// Custom ONNX Runtime operators for the ZipVoice Zipformer (domain ai.zipvoice).
//
// Ported from the standalone ``custom_ops/swoosh_op.cc`` in the ZipVoice repo. The only structural
// change is that these kernels are compiled *into* libmoonshine (which already links onnxruntime)
// and registered on a session directly via ``zipvoice_register_custom_ops`` rather than being built
// as a separate shared library that ORT ``dlopen``s at runtime. This avoids loading a .so/.dll,
// which some platforms (e.g. iOS) make difficult.
//
// SwooshL / SwooshR: the Zipformer activations. Exported to ONNX they decompose into ~11
// element-wise nodes each; these ops collapse each activation into a single kernel.
//   softplus(z) = log(1 + exp(z)) = max(z, 0) + log1p(exp(-|z|))
//   SwooshL(x)  = softplus(x - 4.0) - 0.08 * x - 0.035
//   SwooshR(x)  = softplus(x - 1.0) - 0.08 * x - 0.313261687
// GluGate: out[..., c] = x[..., c] * sigmoid(x[..., C + c]),  C = lastdim / 2
// DepthwiseConv1d: fuses Transpose -> depthwise Conv1d ('same' pad K//2) -> Transpose in [T,N,C].
// BiasNorm: RMS-style norm of (x - bias) over channels, scaled by exp(log_scale).
// Bypass: per-channel residual lerp src_orig + (src - src_orig) * scale.

#include "zipvoice-custom-ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace moonshine_tts {

namespace {

constexpr float kSlope = 0.08f;
constexpr float kLeftOffset = 4.0f;
constexpr float kLeftBias = 0.035f;
constexpr float kRightOffset = 1.0f;
constexpr float kRightBias = 0.313261687f;

// Number of elements processed per ParallelFor task.
constexpr size_t kTile = 1024;

// softplus(z) = max(z,0) + g(|z|),  g(u) = log1p(exp(-u)); g approximated by a degree-8 polynomial
// on [0, kSoftplusU], truncated to 0 outside. Pure FMA (vectorizes on NEON/AVX), ~3.4e-4 abs error.
constexpr float kSoftplusU = 8.0f;
constexpr float kSoftplusC[9] = {
    6.93208957e-01f, -5.00340649e-01f, 1.24495565e-01f,
    3.08191795e-03f, -9.28873378e-03f, 2.42859460e-03f,
    -3.09979591e-04f, 2.03891086e-05f, -5.52490641e-07f};

inline float SoftplusPoly(float z) {
  const float a = std::fabs(z);
  const float u = a < kSoftplusU ? a : kSoftplusU;
  const float* c = kSoftplusC;
  float corr = c[8];
  corr = corr * u + c[7];
  corr = corr * u + c[6];
  corr = corr * u + c[5];
  corr = corr * u + c[4];
  corr = corr * u + c[3];
  corr = corr * u + c[2];
  corr = corr * u + c[1];
  corr = corr * u + c[0];
  if (a >= kSoftplusU) corr = 0.0f;
  const float m = z > 0.0f ? z : 0.0f;
  return m + corr;
}

struct SwooshJob {
  const float* in;
  float* out;
  size_t count;
  float offset;
  float bias;
};

void ComputeTile(void* user_data, size_t idx) {
  const SwooshJob& job = *static_cast<const SwooshJob*>(user_data);
  const size_t start = idx * kTile;
  if (start >= job.count) {
    return;
  }
  const size_t end = std::min(start + kTile, job.count);
  const int n = static_cast<int>(end - start);
  const float* in = job.in + start;
  float* out = job.out + start;

  for (int i = 0; i < n; ++i) {
    const float x = in[i];
    out[i] = SoftplusPoly(x - job.offset) - kSlope * x - job.bias;
  }
}

struct SwooshKernel {
  explicit SwooshKernel(bool is_left)
      : offset_(is_left ? kLeftOffset : kRightOffset),
        bias_(is_left ? kLeftBias : kRightBias) {}

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    Ort::ConstValue input = ctx.GetInput(0);
    auto info = input.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const size_t count = info.GetElementCount();

    Ort::UnownedValue output = ctx.GetOutput(0, shape);
    SwooshJob job{input.GetTensorData<float>(),
                  output.GetTensorMutableData<float>(), count, offset_, bias_};

    if (count == 0) {
      return;
    }
    const size_t num_tiles = (count + kTile - 1) / kTile;
    ctx.ParallelFor(ComputeTile, num_tiles, /*num_batch=*/0, &job);
  }

 private:
  float offset_;
  float bias_;
};

inline float SigmoidScalar(float v) { return 1.0f / (1.0f + std::exp(-v)); }

struct GluJob {
  const float* in;  // [rows, 2C]
  float* out;       // [rows, C]
  size_t half;      // C
};

void ComputeGluRow(void* user_data, size_t r) {
  const GluJob& job = *static_cast<const GluJob*>(user_data);
  const size_t C = job.half;
  const float* val = job.in + r * 2 * C;
  const float* gate = val + C;
  float* out = job.out + r * C;

  // Portable scalar sigmoid (tiled + threaded by the caller). The Accelerate ``vvexpf`` fast path from
  // the standalone custom-op library is intentionally dropped here so libmoonshine needs no Accelerate
  // framework dependency; the difference is a small GluGate micro-optimization.
  for (size_t c = 0; c < C; ++c) out[c] = val[c] * SigmoidScalar(gate[c]);
}

struct GluKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    Ort::ConstValue input = ctx.GetInput(0);
    auto info = input.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    const size_t count = info.GetElementCount();
    if (shape.empty()) {
      ORT_CXX_API_THROW("GluGate expects a tensor with rank >= 1",
                        ORT_INVALID_ARGUMENT);
    }
    const int64_t last = shape.back();
    if (last % 2 != 0) {
      ORT_CXX_API_THROW("GluGate last dim must be even", ORT_INVALID_ARGUMENT);
    }
    const size_t half = static_cast<size_t>(last / 2);
    shape.back() = static_cast<int64_t>(half);

    Ort::UnownedValue output = ctx.GetOutput(0, shape);
    if (count == 0 || half == 0) {
      return;
    }
    const size_t rows = count / (2 * half);
    GluJob job{input.GetTensorData<float>(),
               output.GetTensorMutableData<float>(), half};
    ctx.ParallelFor(ComputeGluRow, rows, /*num_batch=*/0, &job);
  }
};

struct DwJob {
  const float* x;        // [T, N, C]
  const float* wpacked;  // [K, C]: wpacked[j*C + c] = weight[c, 0, j]
  const float* bias;     // [C]
  float* out;            // [T, N, C]
  int64_t T, N, C, K, pad;
};

void ComputeConvRow(void* user_data, size_t row) {
  const DwJob& job = *static_cast<const DwJob*>(user_data);
  const int64_t C = job.C, N = job.N, T = job.T, K = job.K, pad = job.pad;
  const int64_t t = static_cast<int64_t>(row) / N;
  const int64_t n = static_cast<int64_t>(row) % N;
  float* o = job.out + (t * N + n) * C;
  const float* bias = job.bias;
  for (int64_t c = 0; c < C; ++c) o[c] = bias[c];

  int64_t jlo = pad - t;
  if (jlo < 0) jlo = 0;
  int64_t jhi = T - t + pad;  // exclusive; ti = t+j-pad < T  => j < T-t+pad
  if (jhi > K) jhi = K;
  for (int64_t j = jlo; j < jhi; ++j) {
    const float* xrow = job.x + ((t + j - pad) * N + n) * C;
    const float* wj = job.wpacked + j * C;
    for (int64_t c = 0; c < C; ++c) o[c] += xrow[c] * wj[c];
  }
}

struct DepthwiseConvKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    Ort::ConstValue x = ctx.GetInput(0);
    Ort::ConstValue w = ctx.GetInput(1);
    Ort::ConstValue b = ctx.GetInput(2);

    const std::vector<int64_t> xshape = x.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> wshape = w.GetTensorTypeAndShapeInfo().GetShape();
    if (xshape.size() != 3 || wshape.size() != 3) {
      ORT_CXX_API_THROW("DepthwiseConv1d expects x[T,N,C], weight[C,1,K]",
                        ORT_INVALID_ARGUMENT);
    }
    const int64_t T = xshape[0], N = xshape[1], C = xshape[2];
    const int64_t K = wshape[2];

    const float* wp = w.GetTensorData<float>();  // [C,1,K], w[c,0,j]=wp[c*K+j]

    Ort::UnownedValue out = ctx.GetOutput(0, xshape);
    if (T == 0 || N == 0 || C == 0) {
      return;
    }

    std::vector<float> wpacked(static_cast<size_t>(K * C));
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t j = 0; j < K; ++j) {
        wpacked[j * C + c] = wp[c * K + j];
      }
    }

    DwJob job{x.GetTensorData<float>(), wpacked.data(), b.GetTensorData<float>(),
              out.GetTensorMutableData<float>(), T, N, C, K, K / 2};
    ctx.ParallelFor(ComputeConvRow, static_cast<size_t>(T * N),
                    /*num_batch=*/0, &job);
  }
};

struct BiasNormJob {
  const float* x;     // [rows, C]
  const float* bias;  // [C]
  float* out;         // [rows, C]
  int64_t C;
  float scale;        // exp(log_scale)
};

void ComputeBiasNormRow(void* user_data, size_t r) {
  const BiasNormJob& job = *static_cast<const BiasNormJob*>(user_data);
  const int64_t C = job.C;
  const float* x = job.x + static_cast<size_t>(r) * C;
  const float* bias = job.bias;
  float* out = job.out + static_cast<size_t>(r) * C;

  float ss = 0.0f;
  for (int64_t c = 0; c < C; ++c) {
    const float d = x[c] - bias[c];
    ss += d * d;
  }
  const float ms = ss / static_cast<float>(C);
  const float s = job.scale / std::sqrt(ms);
  for (int64_t c = 0; c < C; ++c) out[c] = x[c] * s;
}

struct BiasNormKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    Ort::ConstValue x = ctx.GetInput(0);
    Ort::ConstValue bias = ctx.GetInput(1);
    Ort::ConstValue scale = ctx.GetInput(2);

    auto info = x.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const size_t count = info.GetElementCount();
    if (shape.empty()) {
      ORT_CXX_API_THROW("BiasNorm expects rank >= 1", ORT_INVALID_ARGUMENT);
    }
    const int64_t C = shape.back();

    Ort::UnownedValue out = ctx.GetOutput(0, shape);
    if (count == 0 || C == 0) {
      return;
    }
    BiasNormJob job{x.GetTensorData<float>(), bias.GetTensorData<float>(),
                    out.GetTensorMutableData<float>(), C,
                    scale.GetTensorData<float>()[0]};
    ctx.ParallelFor(ComputeBiasNormRow, count / static_cast<size_t>(C),
                    /*num_batch=*/0, &job);
  }
};

struct BypassJob {
  const float* src_orig;  // [rows, C]
  const float* src;       // [rows, C]
  const float* scale;     // [C]
  float* out;             // [rows, C]
  int64_t C;
};

void ComputeBypassRow(void* user_data, size_t r) {
  const BypassJob& job = *static_cast<const BypassJob*>(user_data);
  const int64_t C = job.C;
  const size_t off = static_cast<size_t>(r) * C;
  const float* a = job.src_orig + off;
  const float* b = job.src + off;
  const float* scale = job.scale;
  float* out = job.out + off;
  for (int64_t c = 0; c < C; ++c) out[c] = a[c] + (b[c] - a[c]) * scale[c];
}

struct BypassKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    Ort::ConstValue src_orig = ctx.GetInput(0);
    Ort::ConstValue src = ctx.GetInput(1);
    Ort::ConstValue scale = ctx.GetInput(2);

    auto info = src.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const size_t count = info.GetElementCount();
    if (shape.empty()) {
      ORT_CXX_API_THROW("Bypass expects rank >= 1", ORT_INVALID_ARGUMENT);
    }
    const int64_t C = shape.back();

    Ort::UnownedValue out = ctx.GetOutput(0, shape);
    if (count == 0 || C == 0) {
      return;
    }
    BypassJob job{src_orig.GetTensorData<float>(), src.GetTensorData<float>(),
                  scale.GetTensorData<float>(),
                  out.GetTensorMutableData<float>(), C};
    ctx.ParallelFor(ComputeBypassRow, count / static_cast<size_t>(C),
                    /*num_batch=*/0, &job);
  }
};

struct SwooshLOp : Ort::CustomOpBase<SwooshLOp, SwooshKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new SwooshKernel(/*is_left=*/true);
  }
  const char* GetName() const { return "SwooshL"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct SwooshROp : Ort::CustomOpBase<SwooshROp, SwooshKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new SwooshKernel(/*is_left=*/false);
  }
  const char* GetName() const { return "SwooshR"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct GluGateOp : Ort::CustomOpBase<GluGateOp, GluKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new GluKernel();
  }
  const char* GetName() const { return "GluGate"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct DepthwiseConvOp
    : Ort::CustomOpBase<DepthwiseConvOp, DepthwiseConvKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new DepthwiseConvKernel();
  }
  const char* GetName() const { return "DepthwiseConv1d"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct BiasNormOp : Ort::CustomOpBase<BiasNormOp, BiasNormKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new BiasNormKernel();
  }
  const char* GetName() const { return "BiasNorm"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct BypassOp : Ort::CustomOpBase<BypassOp, BypassKernel> {
  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
    return new BypassKernel();
  }
  const char* GetName() const { return "Bypass"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

// Op instances must outlive every session that references them, so keep them in static storage.
SwooshLOp c_SwooshLOp;
SwooshROp c_SwooshROp;
GluGateOp c_GluGateOp;
DepthwiseConvOp c_DepthwiseConvOp;
BiasNormOp c_BiasNormOp;
BypassOp c_BypassOp;

// One process-lifetime domain holding all ops; safe to Add() to multiple SessionOptions.
Ort::CustomOpDomain& zipvoice_domain() {
  static Ort::CustomOpDomain domain = [] {
    Ort::CustomOpDomain d("ai.zipvoice");
    d.Add(&c_SwooshLOp);
    d.Add(&c_SwooshROp);
    d.Add(&c_GluGateOp);
    d.Add(&c_DepthwiseConvOp);
    d.Add(&c_BiasNormOp);
    d.Add(&c_BypassOp);
    return d;
  }();
  return domain;
}

}  // namespace

void zipvoice_register_custom_ops(Ort::SessionOptions& opts) {
  opts.Add(zipvoice_domain());
}

}  // namespace moonshine_tts
