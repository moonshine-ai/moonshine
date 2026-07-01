#ifndef MOONSHINE_TTS_ZIPVOICE_CUSTOM_OPS_H
#define MOONSHINE_TTS_ZIPVOICE_CUSTOM_OPS_H

#include <onnxruntime_cxx_api.h>

namespace moonshine_tts {

/// Registers the ``ai.zipvoice`` custom ONNX Runtime operators (SwooshL / SwooshR / GluGate /
/// DepthwiseConv1d / BiasNorm / Bypass) on ``opts``. Unlike the standalone ``swoosh_op`` shared
/// library, these kernels are compiled directly into ``libmoonshine`` and registered in-process, so
/// no ``.so`` / ``.dll`` needs to be loaded at runtime (required for platforms such as iOS). The
/// custom-op domain and op instances have process lifetime and may be added to multiple sessions.
void zipvoice_register_custom_ops(Ort::SessionOptions& opts);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ZIPVOICE_CUSTOM_OPS_H
