/* Forwarding stubs for the two ONNX Runtime entry points Moonshine calls by
   name. Only compiled into the Apple static framework, where
   scripts/prelink-apple-static-library.sh links the bundled ONNX Runtime into
   a single object whose globals are renamed to moonshine_ort_* and hidden.
   Moonshine's own objects keep calling OrtGetApiBase(), which resolves to
   these stubs inside the prelinked Moonshine object and never escapes it, so
   an app that also links a different ONNX Runtime (issue #235) sees neither
   name from libmoonshine.a. */

#include "cpu_provider_factory.h"
#include "onnxruntime_c_api.h"

/* The aliases the prelink script gives the real ONNX Runtime definitions. */
const OrtApiBase *ORT_API_CALL moonshine_ort_get_api_base(void) NO_EXCEPTION;
ORT_API_STATUS_IMPL(moonshine_ort_append_cpu_ep, OrtSessionOptions *options,
                    int use_arena);

const OrtApiBase *ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
  return moonshine_ort_get_api_base();
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CPU,
                    OrtSessionOptions *options, int use_arena) {
  return moonshine_ort_append_cpu_ep(options, use_arena);
}
