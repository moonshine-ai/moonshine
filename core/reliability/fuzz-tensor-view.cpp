// libFuzzer harness for MoonshineTensorView construction.
//
// The shape/dtype/data constructor forwards to
// moonshine_tensor_from_shape_and_dtype, which turns an (attacker-influenced)
// shape into an element count, a byte size, an allocation, and a memcpy. A
// crafted shape can overflow the size arithmetic; this harness feeds arbitrary
// dimensions (including huge values whose product overflows) so ASan/UBSan
// catch any regression in the overflow handling. Unsupported dtypes throw by
// design, so construction is wrapped; only sanitizer errors fail the run.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

#include "moonshine-tensor-view.h"

namespace {

// Minimal sequential reader over the fuzz input; out-of-bytes reads yield zero.
struct Reader {
  const uint8_t *data;
  size_t size;
  size_t pos;

  uint8_t next_u8() { return pos < size ? data[pos++] : 0; }

  int64_t next_i64() {
    int64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value =
          static_cast<int64_t>((static_cast<uint64_t>(value) << 8) | next_u8());
    }
    return value;
  }
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  Reader reader{data, size, 0};

  // First byte selects the dtype across the whole enum, including the
  // unsupported values that legitimately throw.
  const uint32_t dtype = reader.next_u8() % MOONSHINE_DTYPE_MAX;

  // Next byte selects the rank (0..8). A small cap keeps the loop bounded while
  // still letting each dimension be an arbitrary 64-bit value, so their product
  // can overflow the size computation.
  const size_t rank = reader.next_u8() % 9;
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    shape.push_back(reader.next_i64());
  }

  // Decide whether to supply a real source buffer for the copy path. We only do
  // so when the element count is small and non-overflowing, so the fuzzer never
  // exhausts memory and never over-reads a short source. The overflow/oversize
  // shapes are still exercised (with a null source): the interesting size math
  // runs regardless, and UBSan flags any signed-overflow regression there.
  constexpr size_t kMaxCopyElements = 1 << 16;  // 64 Ki elements.
  bool safe_to_copy = true;
  size_t element_count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0 || (dim != 0 && element_count > kMaxCopyElements /
                                                    static_cast<size_t>(dim))) {
      safe_to_copy = false;
      break;
    }
    element_count *= static_cast<size_t>(dim);
    if (element_count > kMaxCopyElements) {
      safe_to_copy = false;
      break;
    }
  }

  try {
    if (safe_to_copy) {
      // 8 bytes/element is the widest dtype, so this source is always at least
      // as large as the copy the constructor performs.
      std::vector<uint8_t> source(element_count * 8, 0);
      MoonshineTensorView view(shape, dtype, source.data(), "fuzz");
    } else {
      MoonshineTensorView view(shape, dtype, nullptr, "fuzz");
    }
  } catch (const std::exception &) {
    // Unsupported dtypes (and other documented failures) throw.
  }
  return 0;
}
