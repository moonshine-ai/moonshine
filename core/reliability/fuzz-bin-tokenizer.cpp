// libFuzzer harness for the binary tokenizer.
//
// The in-memory BinTokenizer constructor parses an untrusted byte blob (a
// tokenizer.bin loaded from disk or an app bundle), so it is a natural fuzz
// target. We first build a tokenizer from the raw input, then round-trip a
// slice of the same input through text_to_tokens / tokens_to_text. Exceptions
// are the library's documented failure mode, so they are caught and ignored;
// only sanitizer errors (out-of-bounds reads, leaks, UB) fail the run.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "bin-tokenizer.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  try {
    BinTokenizer tokenizer(data, size);

    // Exercise encode/decode with a bounded slice of the input as text.
    const size_t text_size = std::min<size_t>(size, 128);
    const std::string text(data, data + text_size);
    try {
      const std::vector<int32_t> tokens =
          tokenizer.text_to_tokens<int32_t>(text);
      tokenizer.tokens_to_text<int32_t>(tokens);
    } catch (const std::exception &) {
      // Unknown byte sequences legitimately throw.
    }
  } catch (const std::exception &) {
    // Malformed blobs legitimately throw.
  }
  return 0;
}
