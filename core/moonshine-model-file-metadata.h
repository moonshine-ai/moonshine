#ifndef MOONSHINE_MODEL_FILE_METADATA_H
#define MOONSHINE_MODEL_FILE_METADATA_H

#include <cstdint>
#include <string>

namespace moonshine {

// Size and checksum for a single downloadable model file, keyed by its full
// download URL. This is the single source of truth for per-file integrity
// metadata; it is generated from the live CDN objects by
// scripts/generate-model-file-metadata.py and baked into
// moonshine-model-file-metadata.generated.cpp so no network access is needed at
// runtime.
struct ModelFileMetadata {
  int64_t size = -1;          // Bytes, or -1 when the URL is not registered.
  std::string checksum;       // Base64-encoded digest, or "" when unknown.
  std::string checksum_type;  // e.g. "crc32c", or "" when unknown.
};

// Returns the registered metadata for `url`, or {-1, "", ""} when the URL is
// not in the generated registry (e.g. a newly published file whose metadata has
// not been regenerated yet). Callers should treat unknown size/checksum as
// "skip verification" rather than an error.
ModelFileMetadata find_model_file_metadata(const std::string& url);

}  // namespace moonshine

#endif  // MOONSHINE_MODEL_FILE_METADATA_H
