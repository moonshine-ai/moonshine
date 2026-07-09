// SPDX-License-Identifier: MIT
// Stage A: ``PYANNOTE_CPP_PARITY`` light/heavy diagnostics for embedding + VBx
// (see ``embedding-vbx-parity-plan.md``).

#ifndef PARITY_LOG_H_
#define PARITY_LOG_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace cppannote::parity {

/// ``PYANNOTE_CPP_PARITY``: unset or ``0`` = off; ``1`` = stderr light log;
/// ``2`` = heavy dumps (needs out dir).
int env_parity_level();

/// ``PYANNOTE_CPP_PARITY_OUT``: directory for level-2 NPZ bundle (created if
/// missing). Ignored when level < 2.
const char* env_parity_out_dir();

/// One stderr line when ``env_parity_level() >= 1``.
void log_light(const std::string& line);

/// True when level >= 2 and ``PYANNOTE_CPP_PARITY_OUT`` is non-empty.
bool heavy_dumps_enabled();

/// Create output directory (recursive). No-op if disabled.
void ensure_parity_out_dir();

/// Path to the single NPZ written for VBx parity dumps.
std::string parity_clustering_npz_path();

/// FNV-1a 64-bit fingerprint of strided ``float`` samples (must match
/// ``cpp/scripts/parity_fingerprint.py``).
/// ``stride`` defaults to 409; mixes in ``n`` at the end.
std::string fingerprint_float32(const float* data, std::size_t n,
                                std::size_t stride = 409);

}  // namespace cppannote::parity

#endif  // PARITY_LOG_H_
