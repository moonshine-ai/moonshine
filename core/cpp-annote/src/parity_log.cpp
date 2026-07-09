// SPDX-License-Identifier: MIT

#include "parity_log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace cppannote::parity {
namespace {

const char* kParityEnv = "PYANNOTE_CPP_PARITY";
const char* kOutEnv = "PYANNOTE_CPP_PARITY_OUT";

}  // namespace

int env_parity_level() {
  const char* v = std::getenv(kParityEnv);
  if (!v || v[0] == '\0' || (v[0] == '0' && v[1] == '\0')) {
    return 0;
  }
  if (v[0] == '2') {
    return 2;
  }
  if (v[0] == '1') {
    return 1;
  }
  return 0;
}

const char* env_parity_out_dir() {
  const char* d = std::getenv(kOutEnv);
  if (!d || d[0] == '\0') {
    return nullptr;
  }
  return d;
}

void log_light(const std::string& line) {
  if (env_parity_level() >= 1) {
    std::cerr << "[PYANNOTE_CPP_PARITY] " << line << '\n';
  }
}

bool heavy_dumps_enabled() {
  return env_parity_level() >= 2 && env_parity_out_dir() != nullptr;
}

void ensure_parity_out_dir() {
  if (!heavy_dumps_enabled()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(env_parity_out_dir(), ec);
  if (ec) {
    std::cerr << "[PYANNOTE_CPP_PARITY] warning: could not create "
                 "PYANNOTE_CPP_PARITY_OUT: "
              << ec.message() << '\n';
  }
}

std::string parity_clustering_npz_path() {
  const char* d = env_parity_out_dir();
  if (!d) {
    return {};
  }
  std::filesystem::path p(d);
  p /= "vbx_parity_dump.npz";
  return p.string();
}

std::string fingerprint_float32(const float* data, std::size_t n,
                                std::size_t stride) {
  if (stride < 1) {
    stride = 1;
  }
  std::uint64_t h = 14695981039346656037ULL;
  const std::uint64_t prime = 1099511628211ULL;
  for (std::size_t i = 0; i < n; i += stride) {
    float v = data[i];
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    h ^= static_cast<std::uint64_t>(bits);
    h *= prime;
  }
  h ^= static_cast<std::uint64_t>(n);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}

}  // namespace cppannote::parity
