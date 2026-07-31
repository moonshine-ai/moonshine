#ifndef MOONSHINE_TTS_TESTS_RULE_G2P_TEST_SUPPORT_H
#define MOONSHINE_TTS_TESTS_RULE_G2P_TEST_SUPPORT_H

/// Shared helpers for rule-G2P / ONNX parity tests (pre-generated reference
/// lines under ``tests/data/``).

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace moonshine_tts::rule_g2p_test {

/// True when *dir* holds a model in any of the forms the loaders accept.
///
/// Models ship in whichever form suits them: a split ORT pair, a single
/// ``.ort``, or a ``.onnx``. Tests gate on the model being present at all, so
/// checking a single hard-coded name would make them go quiet rather than fail
/// when the shipped form changes.
inline bool model_present(const std::filesystem::path& dir,
                          const std::string& stem = "model") {
  namespace fs = std::filesystem;
  return fs::is_regular_file(dir / (stem + ".model.ort")) ||
         fs::is_regular_file(dir / (stem + ".ort")) ||
         fs::is_regular_file(dir / (stem + ".onnx"));
}

/// True when *dir* holds the split ORT pair specifically.
inline bool model_ships_split(const std::filesystem::path& dir,
                              const std::string& stem = "model") {
  namespace fs = std::filesystem;
  return fs::is_regular_file(dir / (stem + ".model.ort")) &&
         fs::is_regular_file(dir / (stem + ".weights.ort"));
}

/// Directory that contains ``data/`` (lexicons, ONNX assets) and usually
/// ``models/``. Prefers the moonshine-tts tree when it embeds ``data/``;
/// otherwise the parent tree (monorepo layout with ``data/`` next to
/// ``moonshine-tts/``).
inline std::filesystem::path repo_root_from_tests_cpp(
    const char* tests_cpp_file) {
  namespace fs = std::filesystem;
  const fs::path component =
      fs::path(tests_cpp_file).parent_path().parent_path();
  if (fs::is_directory(component / "data")) {
    return component;
  }
  return component.parent_path();
}

/// Pre-generated parity lines: ``<tts>/tests/data`` when built in-tree, or
/// legacy monorepo paths.
inline std::filesystem::path tests_data_dir(
    const std::filesystem::path& repo_root) {
  namespace fs = std::filesystem;
  const fs::path direct = repo_root / "tests" / "data";
  if (fs::is_directory(direct)) {
    return direct;
  }
  const fs::path submodule = repo_root / "moonshine-tts" / "tests" / "data";
  if (fs::is_directory(submodule)) {
    return submodule;
  }
  const fs::path integrated =
      repo_root / "core" / "moonshine-tts" / "tests" / "data";
  if (fs::is_directory(integrated)) {
    return integrated;
  }
  return repo_root / "cpp" / "tests" / "data";
}

inline std::vector<std::string> split_unix_lines(std::string block) {
  std::vector<std::string> lines;
  std::istringstream iss(block);
  std::string L;
  while (std::getline(iss, L)) {
    if (!L.empty() && L.back() == '\r') {
      L.pop_back();
    }
    lines.push_back(std::move(L));
  }
  return lines;
}

inline std::string load_ref_text_trimmed(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::string s((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

inline std::vector<std::string> load_ref_lines(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::string block((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  return split_unix_lines(std::move(block));
}

/// Use the first *n* lines from a golden file (generated for up to 100 wiki
/// lines).
inline std::vector<std::string> ref_lines_prefix(
    const std::filesystem::path& golden, std::size_t n) {
  const std::vector<std::string> all = load_ref_lines(golden);
  if (all.size() < n) {
    return {};
  }
  using diff = std::vector<std::string>::difference_type;
  return std::vector<std::string>(all.begin(),
                                  all.begin() + static_cast<diff>(n));
}

inline std::vector<std::string> read_text_first_lines(
    const std::filesystem::path& p, std::size_t n) {
  std::ifstream in(p);
  std::vector<std::string> src;
  std::string line;
  while (src.size() < n && std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    src.push_back(std::move(line));
  }
  return src;
}

/// Path to the bundled ``moonshine-tts`` data tree, relative to the monorepo
/// repository root. Tests that use this path must run with the process current
/// working directory set to that root (see ``moonshine_tts_add_test`` in
/// CMake).
inline std::filesystem::path moonshine_tts_bundled_data_dir_relative() {
  return std::filesystem::path("core") / "moonshine-tts" / "data";
}

}  // namespace moonshine_tts::rule_g2p_test

#endif  // MOONSHINE_TTS_TESTS_RULE_G2P_TEST_SUPPORT_H
