// Integration test: TTS from memory while CWD is an empty sandbox (no repo data).
// Usage: moonshine-c-api-memory-test <ABSOLUTE_PATH_TO_DATA_DIR>
// Example: moonshine-c-api-memory-test /Users/you/projects/moonshine/core/moonshine-tts/data
//
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include "moonshine-c-api.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

std::filesystem::path g_data_root;

std::vector<uint8_t> read_binary_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return {};
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Kokoro voice ids use a two-letter family prefix (e.g. af_alloy -> en_us).
const char* kokoro_lang_for_voice_stem(std::string_view stem) {
  if (stem.size() < 2) {
    return nullptr;
  }
  const std::string_view p = stem.substr(0, 2);
  if (p == "af" || p == "am") {
    return "en_us";
  }
  if (p == "bf" || p == "bm") {
    return "en_gb";
  }
  if (p == "ef" || p == "em") {
    return "es";
  }
  if (p == "ff") {
    return "fr";
  }
  if (p == "hf" || p == "hm") {
    return "hi";
  }
  if (p == "if" || p == "im") {
    return "it";
  }
  if (p == "pf" || p == "pm") {
    return "pt_br";
  }
  if (p == "jf" || p == "jm") {
    return "ja";
  }
  if (p == "zf" || p == "zm") {
    return "zh_hans";
  }
  return nullptr;
}

const char* sample_text_for_kokoro_lang(const char* lang) {
  if (std::strcmp(lang, "en_us") == 0 || std::strcmp(lang, "en_gb") == 0) {
    return "Hello";
  }
  if (std::strcmp(lang, "es") == 0) {
    return "Hola";
  }
  if (std::strcmp(lang, "fr") == 0) {
    return "Bonjour";
  }
  if (std::strcmp(lang, "hi") == 0) {
    return "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87";
  }
  if (std::strcmp(lang, "it") == 0) {
    return "Ciao";
  }
  if (std::strcmp(lang, "pt_br") == 0) {
    return "Ol\xc3\xa1";
  }
  if (std::strcmp(lang, "ja") == 0) {
    return "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf";
  }
  if (std::strcmp(lang, "zh_hans") == 0) {
    return "\xe4\xbd\xa0\xe5\xa5\xbd";
  }
  return "Hello";
}

/// Subtrees needed for Kokoro + rule G2P (Spanish is rule-only; no lexicon files).
void append_files_under(const std::filesystem::path& root, const std::filesystem::path& sub,
                        std::vector<std::pair<std::string, std::vector<uint8_t>>>& out) {
  namespace fs = std::filesystem;
  const fs::path base = root / sub;
  if (!fs::exists(base)) {
    return;
  }
  for (const auto& ent : fs::recursive_directory_iterator(base)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    const fs::path p = ent.path();
    if (p.filename() == ".DS_Store") {
      continue;
    }
    std::error_code ec;
    const fs::path rel = fs::relative(p, root, ec);
    if (ec) {
      continue;
    }
    std::string key = rel.generic_string();
    if (key.empty()) {
      continue;
    }
    std::vector<uint8_t> bytes = read_binary_file(p);
    out.emplace_back(std::move(key), std::move(bytes));
  }
}

void build_kokoro_g2p_memory_bundle(const std::filesystem::path& data_root,
                                    std::vector<std::pair<std::string, std::vector<uint8_t>>>& out) {
  out.clear();
  append_files_under(data_root, "kokoro", out);
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const std::pair<std::string, std::vector<uint8_t>>& pr) {
                             return pr.first.rfind("kokoro/voices/", 0) == 0;
                           }),
            out.end());
  static const char* kLangDirs[] = {"en_us", "en_gb", "es", "fr", "hi", "it", "pt_br", "ja", "zh_hans"};
  for (const char* d : kLangDirs) {
    append_files_under(data_root, d, out);
  }
}

}  // namespace

TEST_CASE("moonshine-c-api-memory: Kokoro every voice uses only buffers; CWD has no data tree") {
  REQUIRE_FALSE(g_data_root.empty());
  REQUIRE(std::filesystem::is_directory(g_data_root));

  namespace fs = std::filesystem;
  const fs::path cwd = fs::current_path();
  REQUIRE(cwd != g_data_root);
  REQUIRE_FALSE(fs::exists(cwd / "kokoro"));
  REQUIRE_FALSE(fs::exists(cwd / "en_us"));

  std::vector<std::pair<std::string, std::vector<uint8_t>>> bundle;
  build_kokoro_g2p_memory_bundle(g_data_root, bundle);
  REQUIRE_FALSE(bundle.empty());

  const fs::path voices_dir = g_data_root / "kokoro" / "voices";
  REQUIRE(std::filesystem::is_directory(voices_dir));

  auto model_key_it =
      std::find_if(bundle.begin(), bundle.end(), [](const auto& pr) { return pr.first == "kokoro/model.onnx"; });
  if (model_key_it == bundle.end()) {
    model_key_it =
        std::find_if(bundle.begin(), bundle.end(), [](const auto& pr) { return pr.first == "kokoro/model.ort"; });
  }
  REQUIRE(model_key_it != bundle.end());
  REQUIRE_FALSE(model_key_it->second.empty());

  std::vector<std::string> voice_stems;
  for (const auto& ent : std::filesystem::directory_iterator(voices_dir)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    const auto& p = ent.path();
    if (p.extension() != ".kokorovoice") {
      continue;
    }
    const std::string stem = p.stem().string();
    REQUIRE_MESSAGE(kokoro_lang_for_voice_stem(stem) != nullptr,
                    "Unrecognized Kokoro voice id (add prefix mapping): " << stem);
    voice_stems.push_back(stem);
  }
  std::sort(voice_stems.begin(), voice_stems.end());
  REQUIRE_FALSE(voice_stems.empty());

  for (const std::string& voice : voice_stems) {
    const char* lang = kokoro_lang_for_voice_stem(voice);
    REQUIRE(lang != nullptr);

    std::vector<std::string> keys_storage;
    std::vector<const uint8_t*> mem_ptrs;
    std::vector<uint64_t> mem_sizes;
    std::vector<std::vector<uint8_t>> voice_scratch;

    for (const auto& pr : bundle) {
      if (pr.first.rfind("kokoro/voices/", 0) == 0) {
        continue;
      }
      keys_storage.push_back(pr.first);
      mem_ptrs.push_back(pr.second.data());
      mem_sizes.push_back(static_cast<uint64_t>(pr.second.size()));
    }
    const std::string vkey = std::string("kokoro/voices/") + voice + ".kokorovoice";
    voice_scratch.push_back(read_binary_file(g_data_root / vkey));
    REQUIRE_FALSE(voice_scratch.back().empty());
    keys_storage.push_back(vkey);
    mem_ptrs.push_back(voice_scratch.back().data());
    mem_sizes.push_back(static_cast<uint64_t>(voice_scratch.back().size()));

    std::vector<const char*> filenames;
    filenames.reserve(keys_storage.size());
    for (const std::string& k : keys_storage) {
      filenames.push_back(k.c_str());
    }

    const std::string model_root_str = cwd.string();
    const std::string kokoro_voice = std::string("kokoro_") + voice;
    const moonshine_option_t opts[] = {
        {"voice", kokoro_voice.c_str()},
        {"speed", "1.0"},
        {"model_root", model_root_str.c_str()},
    };
    const uint64_t n_opts = sizeof(opts) / sizeof(opts[0]);

    int32_t h = moonshine_create_tts_synthesizer_from_memory(
        lang, filenames.data(), static_cast<uint64_t>(filenames.size()), mem_ptrs.data(), mem_sizes.data(),
        opts, n_opts, MOONSHINE_HEADER_VERSION);
    REQUIRE(h >= 0);
    float* audio = nullptr;
    uint64_t audio_n = 0;
    int32_t sr = 0;
    const char* text = sample_text_for_kokoro_lang(lang);
    REQUIRE(moonshine_text_to_speech(h, text, nullptr, 0, &audio, &audio_n, &sr) == MOONSHINE_ERROR_NONE);
    REQUIRE(audio_n > 0);
    REQUIRE(audio != nullptr);
    std::free(audio);
    moonshine_free_tts_synthesizer(h);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "moonshine-c-api-memory-test")
              << " <absolute-path-to-moonshine-tts-data>\n"
              << "  (the directory that contains kokoro/, en_us/, etc. — e.g. "
                 ".../moonshine/core/moonshine-tts/data)\n";
    return 2;
  }
  std::error_code ec;
  g_data_root = std::filesystem::weakly_canonical(std::filesystem::path(argv[1]), ec);
  if (ec || !std::filesystem::is_directory(g_data_root)) {
    std::cerr << "Invalid or missing data directory: " << argv[1] << '\n'
              << "  Pass the real absolute path to core/moonshine-tts/data in your clone "
                 "(not a documentation placeholder).\n";
    return 2;
  }

  namespace fs = std::filesystem;
  const unsigned salt = static_cast<unsigned>(std::time(nullptr)) ^
                         static_cast<unsigned>(std::random_device{}()) ^
#if defined(_WIN32)
                         static_cast<unsigned>(_getpid());
#else
                         static_cast<unsigned>(getpid());
#endif
  const fs::path sandbox = fs::temp_directory_path() / ("moonshine_c_api_mem_" + std::to_string(salt));
  fs::create_directories(sandbox);
  fs::current_path(sandbox);

  std::vector<char*> doctest_argv;
  doctest_argv.reserve(static_cast<size_t>(argc));
  doctest_argv.push_back(argv[0]);
  for (int i = 2; i < argc; ++i) {
    doctest_argv.push_back(argv[i]);
  }

  doctest::Context ctx;
  ctx.applyCommandLine(static_cast<int>(doctest_argv.size()), doctest_argv.data());
  const int r = ctx.run();
  std::filesystem::remove_all(sandbox, ec);
  return r;
}
