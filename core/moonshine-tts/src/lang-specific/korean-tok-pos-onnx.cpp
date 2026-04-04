#include "korean-tok-pos-onnx.h"
#include "g2p-path.h"
#include "moonshine-g2p-options.h"
#include "ort-onnx-external-data.h"
#include "utf8-utils.h"

#include <nlohmann/json.h>

#include <cctype>
#include <cstdint>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

Ort::SessionOptions make_session_options(bool use_cuda) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  (void)use_cuda;
  return session_options;
}

std::unique_ptr<Ort::Session> open_session(Ort::Env& env, const std::filesystem::path& model_path,
                                           bool use_cuda) {
#ifdef _WIN32
  const std::wstring w = model_path.wstring();
  return std::make_unique<Ort::Session>(env, w.c_str(), make_session_options(use_cuda));
#else
  const std::string u8 = model_path.string();
  return std::make_unique<Ort::Session>(env, u8.c_str(), make_session_options(use_cuda));
#endif
}

std::unique_ptr<Ort::Session> open_session_memory(Ort::Env& env, const void* data, size_t len,
                                                  bool use_cuda, const MoonshineG2POptions* opt,
                                                  std::string_view model_map_key) {
  Ort::SessionOptions so = make_session_options(use_cuda);
  if (opt != nullptr && !model_map_key.empty()) {
    ort_add_external_initializer_files_for_onnx_model_buffer(so, opt->files, model_map_key);
  }
  return std::make_unique<Ort::Session>(env, data, len, so);
}

std::string slurp_utf8_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return {};
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

bool bundle_load_utf8(const MoonshineG2POptions* opt, std::string_view bundle_key, std::string_view file,
                      const std::filesystem::path& disk_dir, std::string& out) {
  if (opt && !bundle_key.empty()) {
    const std::string ak = g2p_bundle_file_key(bundle_key, file);
    if (opt->asset_is_available(ak)) {
      out = opt->read_utf8_asset(ak);
      return true;
    }
  }
  const auto p = disk_dir / std::string(file);
  if (std::filesystem::is_regular_file(p)) {
    out = slurp_utf8_file(p);
    return true;
  }
  return false;
}

bool bundle_load_binary(const MoonshineG2POptions* opt, std::string_view bundle_key, std::string_view file,
                        const std::filesystem::path& disk_dir, std::vector<std::uint8_t>& out) {
  if (opt && !bundle_key.empty()) {
    const std::string ak = g2p_bundle_file_key(bundle_key, file);
    if (opt->asset_is_available(ak)) {
      out = opt->read_binary_asset(ak);
      return true;
    }
  }
  const auto p = resolve_prefer_ort_model(disk_dir, file);
  if (!std::filesystem::is_regular_file(p)) {
    return false;
  }
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const auto sz = in.tellg();
  in.seekg(0, std::ios::beg);
  if (sz < 0) {
    return false;
  }
  out.resize(static_cast<size_t>(sz));
  if (!out.empty()) {
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  }
  return true;
}

std::u32string utf8_to_u32(std::string_view utf8) {
  std::u32string out;
  size_t i = 0;
  while (i < utf8.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(std::string(utf8), i, cp, adv);
    out.push_back(cp);
    i += adv;
  }
  return out;
}

std::string u32_to_utf8(const std::u32string& s) {
  std::string out;
  for (char32_t cp : s) {
    utf8_append_codepoint(out, cp);
  }
  return out;
}

bool is_space_u32(char32_t c) {
  if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' || c == U'\v') {
    return true;
  }
  return utf8proc_category(static_cast<utf8proc_int32_t>(c)) == UTF8PROC_CATEGORY_ZS;
}

bool is_control_u32(char32_t c) {
  if (c == U'\t' || c == U'\n' || c == U'\r') {
    return false;
  }
  const auto cat = utf8proc_category(static_cast<utf8proc_int32_t>(c));
  return cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_CN ||
         cat == UTF8PROC_CATEGORY_CO;
}

bool is_punctuation_u32(char32_t c) {
  const std::uint32_t cp = static_cast<std::uint32_t>(c);
  if ((33 <= cp && cp <= 47) || (58 <= cp && cp <= 64) || (91 <= cp && cp <= 96) ||
      (123 <= cp && cp <= 126)) {
    return true;
  }
  const char* cat = utf8proc_category_string(static_cast<utf8proc_int32_t>(c));
  return cat != nullptr && cat[0] == 'P';
}

/// ``token_word_group_indices`` uses :func:`_is_punct_char` (Unicode **P** categories only), not the
/// broader BERT ``_is_punctuation`` (ASCII ranges include ``~`` as Sm).
bool is_punct_char_word_group_u32(char32_t c) {
  const char* cat = utf8proc_category_string(static_cast<utf8proc_int32_t>(c));
  return cat != nullptr && cat[0] == 'P';
}

bool is_chinese_char(std::uint32_t cp) {
  return (0x4E00u <= cp && cp <= 0x9FFFu) || (0x3400u <= cp && cp <= 0x4DBFu) ||
         (0x20000u <= cp && cp <= 0x2A6DFu) || (0x2A700u <= cp && cp <= 0x2B73Fu) ||
         (0x2B740u <= cp && cp <= 0x2B81Fu) || (0x2B820u <= cp && cp <= 0x2CEAFu) ||
         (0xF900u <= cp && cp <= 0xFAFFu) || (0x2F800u <= cp && cp <= 0x2FA1Fu);
}

std::u32string u32_nfc(const std::u32string& s) {
  std::string utf8 = u32_to_utf8(s);
  utf8proc_uint8_t* nfc = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(utf8.c_str()));
  if (nfc == nullptr) {
    return s;
  }
  std::string composed(reinterpret_cast<char*>(nfc));
  std::free(nfc);
  return utf8_to_u32(composed);
}

std::u32string strip_mn_nfd(const std::u32string& s) {
  std::string utf8 = u32_to_utf8(s);
  utf8proc_uint8_t* nfd = utf8proc_NFD(reinterpret_cast<const utf8proc_uint8_t*>(utf8.c_str()));
  if (nfd == nullptr) {
    return s;
  }
  std::string nfd_str(reinterpret_cast<char*>(nfd));
  std::free(nfd);
  std::u32string u = utf8_to_u32(nfd_str);
  std::u32string out;
  for (char32_t cp : u) {
    if (utf8proc_category(static_cast<utf8proc_int32_t>(cp)) != UTF8PROC_CATEGORY_MN) {
      out.push_back(cp);
    }
  }
  return u32_nfc(out);
}

std::u32string to_lower_u32(const std::u32string& s) {
  std::u32string out;
  for (char32_t cp : s) {
    out.push_back(static_cast<char32_t>(utf8proc_tolower(static_cast<utf8proc_int32_t>(cp))));
  }
  return out;
}

std::u32string clean_text_u32(const std::u32string& text) {
  std::u32string out;
  for (char32_t c : text) {
    const std::uint32_t cp = static_cast<std::uint32_t>(c);
    if (cp == 0 || cp == 0xFFFDu || is_control_u32(c)) {
      continue;
    }
    out.push_back(is_space_u32(c) ? U' ' : c);
  }
  return out;
}

std::u32string tokenize_chinese_chars_u32(const std::u32string& text) {
  std::u32string out;
  for (char32_t c : text) {
    if (is_chinese_char(static_cast<std::uint32_t>(c))) {
      out.push_back(U' ');
      out.push_back(c);
      out.push_back(U' ');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::vector<std::u32string> split_u32_whitespace(std::u32string_view s) {
  while (!s.empty() && is_space_u32(s.front())) {
    s.remove_prefix(1);
  }
  while (!s.empty() && is_space_u32(s.back())) {
    s.remove_suffix(1);
  }
  std::vector<std::u32string> out;
  std::u32string cur;
  for (char32_t c : s) {
    if (is_space_u32(c)) {
      if (!cur.empty()) {
        out.push_back(std::move(cur));
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) {
    out.push_back(std::move(cur));
  }
  return out;
}

std::vector<std::u32string> run_split_on_punc_u32(const std::u32string& text) {
  std::vector<std::u32string> chars;
  chars.reserve(text.size());
  for (char32_t c : text) {
    chars.push_back(std::u32string(1, c));
  }
  std::vector<std::vector<char32_t>> output;
  std::size_t i = 0;
  bool start_new_word = true;
  while (i < chars.size()) {
    const char32_t ch = chars[i][0];
    if (is_punctuation_u32(ch)) {
      output.push_back({ch});
      start_new_word = true;
    } else {
      if (start_new_word) {
        output.push_back({});
      }
      start_new_word = false;
      output.back().push_back(ch);
    }
    ++i;
  }
  std::vector<std::u32string> flat;
  for (const auto& v : output) {
    flat.push_back(std::u32string(v.begin(), v.end()));
  }
  return flat;
}

struct BasicTokCfg {
  bool do_lower_case = true;
  bool tokenize_chinese_chars = true;
  std::optional<bool> strip_accents;
};

std::u32string normalization_ref_u32(const std::u32string& text_u32, const BasicTokCfg& cfg) {
  std::u32string t = clean_text_u32(text_u32);
  if (cfg.tokenize_chinese_chars) {
    t = tokenize_chinese_chars_u32(t);
  }
  return u32_nfc(t);
}

std::vector<std::u32string> basic_tokenize_u32(const std::u32string& original_text_u32,
                                                 const BasicTokCfg& cfg) {
  std::u32string text = clean_text_u32(original_text_u32);
  if (cfg.tokenize_chinese_chars) {
    text = tokenize_chinese_chars_u32(text);
  }
  std::u32string unicode_normalized = u32_nfc(text);
  std::vector<std::u32string> orig_tokens = split_u32_whitespace(unicode_normalized);
  std::vector<std::u32string> split_tokens;
  for (std::u32string token : orig_tokens) {
    if (cfg.do_lower_case) {
      token = to_lower_u32(token);
      if (!cfg.strip_accents.has_value() || *cfg.strip_accents) {
        token = strip_mn_nfd(token);
      }
    } else if (cfg.strip_accents.value_or(false)) {
      token = strip_mn_nfd(token);
    }
    const auto p = run_split_on_punc_u32(token);
    split_tokens.insert(split_tokens.end(), p.begin(), p.end());
  }
  std::u32string joined;
  for (size_t i = 0; i < split_tokens.size(); ++i) {
    if (i > 0) {
      joined.push_back(U' ');
    }
    joined += split_tokens[i];
  }
  return split_u32_whitespace(joined);
}

void align_basic_tokens_u32(const std::u32string& ref, const std::vector<std::u32string>& basic_tokens) {
  std::size_t cursor = 0;
  for (const std::u32string& btok : basic_tokens) {
    while (cursor < ref.size() && ref[cursor] == U' ') {
      ++cursor;
    }
    if (cursor + btok.size() > ref.size() || ref.compare(cursor, btok.size(), btok) != 0) {
      throw std::runtime_error("Korean WordPiece: basic token alignment failed at offset " +
                               std::to_string(cursor));
    }
    cursor += btok.size();
  }
}

std::vector<std::u32string> wordpiece_tokenize_u32(const std::u32string& token,
                                                   const std::unordered_map<std::string, int64_t>& vocab,
                                                   const std::string& unk_token_utf8,
                                                   int max_input_chars_per_word) {
  std::vector<std::u32string> output_tokens;
  for (const std::u32string& wt : split_u32_whitespace(token)) {
    if (wt.size() > static_cast<size_t>(max_input_chars_per_word)) {
      output_tokens.push_back(utf8_to_u32(unk_token_utf8));
      continue;
    }
    std::vector<char32_t> chars(wt.begin(), wt.end());
    bool is_bad = false;
    std::vector<std::u32string> sub_tokens;
    std::size_t start = 0;
    while (start < chars.size()) {
      std::size_t end = chars.size();
      std::string cur_substr;
      while (start < end) {
        std::u32string piece_u32(chars.begin() + static_cast<std::ptrdiff_t>(start),
                                 chars.begin() + static_cast<std::ptrdiff_t>(end));
        std::string substr = u32_to_utf8(piece_u32);
        if (start > 0) {
          substr = "##" + substr;
        }
        if (vocab.count(substr) != 0) {
          cur_substr = std::move(substr);
          break;
        }
        --end;
      }
      if (cur_substr.empty()) {
        is_bad = true;
        break;
      }
      sub_tokens.push_back(utf8_to_u32(cur_substr));
      start = end;
    }
    if (is_bad) {
      output_tokens.push_back(utf8_to_u32(unk_token_utf8));
    } else {
      output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
    }
  }
  return output_tokens;
}

struct EncodedWp {
  std::vector<int64_t> input_ids;
  std::vector<std::pair<int, int>> offsets_cp;
  std::u32string ref_u32;
  std::vector<std::vector<int>> word_groups;
};

EncodedWp encode_bert_wordpiece(const std::u32string& text_u32,
                                const std::unordered_map<std::string, int64_t>& vocab,
                                const BasicTokCfg& basic_cfg,
                                const std::string& unk_utf8,
                                const std::string& cls_utf8,
                                const std::string& sep_utf8) {
  const std::u32string ref = normalization_ref_u32(text_u32, basic_cfg);
  std::vector<std::u32string> basic_tokens = basic_tokenize_u32(text_u32, basic_cfg);
  align_basic_tokens_u32(ref, basic_tokens);

  std::vector<std::tuple<std::u32string, int, int>> aligned;
  {
    std::size_t cursor = 0;
    for (const std::u32string& btok : basic_tokens) {
      while (cursor < ref.size() && ref[cursor] == U' ') {
        ++cursor;
      }
      const int s0 = static_cast<int>(cursor);
      cursor += btok.size();
      const int e0 = static_cast<int>(cursor);
      aligned.emplace_back(btok, s0, e0);
    }
  }

  std::vector<int64_t> ids;
  std::vector<std::pair<int, int>> offsets_cp;
  std::vector<std::string> tokens_utf8;
  for (const auto& tup : aligned) {
    const std::u32string& bstr = std::get<0>(tup);
    const int s0 = std::get<1>(tup);
    const int e0 = std::get<2>(tup);
    const auto wps = wordpiece_tokenize_u32(bstr, vocab, unk_utf8, 100);
    int cur = s0;
    for (const std::u32string& wpt : wps) {
      const std::string wpt8 = u32_to_utf8(wpt);
      if (wpt8 == unk_utf8) {
        tokens_utf8.push_back(unk_utf8);
        offsets_cp.push_back({s0, e0});
        ids.push_back(vocab.at(unk_utf8));
        cur = e0;
        break;
      }
      std::u32string raw = wpt;
      if (raw.size() >= 2 && raw[0] == U'#' && raw[1] == U'#') {
        raw = raw.substr(2);
      }
      if (static_cast<std::size_t>(cur) + raw.size() > ref.size() ||
          ref.compare(static_cast<std::size_t>(cur), raw.size(), raw) != 0) {
        throw std::runtime_error("Korean WordPiece: subword span mismatch");
      }
      tokens_utf8.push_back(wpt8);
      offsets_cp.push_back({cur, cur + static_cast<int>(raw.size())});
      const auto it = vocab.find(wpt8);
      ids.push_back(it != vocab.end() ? it->second : vocab.at(unk_utf8));
      cur += static_cast<int>(raw.size());
    }
  }

  const int64_t cls_id = vocab.at(cls_utf8);
  const int64_t sep_id = vocab.at(sep_utf8);
  std::vector<int64_t> all_ids;
  all_ids.reserve(ids.size() + 2);
  all_ids.push_back(cls_id);
  all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  all_ids.push_back(sep_id);

  std::vector<std::pair<int, int>> all_offsets;
  all_offsets.reserve(offsets_cp.size() + 2);
  all_offsets.push_back({0, 0});
  all_offsets.insert(all_offsets.end(), offsets_cp.begin(), offsets_cp.end());
  all_offsets.push_back({0, 0});

  std::vector<std::string> all_tokens;
  all_tokens.reserve(tokens_utf8.size() + 2);
  all_tokens.push_back(cls_utf8);
  all_tokens.insert(all_tokens.end(), tokens_utf8.begin(), tokens_utf8.end());
  all_tokens.push_back(sep_utf8);

  auto token_word_group_indices = [&](const std::vector<std::string>& tokens,
                                      const std::vector<std::pair<int, int>>& offsets,
                                      const std::u32string& ref_text, const std::string& cls_t,
                                      const std::string& sep_t) {
    struct Idx {
      int i;
      int s;
      int e;
    };
    std::vector<Idx> idxs;
    for (std::size_t ti = 0; ti < tokens.size(); ++ti) {
      const std::string& tok = tokens[ti];
      const int s = offsets[ti].first;
      const int e = offsets[ti].second;
      if (tok == cls_t || tok == sep_t || (s == 0 && e == 0)) {
        continue;
      }
      idxs.push_back({static_cast<int>(ti), s, e});
    }
    std::vector<std::vector<int>> groups;
    std::vector<Idx> cur;
    auto u32_ref = ref_text;
    auto gap_str = [&](int pe, int s) {
      if (pe >= s) {
        return std::u32string{};
      }
      return u32_ref.substr(static_cast<std::size_t>(pe), static_cast<std::size_t>(s - pe));
    };
    auto u32_strip_empty = [](std::u32string g) {
      std::size_t a = 0;
      std::size_t b = g.size();
      while (a < b && is_space_u32(g[a])) {
        ++a;
      }
      while (b > a && is_space_u32(g[b - 1])) {
        --b;
      }
      return g.substr(a, b - a);
    };
    for (const Idx& item : idxs) {
      if (cur.empty()) {
        cur.push_back(item);
        continue;
      }
      const int pe = cur.back().e;
      const int s = item.s;
      const std::u32string gap = gap_str(pe, s);
      const bool new_word = !gap.empty() && u32_strip_empty(gap).empty();
      const char32_t last_ch = (pe > 0) ? u32_ref[static_cast<std::size_t>(pe - 1)] : U'\0';
      const char32_t first_ch =
          (s < static_cast<int>(u32_ref.size())) ? u32_ref[static_cast<std::size_t>(s)] : U'\0';
      const bool punct_break =
          (pe > 0) && !is_punct_char_word_group_u32(last_ch) && is_punct_char_word_group_u32(first_ch);
      if (new_word || punct_break) {
        std::vector<int> g;
        for (const Idx& t : cur) {
          g.push_back(t.i);
        }
        groups.push_back(std::move(g));
        cur = {item};
      } else {
        cur.push_back(item);
      }
    }
    if (!cur.empty()) {
      std::vector<int> g;
      for (const Idx& t : cur) {
        g.push_back(t.i);
      }
      groups.push_back(std::move(g));
    }
    return groups;
  };

  std::vector<std::vector<int>> groups =
      token_word_group_indices(all_tokens, all_offsets, ref, cls_utf8, sep_utf8);

  EncodedWp out;
  out.input_ids = std::move(all_ids);
  out.offsets_cp = std::move(all_offsets);
  out.ref_u32 = std::move(ref);
  out.word_groups = std::move(groups);
  return out;
}

const std::unordered_set<std::string>& ud_upos_set() {
  static const std::unordered_set<std::string> k = {
      "ADJ",  "ADP",  "ADV",  "AUX",  "CCONJ", "DET",  "INTJ", "NOUN", "NUM", "PART", "PRON",
      "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
  };
  return k;
}

std::string morph_label_to_upos(std::string label) {
  while (!label.empty() && (label.rfind("B-", 0) == 0 || label.rfind("I-", 0) == 0)) {
    label = label.substr(2);
  }
  std::size_t start = 0;
  while (start < label.size()) {
    std::size_t plus = label.find('+', start);
    const std::string part =
        (plus == std::string::npos) ? label.substr(start) : label.substr(start, plus - start);
    if (ud_upos_set().count(part) != 0) {
      return part;
    }
    if (plus == std::string::npos) {
      break;
    }
    start = plus + 1;
  }
  return "X";
}

std::unordered_map<std::string, int64_t> load_vocab_txt_stream(std::istream& in) {
  std::unordered_map<std::string, int64_t> vocab;
  std::string line;
  int64_t index = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    vocab[line] = index;
    ++index;
  }
  return vocab;
}

std::unordered_map<std::string, int64_t> load_vocab_txt_string(std::string_view utf8) {
  const std::string buf(utf8);
  std::istringstream in(buf);
  return load_vocab_txt_stream(in);
}

}  // namespace

std::filesystem::path default_korean_tok_pos_model_dir(const std::filesystem::path& g2p_data_root) {
  return g2p_data_root / "ko" / "roberta_korean_morph_upos_onnx";
}

KoreanTokPosOnnx::KoreanTokPosOnnx(std::filesystem::path model_dir, bool use_cuda)
    : KoreanTokPosOnnx(nullptr, std::string_view{}, std::move(model_dir), use_cuda) {}

KoreanTokPosOnnx::KoreanTokPosOnnx(const MoonshineG2POptions* opt, std::string_view onnx_bundle_key,
                                   std::filesystem::path model_dir_fallback, bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "moonshine_korean_tok_pos"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      model_dir_(std::move(model_dir_fallback)) {
  const MoonshineG2POptions* opt_ptr = opt;
  std::string meta_json;
  if (!bundle_load_utf8(opt_ptr, onnx_bundle_key, "meta.json", model_dir_, meta_json)) {
    throw std::runtime_error("KoreanTokPosOnnx: missing meta.json (memory or disk under " +
                             model_dir_.string() + ")");
  }
  const nlohmann::json meta = nlohmann::json::parse(meta_json);
  for (const auto& s : meta.at("id2label")) {
    id2label_.push_back(s.get<std::string>());
  }
  pad_id_ = meta.value("pad_token_id", 1);
  max_sequence_length_ = meta.value("max_sequence_length", 512);
  const std::string onnx_name = meta.value("onnx_model_file", std::string("model.ort"));
  if (!bundle_load_utf8(opt_ptr, onnx_bundle_key, "vocab.txt", model_dir_, cached_vocab_txt_)) {
    throw std::runtime_error("KoreanTokPosOnnx: missing vocab.txt");
  }
  if (!bundle_load_utf8(opt_ptr, onnx_bundle_key, "tokenizer_config.json", model_dir_,
                        cached_tokenizer_cfg_json_)) {
    throw std::runtime_error("KoreanTokPosOnnx: missing tokenizer_config.json");
  }
  if (bundle_load_binary(opt_ptr, onnx_bundle_key, onnx_name, model_dir_, onnx_model_storage_) &&
      !onnx_model_storage_.empty()) {
    const std::string model_map_key = g2p_bundle_file_key(onnx_bundle_key, onnx_name);
    session_ = open_session_memory(env_, onnx_model_storage_.data(), onnx_model_storage_.size(),
                                   use_cuda, opt_ptr, model_map_key);
  } else {
    const auto onnx_path = resolve_prefer_ort_model(model_dir_, onnx_name);
    if (!std::filesystem::is_regular_file(onnx_path)) {
      throw std::runtime_error("KoreanTokPosOnnx: missing " + onnx_path.string());
    }
    session_ = open_session(env_, onnx_path, use_cuda);
  }
  {
    Ort::AllocatorWithDefaultOptions alloc;
    auto out_ptr = session_->GetOutputNameAllocated(0, alloc);
    logits_output_name_ = std::string(out_ptr.get());
  }
}

std::string KoreanTokPosOnnx::format_annotated_line(
    const std::vector<std::pair<std::string, std::string>>& pairs) {
  std::string s;
  for (const auto& pr : pairs) {
    s += pr.first;
    s += '/';
    s += pr.second;
    s += ' ';
  }
  return s;
}

std::vector<std::pair<std::string, std::string>> KoreanTokPosOnnx::annotate(std::string_view text_utf8) {
  std::string trimmed = trim_ascii_ws_copy(text_utf8);
  if (trimmed.empty()) {
    return {};
  }

  if (cached_vocab_txt_.empty() || cached_tokenizer_cfg_json_.empty()) {
    throw std::runtime_error("KoreanTokPosOnnx: tokenizer assets not loaded");
  }
  const std::unordered_map<std::string, int64_t> vocab = load_vocab_txt_string(cached_vocab_txt_);
  const nlohmann::json cfg = nlohmann::json::parse(cached_tokenizer_cfg_json_);
  BasicTokCfg bcfg;
  bcfg.do_lower_case = cfg.value("do_lower_case", true);
  bcfg.tokenize_chinese_chars = cfg.value("tokenize_chinese_chars", true);
  if (cfg.contains("strip_accents") && !cfg["strip_accents"].is_null()) {
    bcfg.strip_accents = cfg["strip_accents"].get<bool>();
  }
  const std::string unk_utf8 = cfg.at("unk_token").get<std::string>();
  const std::string cls_utf8 = cfg.at("cls_token").get<std::string>();
  const std::string sep_utf8 = cfg.at("sep_token").get<std::string>();

  const std::u32string text_u32 = utf8_to_u32(trimmed);
  EncodedWp enc = encode_bert_wordpiece(text_u32, vocab, bcfg, unk_utf8, cls_utf8, sep_utf8);

  const int64_t T = static_cast<int64_t>(enc.input_ids.size());
  if (T > max_sequence_length_) {
    throw std::runtime_error("KoreanTokPosOnnx: sequence length > max_sequence_length not supported");
  }

  std::vector<int64_t> ids = enc.input_ids;
  std::vector<int64_t> mask(static_cast<size_t>(T), 1);
  for (int64_t i = 0; i < T; ++i) {
    if (ids[static_cast<size_t>(i)] == pad_id_) {
      mask[static_cast<size_t>(i)] = 0;
    }
  }

  const std::array<int64_t, 2> shape{1, T};
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      mem_, ids.data(), ids.size(), shape.data(), shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      mem_, mask.data(), mask.size(), shape.data(), shape.size()));

  const char* in_names[] = {"input_ids", "attention_mask"};
  const char* out_names[] = {logits_output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(),
                               out_names, 1);
  const float* logits = outputs[0].GetTensorData<float>();
  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto oshape = info.GetShape();
  if (oshape.size() != 3 || oshape[0] != 1 || oshape[1] != T) {
    throw std::runtime_error("KoreanTokPosOnnx: unexpected logits shape");
  }
  const int64_t num_labels = oshape[2];
  if (num_labels != static_cast<int64_t>(id2label_.size())) {
    throw std::runtime_error("KoreanTokPosOnnx: logits last dim != id2label size");
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  for (const std::vector<int>& g : enc.word_groups) {
    if (g.empty()) {
      continue;
    }
    std::vector<double> pooled(static_cast<size_t>(num_labels), 0.0);
    for (int ti : g) {
      if (ti < 0 || static_cast<int64_t>(ti) >= T) {
        continue;
      }
      const std::size_t base = static_cast<std::size_t>(ti) * static_cast<std::size_t>(num_labels);
      for (int64_t j = 0; j < num_labels; ++j) {
        pooled[static_cast<size_t>(j)] += static_cast<double>(logits[base + static_cast<size_t>(j)]);
      }
    }
    const double denom = static_cast<double>(g.size());
    int best = 0;
    double best_v = pooled[0] / denom;
    for (int64_t j = 1; j < num_labels; ++j) {
      const double v = pooled[static_cast<size_t>(j)] / denom;
      if (v > best_v) {
        best_v = v;
        best = static_cast<int>(j);
      }
    }
    const std::string& raw_label = id2label_[static_cast<size_t>(best)];
    const std::string upos = morph_label_to_upos(raw_label);
    const int st = enc.offsets_cp[static_cast<size_t>(g.front())].first;
    const int en = enc.offsets_cp[static_cast<size_t>(g.back())].second;
    std::u32string surf_u = enc.ref_u32.substr(static_cast<std::size_t>(st),
                                               static_cast<std::size_t>(en - st));
    pairs.emplace_back(u32_to_utf8(surf_u), upos);
  }
  return pairs;
}

}  // namespace moonshine_tts
