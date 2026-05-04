#include "chinese-numbers.h"

#include "utf8-utils.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {
namespace {

constexpr std::uint64_t kMaxCardinal = 10'000'000'000'000'000ULL;  // 10^16

char32_t digit_cp(unsigned d) {
  static constexpr char32_t k[] = {U'\u96F6', U'\u4E00', U'\u4E8C', U'\u4E09', U'\u56DB', U'\u4E94',
                                   U'\u516D', U'\u4E03', U'\u516B', U'\u4E5D'};
  if (d > 9) {
    return 0;
  }
  return k[d];
}

std::string cn_digit(unsigned d) {
  std::string o;
  utf8_append_codepoint(o, digit_cp(d));
  return o;
}

bool ends_with_ling_utf8(const std::string& s) {
  static const unsigned char kLing[] = {0xE9, 0x9B, 0xB6};
  return s.size() >= 3 && static_cast<unsigned char>(s[s.size() - 3]) == kLing[0] &&
         static_cast<unsigned char>(s[s.size() - 2]) == kLing[1] &&
         static_cast<unsigned char>(s[s.size() - 1]) == kLing[2];
}

std::string section_under_10000(unsigned n) {
  if (n == 0 || n >= 10000) {
    throw std::logic_error("section_under_10000");
  }
  std::string out;
  const unsigned q = n / 1000;
  unsigned r = n % 1000;
  if (q != 0) {
    out += cn_digit(q);
    utf8_append_codepoint(out, U'\u5343');  // 千
    if (r > 0 && r < 100) {
      out += cn_digit(0);
    }
  }
  const unsigned b = r / 100;
  r %= 100;
  if (b != 0) {
    out += cn_digit(b);
    utf8_append_codepoint(out, U'\u767E');  // 百
    if (r > 0 && r < 10) {
      out += cn_digit(0);
    }
  }
  const unsigned s = r / 10;
  const unsigned t = r % 10;
  const bool after_ling = ends_with_ling_utf8(out);
  const bool tens_prefix = (q != 0) || (b != 0) || after_ling;

  if (s == 0) {
    if (t != 0) {
      out += cn_digit(t);
    }
  } else if (s == 1) {
    if (t == 0) {
      if (tens_prefix) {
        out += cn_digit(1);
      }
      utf8_append_codepoint(out, U'\u5341');  // 十
    } else {
      if (after_ling) {
        out += cn_digit(1);
      }
      utf8_append_codepoint(out, U'\u5341');
      out += cn_digit(t);
    }
  } else {
    out += cn_digit(s);
    utf8_append_codepoint(out, U'\u5341');
    if (t != 0) {
      out += cn_digit(t);
    }
  }
  return out;
}

std::string int_to_han_u64(std::uint64_t n) {
  if (n == 0) {
    return cn_digit(0);
  }
  if (n >= kMaxCardinal) {
    const std::string s = std::to_string(n);
    std::string o;
    for (char c : s) {
      if (c >= '0' && c <= '9') {
        o += cn_digit(static_cast<unsigned>(c - '0'));
      }
    }
    return o;
  }
  std::vector<unsigned> low_first;
  std::uint64_t x = n;
  while (x > 0) {
    low_first.push_back(static_cast<unsigned>(x % 10000U));
    x /= 10000U;
  }
  std::vector<unsigned> gs(low_first.rbegin(), low_first.rend());
  static constexpr char32_t kUnits[] = {U'\0', U'\u4E07', U'\u4EBF', U'\u5146', U'\u4EAC', U'\u57C3'};
  const size_t m = gs.size();
  std::string out;
  bool zero_pending = false;
  for (size_t i = 0; i < m; ++i) {
    const unsigned g = gs[i];
    if (g == 0) {
      if (!out.empty()) {
        zero_pending = true;
      }
      continue;
    }
    if (zero_pending) {
      out += cn_digit(0);
      zero_pending = false;
    }
    if (i > 0 && g < 1000 && !out.empty()) {
      out += cn_digit(0);
    }
    out += section_under_10000(g);
    const size_t ui = m - 1 - i;
    if (ui < sizeof(kUnits) / sizeof(kUnits[0]) && kUnits[ui] != U'\0') {
      utf8_append_codepoint(out, kUnits[ui]);
    }
  }
  return out;
}

bool ascii_digit_string(std::string_view t, std::uint64_t& out_val) {
  if (t.empty()) {
    return false;
  }
  out_val = 0;
  for (char c : t) {
    if (c < '0' || c > '9') {
      return false;
    }
    const unsigned d = static_cast<unsigned>(c - '0');
    if (out_val > (std::numeric_limits<std::uint64_t>::max() - d) / 10) {
      return false;
    }
    out_val = out_val * 10 + d;
  }
  return true;
}

}  // namespace

std::string int_to_mandarin_cardinal_han(std::uint64_t n) { return int_to_han_u64(n); }

std::optional<std::string> arabic_numeral_token_to_han(std::string_view token_sv) {
  std::string token(trim_ascii_ws_copy(token_sv));
  if (token.empty()) {
    return std::nullopt;
  }
  // Fullwidth digits → ASCII where applicable
  {
    std::string norm;
    for (size_t i = 0; i < token.size();) {
      char32_t cp = 0;
      size_t adv = 0;
      if (!utf8_decode_at(token, i, cp, adv)) {
        norm.push_back(token[i]);
        ++i;
        continue;
      }
      if (cp >= U'\uFF10' && cp <= U'\uFF19') {
        norm.push_back(static_cast<char>('0' + (cp - U'\uFF10')));
      } else {
        utf8_append_codepoint(norm, cp);
      }
      i += adv;
    }
    token = std::move(norm);
  }
  // strip thousands separators
  {
    std::string stripped;
    for (char c : token) {
      if (c == ',' || c == '_' || c == ' ') {
        continue;
      }
      stripped.push_back(c);
    }
    token = std::move(stripped);
  }

  bool neg = false;
  if (!token.empty() && (token.front() == '+' || token.front() == '-')) {
    neg = token.front() == '-';
    token.erase(token.begin());
  }
  if (token.empty()) {
    return std::nullopt;
  }

  char dec_sep = '\0';
  for (char c : token) {
    if (c == '.' || c == ',') {
      if (dec_sep != '\0') {
        return std::nullopt;
      }
      dec_sep = c;
    }
  }

  std::string whole;
  std::string frac;
  if (dec_sep != '\0') {
    const auto p = token.find(dec_sep);
    whole = token.substr(0, p);
    frac = token.substr(p + 1);
    if (whole.empty()) {
      whole = "0";
    }
    for (char c : frac) {
      if (c < '0' || c > '9') {
        return std::nullopt;
      }
    }
  } else {
    whole = token;
  }

  for (char c : whole) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
  }

  std::string body;
  if (neg) {
    utf8_append_codepoint(body, U'\u8D1F');  // 负
  }

  if (!frac.empty()) {
    std::uint64_t wv = 0;
    if (!ascii_digit_string(whole, wv)) {
      return std::nullopt;
    }
    body += int_to_han_u64(wv);
    utf8_append_codepoint(body, U'\u70B9');  // 点
    for (char c : frac) {
      body += cn_digit(static_cast<unsigned>(c - '0'));
    }
    return body;
  }

  if (whole.size() > 1 && whole.front() == '0') {
    for (char c : whole) {
      if (c < '0' || c > '9') {
        return std::nullopt;
      }
      body += cn_digit(static_cast<unsigned>(c - '0'));
    }
    return body;
  }

  std::uint64_t wv = 0;
  if (!ascii_digit_string(whole, wv)) {
    return std::nullopt;
  }
  body += int_to_han_u64(wv);
  return body;
}

}  // namespace moonshine_tts
