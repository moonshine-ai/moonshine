#include "korean-numbers.h"
#include "utf8-utils.h"

#include <cctype>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {
namespace {

constexpr std::uint64_t kMaxCardinal = 10000000000000000ULL;  // 10^16

constexpr const char* kSino[] = {"영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"};

bool is_ascii_digit(char c) {
  return c >= '0' && c <= '9';
}

bool thousands_lookahead_ok(std::string_view suf) {
  size_t pos = 0;
  const size_t n = suf.size();
  while (pos + 3 <= n && is_ascii_digit(suf[pos]) && is_ascii_digit(suf[pos + 1]) &&
         is_ascii_digit(suf[pos + 2])) {
    pos += 3;
    if (pos == n) {
      return true;
    }
    if (!is_ascii_digit(static_cast<unsigned char>(suf[pos]))) {
      return true;
    }
  }
  return false;
}

std::string strip_thousands_commas(std::string_view raw) {
  std::string out;
  out.reserve(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] == ',' && i > 0 && is_ascii_digit(raw[i - 1]) &&
        thousands_lookahead_ok(raw.substr(i + 1))) {
      continue;
    }
    out.push_back(raw[i]);
  }
  return out;
}

std::string normalize_numeral_token_string(std::string_view raw) {
  std::string t = trim_ascii_ws_copy(raw);
  std::string v;
  v.reserve(t.size());
  for (unsigned char uc : t) {
    const char c = static_cast<char>(uc);
    if (c == '_' || std::isspace(uc) != 0) {
      continue;
    }
    v.push_back(c);
  }
  return strip_thousands_commas(v);
}

std::string hangul_digits_only(std::string_view s) {
  std::string out;
  for (char c : s) {
    if (is_ascii_digit(static_cast<unsigned char>(c)) != 0) {
      out += kSino[static_cast<unsigned>(c - '0')];
    }
  }
  return out;
}

std::string section_under_10000(unsigned n) {
  if (n <= 0 || n >= 10000) {
    return {};
  }
  const unsigned q = n / 1000;
  const unsigned r = n % 1000;
  const unsigned b = r / 100;
  const unsigned r2 = r % 100;
  const unsigned s = r2 / 10;
  const unsigned t = r2 % 10;
  std::string parts;
  if (q > 0) {
    if (q == 1) {
      parts += "천";
    } else {
      parts += kSino[q];
      parts += "천";
    }
  }
  if (b > 0) {
    if (b == 1) {
      parts += "백";
    } else {
      parts += kSino[b];
      parts += "백";
    }
  }
  if (s == 0) {
    if (t != 0) {
      parts += kSino[t];
    }
  } else if (s == 1) {
    if (t == 0) {
      parts += "십";
    } else {
      parts += "십";
      parts += kSino[t];
    }
  } else {
    parts += kSino[s];
    parts += "십";
    if (t != 0) {
      parts += kSino[t];
    }
  }
  return parts;
}

bool parse_uint_strict(std::string_view sv, std::uint64_t& out) {
  if (sv.empty()) {
    return false;
  }
  std::uint64_t v = 0;
  for (char c : sv) {
    if (!is_ascii_digit(static_cast<unsigned char>(c))) {
      return false;
    }
    const int d = c - '0';
    if (v > (std::numeric_limits<std::uint64_t>::max() - static_cast<unsigned>(d)) / 10ULL) {
      return false;
    }
    v = v * 10ULL + static_cast<std::uint64_t>(d);
  }
  out = v;
  return true;
}

}  // namespace

std::string int_to_sino_korean_hangul(std::uint64_t n) {
  if (n == 0) {
    return "영";
  }
  if (n >= kMaxCardinal) {
    return hangul_digits_only(std::to_string(n));
  }
  std::vector<unsigned> low_first;
  std::uint64_t x = n;
  while (x > 0) {
    low_first.push_back(static_cast<unsigned>(x % 10000ULL));
    x /= 10000ULL;
  }
  std::vector<unsigned> gs(low_first.rbegin(), low_first.rend());
  static constexpr const char* units[] = {"", "만", "억", "조", "경"};
  std::string parts;
  bool zero_pending = false;
  for (size_t i = 0; i < gs.size(); ++i) {
    const unsigned g = gs[i];
    if (g == 0) {
      if (!parts.empty()) {
        zero_pending = true;
      }
      continue;
    }
    if (zero_pending) {
      parts += "영";
      zero_pending = false;
    }
    const size_t ui = gs.size() - 1 - i;
    const char* u = ui < 5 ? units[ui] : units[4];
    if (std::strcmp(u, "만") == 0 && g == 1) {
      parts += "만";
    } else {
      parts += section_under_10000(g);
      parts += u;
    }
  }
  return parts;
}

std::optional<std::vector<std::string>> korean_reading_fragments_from_ascii_numeral_token(
    std::string_view token) {
  const std::string raw = normalize_numeral_token_string(token);
  if (raw.empty()) {
    return std::nullopt;
  }

  std::optional<char> sign;
  size_t i = 0;
  if (raw[0] == '+' || raw[0] == '-') {
    sign = raw[0];
    i = 1;
  }
  if (i >= raw.size()) {
    return std::nullopt;
  }

  size_t dot_or_comma = std::string::npos;
  for (size_t j = i; j < raw.size(); ++j) {
    if (raw[j] == '.' || raw[j] == ',') {
      dot_or_comma = j;
      break;
    }
    if (!is_ascii_digit(static_cast<unsigned char>(raw[j]))) {
      return std::nullopt;
    }
  }

  const bool neg = sign.has_value() && *sign == '-';
  std::vector<std::string> frags;

  if (dot_or_comma != std::string::npos) {
    const std::string_view whole = std::string_view(raw).substr(i, dot_or_comma - i);
    const std::string_view frac = std::string_view(raw).substr(dot_or_comma + 1);
    if (whole.empty() && frac.empty()) {
      return std::nullopt;
    }
    for (char c : frac) {
      if (!is_ascii_digit(static_cast<unsigned char>(c))) {
        return std::nullopt;
      }
    }
    if (whole.size() > 1 && whole[0] == '0') {
      return std::nullopt;
    }
    std::uint64_t wv = 0;
    if (!whole.empty()) {
      if (!parse_uint_strict(whole, wv)) {
        return std::nullopt;
      }
    }
    std::string body = int_to_sino_korean_hangul(wv) + "점" + hangul_digits_only(frac);
    if (neg) {
      frags.push_back("마이너스");
      frags.push_back(std::move(body));
    } else {
      frags.push_back(std::move(body));
    }
    return frags;
  }

  const std::string_view whole = std::string_view(raw).substr(i);
  if (whole.empty()) {
    return std::nullopt;
  }
  for (char c : whole) {
    if (!is_ascii_digit(static_cast<unsigned char>(c))) {
      return std::nullopt;
    }
  }

  if (whole.size() > 1 && whole[0] == '0') {
    std::string seq = hangul_digits_only(whole);
    if (neg) {
      frags.push_back("마이너스");
      frags.push_back(std::move(seq));
    } else {
      frags.push_back(std::move(seq));
    }
    return frags;
  }

  std::uint64_t wv = 0;
  if (!parse_uint_strict(whole, wv)) {
    return std::nullopt;
  }
  std::string body = int_to_sino_korean_hangul(wv);
  if (neg) {
    frags.push_back("마이너스");
    frags.push_back(std::move(body));
  } else {
    frags.push_back(std::move(body));
  }
  return frags;
}

bool is_ascii_numeral_token(std::string_view token) {
  return korean_reading_fragments_from_ascii_numeral_token(token).has_value();
}

}  // namespace moonshine_tts
