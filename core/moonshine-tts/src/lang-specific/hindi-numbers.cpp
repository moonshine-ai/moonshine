#include "hindi-numbers.h"
#include "utf8-utils.h"

#include <cctype>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>

namespace moonshine_tts {
namespace {

constexpr const char* kDigitWord[10] = {
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"};

constexpr const char* kTeens[10] = {
    "दस",     "ग्यारह", "बारह",   "तेरह",     "चौदह",   "पंद्रह", "सोलह",
    "सत्रह",   "अठारह",   "उन्नीस"};

constexpr const char* kTens[10] = {
    "",         "",         "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"};

// Rows for 21–29 … 91–99 (tens index 2..9 → row tens-2)
constexpr const char* kUnitsAfterTen[8][9] = {
    {"इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस"},
    {"इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस"},
    {"इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास"},
    {"इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ"},
    {"इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर"},
    {"इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी"},
    {"इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अठासी", "नवासी"},
    {"इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"},
};

void append_join(std::vector<std::string>& out, const std::vector<std::string>& add) {
  for (const auto& s : add) {
    out.push_back(s);
  }
}

std::vector<std::string> under_100(int n) {
  if (n < 0 || n >= 100) {
    return {};
  }
  if (n < 10) {
    return {kDigitWord[n]};
  }
  if (n < 20) {
    return {kTeens[n - 10]};
  }
  if (n % 10 == 0) {
    return {kTens[n / 10]};
  }
  const int tens = n / 10;
  const int unit = n % 10;
  return {kUnitsAfterTen[tens - 2][unit - 1]};
}

std::vector<std::string> tokens_0_999(int n) {
  if (n < 0 || n > 999) {
    return {};
  }
  if (n == 0) {
    return {"शून्य"};
  }
  const int h = n / 100;
  const int r = n % 100;
  std::vector<std::string> parts;
  if (h > 0) {
    if (h == 1) {
      parts.emplace_back("सौ");
    } else {
      parts.emplace_back(kDigitWord[h]);
      parts.emplace_back("सौ");
    }
    if (r == 0) {
      return parts;
    }
  }
  append_join(parts, under_100(r));
  return parts;
}

std::vector<std::string> below_1_000_000_tokens(int n) {
  if (n < 0 || n >= 1'000'000) {
    return {};
  }
  if (n < 1000) {
    return tokens_0_999(n);
  }
  const int q = n / 1000;
  const int r = n % 1000;
  std::vector<std::string> left;
  if (q == 1) {
    left.emplace_back("हज़ार");
  } else {
    append_join(left, tokens_0_999(q));
    left.emplace_back("हज़ार");
  }
  if (r == 0) {
    return left;
  }
  if (r < 100) {
    append_join(left, under_100(r));
    return left;
  }
  append_join(left, tokens_0_999(r));
  return left;
}

std::string join_space(const std::vector<std::string>& v) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += v[i];
  }
  return o;
}

bool all_ascii_digits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (unsigned char c : s) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string expand_cardinal_digits_to_hindi_words(std::string_view s) {
  if (!all_ascii_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      o += kDigitWord[static_cast<unsigned>(s[i] - '0')];
    }
    return o;
  }
  const long long n = std::stoll(std::string(s));
  if (n > 999'999) {
    return std::string(s);
  }
  if (n == 0) {
    return "शून्य";
  }
  return join_space(below_1_000_000_tokens(static_cast<int>(n)));
}

std::string expand_hindi_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"((\b)(\d+)-(\d+)(\b))");
  static const std::regex digit_re(R"((\b)(\d+)(\b))");
  std::string out;
  out.reserve(text.size() + 64);
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t last = 0;
  for (; it != end; ++it) {
    const auto& m = *it;
    out.append(text, last, m.position() - last);
    const std::string a = m.str(2);
    const std::string b = m.str(3);
    out += expand_cardinal_digits_to_hindi_words(a);
    out += " - ";
    out += expand_cardinal_digits_to_hindi_words(b);
    last = m.position() + m.length();
  }
  out.append(text, last, std::string::npos);
  text = std::move(out);
  out.clear();
  last = 0;
  std::sregex_iterator it2(text.begin(), text.end(), digit_re);
  for (; it2 != end; ++it2) {
    const auto& m = *it2;
    out.append(text, last, m.position() - last);
    out += expand_cardinal_digits_to_hindi_words(m.str(2));
    last = m.position() + m.length();
  }
  out.append(text, last, std::string::npos);
  return out;
}

std::string expand_devanagari_digit_runs_in_text(std::string text) {
  const std::u32string u32 = utf8_str_to_u32(text);
  std::string result;
  result.reserve(text.size() + 32);
  size_t i = 0;
  while (i < u32.size()) {
    const char32_t cp = u32[i];
    if (cp >= 0x0966 && cp <= 0x096F) {
      size_t j = i;
      while (j < u32.size() && u32[j] >= 0x0966 && u32[j] <= 0x096F) {
        ++j;
      }
      std::string ascii;
      for (size_t k = i; k < j; ++k) {
        ascii.push_back(static_cast<char>('0' + static_cast<int>(u32[k] - 0x0966)));
      }
      result += expand_cardinal_digits_to_hindi_words(ascii);
      i = j;
    } else {
      utf8_append_codepoint(result, cp);
      ++i;
    }
  }
  return result;
}

}  // namespace moonshine_tts
