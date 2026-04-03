#include "english-numbers.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace moonshine_tts {
namespace {

const char* kUnits[] = {"ˈzɪroʊ", "wˈʌn", "tˈu", "θɹˈi", "fˈɔɹ", "fˈaɪv",
                        "sˈɪks", "sˈɛvən", "ˈeɪt", "nˈaɪn"};
const char* kTeens[] = {"tˈɛn",      "ɪlˈɛvən",   "twˈɛlv",    "θɝˈtin",   "fɔɹˈtin",
                        "fˈɪftin",  "sˈɪkstin",  "sˈɛvəntin", "ˈeɪtin",   "nˈaɪntin"};
const char* kTens[] = {nullptr, nullptr, "twˈɛnti", "θˈɝdi", "fˈɔɹti", "fˈɪfti",
                       "sˈɪksti", "sˈɛvənti", "ˈeɪti", "nˈaɪnti"};

const char* kDigitByDigit[] = {"ˈzɪroʊ", "ˈwʌn", "ˈtu", "ˈθɹi", "ˈfɔɹ", "ˈfaɪv",
                               "ˈsɪks", "ˈsɛvən", "ˈeɪt", "ˈnaɪn"};

std::string digit_sequence_ipa(std::string_view digits) {
  std::string out;
  for (char ch : digits) {
    if (ch >= '0' && ch <= '9') {
      if (!out.empty()) {
        out += "ˌ";
      }
      out += kDigitByDigit[static_cast<unsigned>(ch - '0')];
    }
  }
  return out;
}

std::string under_100_ipa(int n) {
  if (n < 10) {
    return kUnits[n];
  }
  if (n < 20) {
    return kTeens[n - 10];
  }
  int tens = n / 10;
  int u = n % 10;
  std::string t = kTens[tens];
  if (u == 0) {
    return t;
  }
  return t + std::string("ˌ") + kUnits[u];
}

std::string under_1000_ipa(int n) {
  if (n < 100) {
    return under_100_ipa(n);
  }
  int h = n / 100;
  int r = n % 100;
  std::string head = std::string(kUnits[h]) + "ˌhˈʌndɹɪd";
  if (r == 0) {
    return head;
  }
  return head + "ˌ" + under_100_ipa(r);
}

std::optional<std::string> cardinal_non_negative_ipa(long long n) {
  if (n < 0) {
    return std::nullopt;
  }
  if (n == 0) {
    return std::string("ˈzɪroʊ");
  }
  if (n >= 1000000000000000LL) {
    return std::nullopt;
  }
  struct Scale {
    long long mag;
    const char* sfx;
  };
  const Scale sc[] = {{1000000000000LL, "ˌtɹˈɪljən"},
                      {1000000000LL, "ˌbˈɪljən"},
                      {1000000LL, "ˌmˈɪljən"},
                      {1000LL, "ˌθˈaʊzənd"}};
  long long rem = n;
  std::vector<std::string> parts;
  for (const Scale& x : sc) {
    if (rem >= x.mag) {
      const long long q = rem / x.mag;
      rem %= x.mag;
      if (q > 0) {
        parts.push_back(under_1000_ipa(static_cast<int>(q)) + x.sfx);
      }
    }
  }
  if (rem > 0) {
    parts.push_back(under_1000_ipa(static_cast<int>(rem)));
  }
  if (parts.empty()) {
    return under_1000_ipa(static_cast<int>(n));
  }
  std::string out = parts[0];
  for (size_t i = 1; i < parts.size(); ++i) {
    out += "ˌ";
    out += parts[i];
  }
  return out;
}

std::optional<std::string> integer_decimal_string_ipa(std::string s) {
  auto strip_sep = [](std::string& t) {
    std::string o;
    o.reserve(t.size());
    for (char c : t) {
      if (c == ',' || c == '_' || c == ' ') {
        continue;
      }
      o.push_back(c);
    }
    t = std::move(o);
  };
  strip_sep(s);
  if (s.empty()) {
    return std::nullopt;
  }
  bool neg = false;
  if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
    neg = (s[0] == '-');
    s.erase(0, 1);
  }
  if (s.empty()) {
    return std::nullopt;
  }
  size_t ndot = 0;
  for (char c : s) {
    if (c == '.') {
      ++ndot;
    }
  }
  if (ndot > 1) {
    return std::nullopt;
  }
  auto prefix_neg = [neg](std::string out) {
    if (neg) {
      return std::string("nˈɛɡətɪvˌ") + out;
    }
    return out;
  };
  if (s.find('.') != std::string::npos) {
    const size_t dot = s.find('.');
    std::string whole = s.substr(0, dot);
    std::string frac = s.substr(dot + 1);
    for (char c : whole) {
      if (c != '\0' && !std::isdigit(static_cast<unsigned char>(c))) {
        return std::nullopt;
      }
    }
    for (char c : frac) {
      if (!c) {
        break;
      }
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        return std::nullopt;
      }
    }
    std::string left;
    if (whole.empty()) {
      left = "ˈzɪroʊ";
    } else if (whole.size() > 1 && whole[0] == '0') {
      left = digit_sequence_ipa(whole);
    } else {
      long long n = std::stoll(whole);
      auto c = cardinal_non_negative_ipa(n);
      if (!c) {
        left = digit_sequence_ipa(whole);
      } else {
        left = *c;
      }
    }
    if (frac.empty()) {
      return prefix_neg(std::move(left));
    }
    return prefix_neg(left + "ˌˈpɔɪntˌ" + digit_sequence_ipa(frac));
  }
  if (!std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
    return std::nullopt;
  }
  if (s.size() > 1 && s[0] == '0') {
    return prefix_neg(digit_sequence_ipa(s));
  }
  long long n = std::stoll(s);
  auto c = cardinal_non_negative_ipa(n);
  if (!c) {
    return prefix_neg(digit_sequence_ipa(s));
  }
  return prefix_neg(std::move(*c));
}

}  // namespace

std::optional<std::string> english_number_token_ipa(std::string_view token) {
  std::string t(token.begin(), token.end());
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) {
    t.erase(0, 1);
  }
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) {
    t.pop_back();
  }
  if (t.empty()) {
    return std::nullopt;
  }
  return integer_decimal_string_ipa(std::move(t));
}

}  // namespace moonshine_tts
