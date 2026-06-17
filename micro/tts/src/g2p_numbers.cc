#include "g2p_numbers.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace tts {

namespace {

const char* kUnits[] = {
    "\u02C8z\u026Aro\u028A", "w\u02C8\u028Cn",        "t\u02C8u",
    "\u03B8\u0279\u02C8i",   "f\u02C8\u0254\u0279",   "f\u02C8a\u026Av",
    "s\u02C8\u026Aks",       "s\u02C8\u025Bv\u0259n", "\u02C8e\u026At",
    "n\u02C8a\u026An"};
const char* kTeens[] = {"t\u02C8\u025Bn",         "\u026Al\u02C8\u025Bv\u0259n",
                        "tw\u02C8\u025Blv",       "\u03B8\u025D\u02C8tin",
                        "f\u0254\u0279\u02C8tin", "f\u02C8\u026Aftin",
                        "s\u02C8\u026Akstin",     "s\u02C8\u025Bv\u0259ntin",
                        "\u02C8e\u026Atin",       "n\u02C8a\u026Antin"};
const char* kTens[] = {nullptr,
                       nullptr,
                       "tw\u02C8\u025Bnti",
                       "\u03B8\u02C8\u025Ddi",
                       "f\u02C8\u0254\u0279ti",
                       "f\u02C8\u026Afti",
                       "s\u02C8\u026Aksti",
                       "s\u02C8\u025Bv\u0259nti",
                       "\u02C8e\u026Ati",
                       "n\u02C8a\u026Anti"};
const char* kDigitByDigit[] = {
    "\u02C8z\u026Aro\u028A", "\u02C8w\u028Cn",        "\u02C8tu",
    "\u02C8\u03B8\u0279i",   "\u02C8f\u0254\u0279",   "\u02C8fa\u026Av",
    "\u02C8s\u026Aks",       "\u02C8s\u025Bv\u0259n", "\u02C8e\u026At",
    "\u02C8na\u026An"};

constexpr std::string_view kStress{"\u02CC"};  // secondary stress (separator)

std::string DigitSequenceIpa(std::string_view digits) {
  std::string out;
  for (char ch : digits) {
    if (ch >= '0' && ch <= '9') {
      if (!out.empty()) out += kStress;
      out += kDigitByDigit[static_cast<unsigned>(ch - '0')];
    }
  }
  return out;
}

std::string Under100Ipa(int n) {
  if (n < 10) return kUnits[n];
  if (n < 20) return kTeens[n - 10];
  const int tens = n / 10;
  const int u = n % 10;
  std::string t = kTens[tens];
  if (u == 0) return t;
  return t + std::string(kStress) + kUnits[u];
}

std::string Under1000Ipa(int n) {
  if (n < 100) return Under100Ipa(n);
  const int h = n / 100;
  const int r = n % 100;
  std::string head = std::string(kUnits[h]) + "\u02CCh\u02C8\u028Cndr\u026Ad";
  if (r == 0) return head;
  return head + std::string(kStress) + Under100Ipa(r);
}

bool CardinalNonNegativeIpa(long long n, std::string* out) {
  if (n < 0) return false;
  if (n == 0) {
    *out = "\u02C8z\u026Aro\u028A";
    return true;
  }
  if (n >= 1000000000000000LL) return false;
  struct Scale {
    long long mag;
    const char* sfx;
  };
  const Scale sc[] = {{1000000000000LL, "\u02CCtr\u02C8\u026Alj\u0259n"},
                      {1000000000LL, "\u02CCb\u02C8\u026Alj\u0259n"},
                      {1000000LL, "\u02CCm\u02C8\u026Alj\u0259n"},
                      {1000LL, "\u02CC\u03B8\u02C8a\u028Az\u0259nd"}};
  long long rem = n;
  std::vector<std::string> parts;
  for (const Scale& x : sc) {
    if (rem >= x.mag) {
      const long long q = rem / x.mag;
      rem %= x.mag;
      if (q > 0) parts.push_back(Under1000Ipa(static_cast<int>(q)) + x.sfx);
    }
  }
  if (rem > 0) parts.push_back(Under1000Ipa(static_cast<int>(rem)));
  if (parts.empty()) {
    *out = Under1000Ipa(static_cast<int>(n));
    return true;
  }
  std::string s = parts[0];
  for (size_t i = 1; i < parts.size(); ++i) {
    s += kStress;
    s += parts[i];
  }
  *out = std::move(s);
  return true;
}

bool IntegerDecimalStringIpa(std::string s, std::string* out) {
  // Strip grouping separators.
  std::string stripped;
  stripped.reserve(s.size());
  for (char c : s) {
    if (c == ',' || c == '_' || c == ' ') continue;
    stripped.push_back(c);
  }
  s = std::move(stripped);
  if (s.empty()) return false;

  bool neg = false;
  if (s[0] == '+' || s[0] == '-') {
    neg = (s[0] == '-');
    s.erase(0, 1);
  }
  if (s.empty()) return false;

  int ndot = 0;
  for (char c : s) {
    if (c == '.') ++ndot;
  }
  if (ndot > 1) return false;

  auto prefix_neg = [neg](std::string v) {
    return neg ? std::string("n\u02C8\u025B\u0261\u0259t\u026Av\u02CC") + v : v;
  };

  const size_t dot = s.find('.');
  if (dot != std::string::npos) {
    std::string whole = s.substr(0, dot);
    std::string frac = s.substr(dot + 1);
    for (char c : whole) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    for (char c : frac) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    std::string left;
    if (whole.empty()) {
      left = "\u02C8z\u026Aro\u028A";
    } else if (whole.size() > 1 && whole[0] == '0') {
      left = DigitSequenceIpa(whole);
    } else {
      long long n = 0;
      for (char c : whole) n = n * 10 + (c - '0');
      std::string c;
      left = CardinalNonNegativeIpa(n, &c) ? c : DigitSequenceIpa(whole);
    }
    if (frac.empty()) {
      *out = prefix_neg(std::move(left));
      return true;
    }
    *out = prefix_neg(left + "\u02CC\u02C8p\u0254\u026Ant\u02CC" +
                      DigitSequenceIpa(frac));
    return true;
  }

  if (!std::all_of(s.begin(), s.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
      })) {
    return false;
  }
  if (s.size() > 1 && s[0] == '0') {
    *out = prefix_neg(DigitSequenceIpa(s));
    return true;
  }
  long long n = 0;
  for (char c : s) n = n * 10 + (c - '0');
  std::string c;
  *out = prefix_neg(CardinalNonNegativeIpa(n, &c) ? c : DigitSequenceIpa(s));
  return true;
}

}  // namespace

bool NumberWordToIpa(std::string_view token, std::string* ipa) {
  std::string t(token);
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) {
    t.erase(0, 1);
  }
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) {
    t.pop_back();
  }
  if (t.empty()) return false;
  return IntegerDecimalStringIpa(std::move(t), ipa);
}

}  // namespace tts
