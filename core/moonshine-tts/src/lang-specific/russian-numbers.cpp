// Russian cardinal expansion (russian_numbers.py). #include from russian.cpp (same TU).

#include "utf8-utils.h"

#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool ru_ascii_all_digits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

// UTF-8 literals (same orthography as Python russian_numbers.py).
static const char* kRu0 = "\xd0\xbd\xd0\xbe\xd0\xbb\xd1\x8c";
static const char* kRuOnesM[10] = {
    nullptr,
    "\xd0\xbe\xd0\xb4\xd0\xb8\xd0\xbd",
    "\xd0\xb4\xd0\xb2\xd0\xb0",
    "\xd1\x82\xd1\x80\xd0\xb8",
    "\xd1\x87\xd0\xb5\xd1\x82\xd1\x8b\xd1\x80\xd0\xb5",
    "\xd0\xbf\xd1\x8f\xd1\x82\xd1\x8c",
    "\xd1\x88\xd0\xb5\xd1\x81\xd1\x82\xd1\x8c",
    "\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c",
    "\xd0\xb2\xd0\xbe\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c",
    "\xd0\xb4\xd0\xb5\xd0\xb2\xd1\x8f\xd1\x82\xd1\x8c",
};
static const char* kRuFem1 = "\xd0\xbe\xd0\xb4\xd0\xbd\xd0\xb0";
static const char* kRuFem2 = "\xd0\xb4\xd0\xb2\xd0\xb5";

static const char* kRuTeens[10] = {
    "\xd0\xb4\xd0\xb5\xd1\x81\xd1\x8f\xd1\x82\xd1\x8c",
    "\xd0\xbe\xd0\xb4\xd0\xb8\xd0\xbd\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd0\xb4\xd0\xb2\xd0\xb5\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x82\xd1\x80\xd0\xb8\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x87\xd0\xb5\xd1\x82\xd1\x8b\xd1\x80\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd0\xbf\xd1\x8f\xd1\x82\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x88\xd0\xb5\xd1\x81\xd1\x82\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x81\xd0\xb5\xd0\xbc\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd0\xb2\xd0\xbe\xd1\x81\xd0\xb5\xd0\xbc\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd0\xb4\xd0\xb5\xd0\xb2\xd1\x8f\xd1\x82\xd0\xbd\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
};

static const char* kRuTens[10] = {
    nullptr,
    nullptr,
    "\xd0\xb4\xd0\xb2\xd0\xb0\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x82\xd1\x80\xd0\xb8\xd0\xb4\xd1\x86\xd0\xb0\xd1\x82\xd1\x8c",
    "\xd1\x81\xd0\xbe\xd1\x80\xd0\xbe\xd0\xba",
    "\xd0\xbf\xd1\x8f\xd1\x82\xd1\x8c\xd0\xb4\xd0\xb5\xd1\x81\xd1\x8f\xd1\x82",
    "\xd1\x88\xd0\xb5\xd1\x81\xd1\x82\xd1\x8c\xd0\xb4\xd0\xb5\xd1\x81\xd1\x8f\xd1\x82",
    "\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c\xd0\xb4\xd0\xb5\xd1\x81\xd1\x8f\xd1\x82",
    "\xd0\xb2\xd0\xbe\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c\xd0\xb4\xd0\xb5\xd1\x81\xd1\x8f\xd1\x82",
    "\xd0\xb4\xd0\xb5\xd0\xb2\xd1\x8f\xd0\xbd\xd0\xbe\xd1\x81\xd1\x82\xd0\xbe",
};

static const char* kRuHundred[10] = {
    nullptr,
    "\xd1\x81\xd1\x82\xd0\xbe",
    "\xd0\xb4\xd0\xb2\xd0\xb5\xd1\x81\xd1\x82\xd0\xb8",
    "\xd1\x82\xd1\x80\xd0\xb8\xd1\x81\xd1\x82\xd0\xb0",
    "\xd1\x87\xd0\xb5\xd1\x82\xd1\x8b\xd1\x80\xd0\xb5\xd1\x81\xd1\x82\xd0\xb0",
    "\xd0\xbf\xd1\x8f\xd1\x82\xd1\x8c\xd1\x81\xd0\xbe\xd1\x82",
    "\xd1\x88\xd0\xb5\xd1\x81\xd1\x82\xd1\x8c\xd1\x81\xd0\xbe\xd1\x82",
    "\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c\xd1\x81\xd0\xbe\xd1\x82",
    "\xd0\xb2\xd0\xbe\xd1\x81\xd0\xb5\xd0\xbc\xd1\x8c\xd1\x81\xd0\xbe\xd1\x82",
    "\xd0\xb4\xd0\xb5\xd0\xb2\xd1\x8f\xd1\x82\xd1\x8c\xd1\x81\xd0\xbe\xd1\x82",
};

static const char* kRuTysyacha = "\xd1\x82\xd1\x8b\xd1\x81\xd1\x8f\xd1\x87\xd0\xb0";
static const char* kRuTysyachi = "\xd1\x82\xd1\x8b\xd1\x81\xd1\x8f\xd1\x87\xd0\xb8";
static const char* kRuTysyach = "\xd1\x82\xd1\x8b\xd1\x81\xd1\x8f\xd1\x87";

std::string ru_ones_digit(int n, bool feminine) {
  if (n < 1 || n > 9) {
    throw std::out_of_range("ru_ones_digit");
  }
  if (feminine) {
    if (n == 1) {
      return kRuFem1;
    }
    if (n == 2) {
      return kRuFem2;
    }
  }
  return kRuOnesM[n];
}

void ru_append_under_100(int n, bool feminine, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    throw std::out_of_range("ru_append_under_100");
  }
  if (n < 10) {
    out.push_back(ru_ones_digit(n, feminine));
    return;
  }
  if (n < 20) {
    out.emplace_back(kRuTeens[n - 10]);
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  out.emplace_back(kRuTens[t]);
  if (u != 0) {
    out.push_back(ru_ones_digit(u, feminine));
  }
}

void ru_append_cardinal_1_to_999(int n, bool feminine, std::vector<std::string>& out) {
  if (n < 1 || n > 999) {
    throw std::out_of_range("ru_append_cardinal_1_to_999");
  }
  if (n < 100) {
    ru_append_under_100(n, feminine, out);
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  out.emplace_back(kRuHundred[h]);
  if (r != 0) {
    ru_append_under_100(r, feminine, out);
  }
}

const char* ru_thousand_suffix(int q) {
  const int m100 = q % 100;
  if (m100 >= 11 && m100 <= 14) {
    return kRuTysyach;
  }
  const int k = q % 10;
  if (k == 1) {
    return kRuTysyacha;
  }
  if (k >= 2 && k <= 4) {
    return kRuTysyachi;
  }
  return kRuTysyach;
}

void ru_append_below_1_000_000(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("ru_append_below_1_000_000");
  }
  if (n < 1000) {
    ru_append_cardinal_1_to_999(n, false, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  ru_append_cardinal_1_to_999(q, true, out);
  out.emplace_back(ru_thousand_suffix(q));
  if (r != 0) {
    ru_append_cardinal_1_to_999(r, false, out);
  }
}

std::string expand_cardinal_digits_to_russian_words(std::string_view s) {
  if (!ru_ascii_all_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      const int d = s[i] - '0';
      o += (d == 0) ? kRu0 : kRuOnesM[static_cast<size_t>(d)];
    }
    return o;
  }
  long long n = 0;
  for (char c : s) {
    n = n * 10 + (c - '0');
  }
  if (n > 999'999) {
    return std::string(s);
  }
  if (n == 0) {
    return kRu0;
  }
  std::vector<std::string> parts;
  ru_append_below_1_000_000(static_cast<int>(n), parts);
  std::string o;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += parts[i];
  }
  return o;
}

std::string expand_russian_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    const size_t g1s = static_cast<size_t>(m.position(1));
    const size_t g1e = g1s + m.str(1).size();
    const size_t g2s = static_cast<size_t>(m.position(2));
    const size_t g2e = g2s + m.str(2).size();
    if (moonshine_tts::digit_ascii_span_expandable_python_w(text, g1s, g1e) &&
        moonshine_tts::digit_ascii_span_expandable_python_w(text, g2s, g2e)) {
      out += expand_cardinal_digits_to_russian_words(m.str(1));
      out += " - ";
      out += expand_cardinal_digits_to_russian_words(m.str(2));
    } else {
      out += m.str();
    }
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  text = std::move(out);
  out.clear();
  pos = 0;
  std::sregex_iterator it2(text.begin(), text.end(), dig_re);
  for (; it2 != end; ++it2) {
    const std::smatch& m = *it2;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    const size_t gs = static_cast<size_t>(m.position());
    const size_t ge = gs + m.str().size();
    if (moonshine_tts::digit_ascii_span_expandable_python_w(text, gs, ge)) {
      out += expand_cardinal_digits_to_russian_words(m.str());
    } else {
      out += m.str();
    }
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

}  // namespace
