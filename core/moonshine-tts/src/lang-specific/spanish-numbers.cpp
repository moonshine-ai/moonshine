// Spanish cardinal expansion (spanish_numbers.py). #include from spanish.cpp (same TU).

#include "utf8-utils.h"

#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool es_ascii_all_digits(std::string_view s) {
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

static const char* kEsDigit[] = {"cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
                                 "nueve"};

static const char* kEsUnder30[] = {
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "diecis\xc3\xa9is",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintid\xc3\xb3s",
    "veintitr\xc3\xa9s",
    "veinticuatro",
    "veinticinco",
    "veintis\xc3\xa9is",
    "veintisiete",
    "veintiocho",
    "veintinueve",
};

static const char* kEsTens[] = {"", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
                                "ochenta", "noventa"};

static const char* kEsHundreds[] = {"", "", "doscientos", "trescientos", "cuatrocientos", "quinientos",
                                    "seiscientos", "setecientos", "ochocientos", "novecientos"};

void es_append_under_100(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    throw std::out_of_range("es_append_under_100");
  }
  if (n < 10) {
    out.emplace_back(kEsDigit[n]);
    return;
  }
  if (n < 30) {
    out.emplace_back(kEsUnder30[n]);
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  out.emplace_back(kEsTens[t]);
  if (u != 0) {
    out.emplace_back("y");
    out.emplace_back(kEsDigit[u]);
  }
}

void es_append_below_1000(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1000) {
    throw std::out_of_range("es_append_below_1000");
  }
  if (n < 100) {
    es_append_under_100(n, out);
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h == 1) {
    if (r == 0) {
      out.emplace_back("cien");
      return;
    }
    out.emplace_back("ciento");
    es_append_under_100(r, out);
    return;
  }
  out.emplace_back(kEsHundreds[h]);
  if (r != 0) {
    es_append_under_100(r, out);
  }
}

void es_append_below_1_000_000(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("es_append_below_1_000_000");
  }
  if (n < 1000) {
    es_append_below_1000(n, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  if (q == 1) {
    out.emplace_back("mil");
  } else {
    es_append_below_1000(q, out);
    out.emplace_back("mil");
  }
  if (r != 0) {
    es_append_below_1000(r, out);
  }
}

std::string expand_cardinal_digits_to_spanish_words(std::string_view s) {
  if (!es_ascii_all_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      o += kEsDigit[static_cast<size_t>(s[i] - '0')];
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
    return "cero";
  }
  std::vector<std::string> parts;
  es_append_below_1_000_000(static_cast<int>(n), parts);
  std::string o;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += parts[i];
  }
  return o;
}

std::string expand_spanish_digit_tokens_in_text(std::string text) {
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
      out += expand_cardinal_digits_to_spanish_words(m.str(1));
      out += " - ";
      out += expand_cardinal_digits_to_spanish_words(m.str(2));
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
      out += expand_cardinal_digits_to_spanish_words(m.str());
    } else {
      out += m.str();
    }
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

}  // namespace
