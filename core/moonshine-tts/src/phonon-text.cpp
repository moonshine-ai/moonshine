#include "phonon-text.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sentence-splitter.h"
#include "utf8-utils.h"

namespace moonshine_tts {

namespace {

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

bool is_double_quote(char32_t c) {
  switch (c) {
    case U'»':
    case U'«':
    case U'‹':
    case U'›':
    case U'“':
    case U'”':
    case U'„':
    case U'‟':
    case U'″':
    case U'‶':
    case U'⹂':
    case U'❝':
    case U'❞':
    case U'❠':
    case U'〝':
    case U'〞':
    case U'〟':
    case U'＂':
      return true;
    default:
      return false;
  }
}

bool is_single_quote(char32_t c) {
  switch (c) {
    case U'‘':
    case U'’':
    case U'‚':
    case U'‛':
    case U'′':
    case U'‵':
    case U'❛':
    case U'❜':
    case U'❟':
    case U'＇':
    case U'ʼ':
    case U'ʻ':
    case U'ʹ':
    case U'´':
    case U'`':
      return true;
    default:
      return false;
  }
}

bool is_emoji(char32_t c) {
  return (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x2600 && c <= 0x27BF) ||
         (c >= 0x2B00 && c <= 0x2BFF) || c == 0xFE0F || c == 0x200D;
}

bool is_space(char32_t c) {
  return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' ||
         c == U'\v' || c == 0xA0 || c == 0x2007 || c == 0x202F || c == 0x3000;
}

bool is_control(char32_t c) { return c < 0x20 || (c >= 0x7F && c < 0xA0); }

bool is_ascii_digit(char32_t c) { return c >= U'0' && c <= U'9'; }

bool is_ascii_alpha(char32_t c) {
  return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

/// Letters and digits in the scripts the model's tokenizer has pieces for.
/// Good enough to decide whether a prompt ends in a word and needs a full
/// stop, which is all it is used for.
bool is_alnum(char32_t c) {
  return is_ascii_digit(c) || is_ascii_alpha(c) ||
         (c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7) ||
         (c >= 0x370 && c <= 0x3FF) || (c >= 0x400 && c <= 0x4FF) ||
         (c >= 0x900 && c <= 0x97F) || (c >= 0x3040 && c <= 0x30FF) ||
         (c >= 0x3400 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7AF);
}

/// Letters with a diacritic the tokenizer has no piece for, paired with the
/// letter they fold to, as in the reference runtime's ``FOLDS`` table.
constexpr std::u32string_view kFolds =
    U"ÌIÒOÝYìiòoýyÿyĀAāaĂAăaĄAąaĆCćcĈCĉcĊCċcČCčcĎDďdĒEēeĔEĕeĖEėeĘEęeĚEěeĜGĝgĞGğ"
    U"gĠGġgĢGģgĤHĥhĨIĩiĪIīiĬIĭiĮIįiİIĴJĵjĶKķkĹLĺlĻLļlĽLľlŃNńnŅNņnŇNňnŌOōoŎOŏo"
    U"ŐOőoŔRŕrŖRŗrŘRřrŚSśsŜSŝsŞSşsŠSšsŢTţtŤTťtŨUũuŪUūuŬUŭuŮUůuŰUűuŲUųuŴWŵwŶYŷy"
    U"ŸYŹZźzŻZżzŽZžzƠOơoƯUưuǍAǎaǏIǐiǑOǒoǓUǔuǕUǖuǗUǘuǙUǚuǛUǜuǞAǟaǠAǡaǦGǧgǨKǩkǪO"
    U"ǫoǬOǭoǰjǴGǵgǸNǹnǺAǻaȀAȁaȂAȃaȄEȅeȆEȇeȈIȉiȊIȋiȌOȍoȎOȏoȐRȑrȒRȓrȔUȕuȖUȗuȘSșs"
    U"ȚTțtȞHȟhȦAȧaȨEȩeȪOȫoȬOȭoȮOȯoȰOȱoȲYȳyḀAḁaḂBḃbḄBḅbḆBḇbḈCḉcḊDḋdḌDḍdḎDḏdḐDḑd"
    U"ḒDḓdḔEḕeḖEḗeḘEḙeḚEḛeḜEḝeḞFḟfḠGḡgḢHḣhḤHḥhḦHḧhḨHḩhḪHḫhḬIḭiḮIḯiḰKḱkḲKḳkḴKḵk"
    U"ḶLḷlḸLḹlḺLḻlḼLḽlḾMḿmṀMṁmṂMṃmṄNṅnṆNṇnṈNṉnṊNṋnṌOṍoṎOṏoṐOṑoṒOṓoṔPṕpṖPṗpṘRṙr"
    U"ṚRṛrṜRṝrṞRṟrṠSṡsṢSṣsṤSṥsṦSṧsṨSṩsṪTṫtṬTṭtṮTṯtṰTṱtṲUṳuṴUṵuṶUṷuṸUṹuṺUṻuṼVṽv"
    U"ṾVṿvẀWẁwẂWẃwẄWẅwẆWẇwẈWẉwẊXẋxẌXẍxẎYẏyẐZẑzẒZẓzẔZẕzẖhẗtẘwẙyẠAạaẢAảaẤAấaẦAầa"
    U"ẨAẩaẪAẫaẬAậaẮAắaẰAằaẲAẳaẴAẵaẶAặaẸEẹeẺEẻeẼEẽeẾEếeỀEềeỂEểeỄEễeỆEệeỈIỉiỊIịi"
    U"ỌOọoỎOỏoỐOốoỒOồoỔOổoỖOỗoỘOộoỚOớoỜOờoỞOởoỠOỡoỢOợoỤUụuỦUủuỨUứuỪUừuỬUửuỮUữu"
    U"ỰUựuỲYỳyỴYỵyỶYỷyỸYỹy";

std::optional<char32_t> unaccented(char32_t c) {
  for (size_t i = 0; i + 1 < kFolds.size(); i += 2) {
    if (kFolds[i] == c) {
      return kFolds[i + 1];
    }
  }
  return std::nullopt;
}

/// Math symbols read out in words.
std::optional<std::u32string_view> symbol_words(char32_t c) {
  switch (c) {
    case U'×':
      return U"times";
    case U'÷':
      return U"divided by";
    case U'≤':
      return U"less than or equal to";
    case U'≥':
      return U"greater than or equal to";
    case U'≠':
      return U"is not equal to";
    case U'≈':
      return U"approximately";
    case U'√':
      return U"square root of";
    case U'∞':
      return U"infinity";
    case U'∑':
      return U"sigma";
    case U'‰':
      return U"per mille";
    case U'±':
      return U"plus or minus";
    default:
      return std::nullopt;
  }
}

// ---------------------------------------------------------------------------
// The character pass
// ---------------------------------------------------------------------------

/// The reference runtime's ``StringAppender``: a buffer that never holds two
/// spaces in a row or a leading space, so rules can push words freely.
struct Appender {
  std::u32string buffer{};

  bool last_is_space() const {
    return !this->buffer.empty() && is_space(this->buffer.back());
  }
  void push(char32_t c) { this->buffer.push_back(c); }
  void push_str(std::u32string_view s) { this->buffer.append(s); }
  void push_space() {
    if (!this->buffer.empty() && !this->last_is_space()) {
      this->buffer.push_back(U' ');
    }
  }
  void pop_space() {
    while (this->last_is_space()) {
      this->buffer.pop_back();
    }
  }
  void push_spoken(std::u32string_view word) {
    this->push_space();
    this->push_str(word);
    this->push_space();
  }
  /// Whether the buffer ends in a digit, for ``°`` after a number.
  bool ends_with_digit() const {
    return !this->buffer.empty() && is_ascii_digit(this->buffer.back());
  }
  /// Whether the number the buffer ends with is exactly ``1``.
  bool ends_with_one() const {
    const size_t n = this->buffer.size();
    return n > 0 && this->buffer[n - 1] == U'1' &&
           (n == 1 || !is_ascii_digit(this->buffer[n - 2]));
  }
};

std::u32string fold_characters(const std::u32string& in) {
  Appender res;
  std::optional<char32_t> prev;
  for (size_t i = 0; i < in.size(); ++i) {
    const char32_t c = in[i];
    const std::optional<char32_t> next =
        (i + 1 < in.size()) ? std::optional<char32_t>(in[i + 1]) : std::nullopt;
    if (is_double_quote(c)) {
      res.push(U'"');
    } else if (is_single_quote(c)) {
      res.push(U'\'');
    } else if (c == U'‐' || c == U'‑' || c == U'‒' || c == U'―') {
      res.push(U'-');
    } else if (c == U'°' && res.ends_with_digit()) {
      res.push_spoken(res.ends_with_one() ? U"degree" : U"degrees");
      if (next == U'C' || next == U'F') {
        const bool letter_after = (i + 2 < in.size()) && is_alnum(in[i + 2]);
        if (!letter_after) {
          res.push_str(next == U'C' ? U"Celsius" : U"Fahrenheit");
          ++i;
        }
      }
    } else if (const auto folded = unaccented(c)) {
      res.push(*folded);
    } else if (const auto words = symbol_words(c)) {
      res.push_spoken(*words);
    } else if (c == U'~' && next && is_ascii_digit(*next)) {
      res.push_spoken(U"about");
    } else if (c == U'–' || c == U'*' || c == U'—' || c == U'[' || c == U']' ||
               c == U'{' || c == U'}') {
      res.push_space();
    } else if (c == U'•' || c == U'‣' || c == U'◦' || c == U'·' || c == U'→' ||
               c == U'←' || c == U'↑' || c == U'↓' || c == U'➡' || c == U'➜') {
      res.push_space();
    } else if (c == U'…') {
      res.push(U'.');
    } else if (c == U'=') {
      res.push_spoken(U"equals");
    } else if (c == U':' && prev && !is_space(*prev) && next &&
               !is_space(*next)) {
      // "10:30" keeps its colon.
      res.push(U':');
    } else if (c == U';' || c == U':' || c == U'(' || c == U')') {
      res.pop_space();
      res.push(U',');
      res.push_space();
    } else if (is_emoji(c) || is_control(c) || is_space(c)) {
      res.push_space();
    } else {
      res.push(c);
    }
    prev = c;
  }
  res.pop_space();
  return std::move(res.buffer);
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

constexpr std::u32string_view kPunctuation = U"!?.:,;…";

bool is_punctuation(char32_t c) {
  return kPunctuation.find(c) != std::u32string_view::npos;
}

bool all_digits(std::u32string_view s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), is_ascii_digit);
}

/// ``word`` without its trailing punctuation, and that punctuation.
std::pair<std::u32string_view, std::u32string_view> split_suffix(
    std::u32string_view word) {
  size_t end = word.size();
  while (end > 0 && is_punctuation(word[end - 1])) {
    --end;
  }
  return {word.substr(0, end), word.substr(end)};
}

std::u32string u32_from_int(int64_t v) {
  std::u32string out;
  if (v == 0) {
    return U"0";
  }
  while (v > 0) {
    out.insert(out.begin(), static_cast<char32_t>(U'0' + (v % 10)));
    v /= 10;
  }
  return out;
}

/// Digits spelled one by one: a code rather than a quantity.
std::u32string spell_digits(std::u32string_view s) {
  static constexpr std::array<std::u32string_view, 10> kNames = {
      U"zero", U"one", U"two",   U"three", U"four",
      U"five", U"six", U"seven", U"eight", U"nine"};
  std::u32string out;
  for (const char32_t c : s) {
    if (c == U'-') {
      continue;
    }
    if (!out.empty()) {
      out.push_back(U' ');
    }
    if (is_ascii_digit(c)) {
      out.append(kNames[static_cast<size_t>(c - U'0')]);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

/// "10-20" as "10 to 20", for small numbers and years.
std::optional<std::u32string> number_range(std::u32string_view body) {
  const size_t dash = body.find(U'-');
  if (dash == std::u32string_view::npos) {
    return std::nullopt;
  }
  const std::u32string_view from = body.substr(0, dash);
  const std::u32string_view to = body.substr(dash + 1);
  const auto small = [](std::u32string_view s) {
    return s.size() >= 2 && s.size() <= 3 && all_digits(s) && s[0] != U'0';
  };
  const auto year = [](std::u32string_view s) {
    return s.size() == 4 && all_digits(s) &&
           (s[0] == U'1' || (s[0] == U'2' && s[1] == U'0'));
  };
  bool ok = false;
  if (small(from) && small(to)) {
    ok = from.size() < to.size() || (from.size() == to.size() && from < to);
  } else if (from.size() == 4 && to.size() == 4) {
    ok = year(from) && year(to) && from < to;
  } else if (from.size() == 4 && to.size() == 2) {
    ok = year(from) && all_digits(to) && from.substr(2) < to;
  }
  if (!ok) {
    return std::nullopt;
  }
  return std::u32string(from) + U" to " + std::u32string(to);
}

/// What the numbers rule reads ``word`` as, and the trailing punctuation it
/// carries through. English groups thousands with commas and marks decimals
/// with a dot.
std::optional<std::pair<std::u32string, std::u32string_view>> number_words(
    std::u32string_view word) {
  if (word.size() >= 2 && word[0] == U'0' && is_ascii_digit(word[1])) {
    return std::nullopt;
  }
  bool negative = false;
  std::u32string_view rest = word;
  if (!rest.empty() && rest[0] == U'-') {
    negative = true;
    rest.remove_prefix(1);
  }
  size_t end = 0;
  while (end < rest.size() &&
         (is_ascii_digit(rest[end]) || rest[end] == U',')) {
    ++end;
  }
  std::u32string_view integer = rest.substr(0, end);
  while (!integer.empty() && integer.back() == U',') {
    integer.remove_suffix(1);
  }
  if (integer.empty()) {
    return std::nullopt;
  }
  rest.remove_prefix(integer.size());
  std::u32string_view decimal;
  if (!rest.empty() && rest[0] == U'.') {
    size_t d = 1;
    while (d < rest.size() && is_ascii_digit(rest[d])) {
      ++d;
    }
    if (d > 1) {
      decimal = rest.substr(1, d - 1);
      rest.remove_prefix(d);
    }
  }
  if (!std::all_of(rest.begin(), rest.end(), is_punctuation)) {
    return std::nullopt;
  }
  int64_t value = 0;
  for (const char32_t c : integer) {
    if (is_ascii_digit(c)) {
      value = value * 10 + static_cast<int64_t>(c - U'0');
      if (value >= 1000000000000LL) {
        return std::nullopt;
      }
    }
  }
  std::u32string words;
  if (value < 1000 || (value >= 1900 && value < 2100)) {
    words = u32_from_int(value);
  } else {
    static constexpr std::array<std::pair<int64_t, std::u32string_view>, 3>
        kScales = {{{1000000000LL, U"billion"},
                    {1000000LL, U"million"},
                    {1000LL, U"thousand"}}};
    for (const auto& [div, name] : kScales) {
      const int64_t group = (value / div) % 1000;
      if (group == 0) {
        continue;
      }
      if (!words.empty()) {
        words.push_back(U' ');
      }
      words += u32_from_int(group) + U" " + std::u32string(name);
    }
    if (value % 1000 != 0) {
      words += U" " + u32_from_int(value % 1000);
    }
  }
  if (!decimal.empty()) {
    words += U" point " + std::u32string(decimal);
  }
  if (negative) {
    words = U"minus " + words;
  }
  return std::make_pair(std::move(words), rest);
}

/// The numbers rule for one word, or nothing when it does not claim it.
std::optional<std::u32string> rewrite_number(std::u32string_view word) {
  const auto [body, end] = split_suffix(word);
  if (body.size() >= 2 && body[0] == U'0' && all_digits(body)) {
    return spell_digits(body) + std::u32string(end);
  }
  if (const auto range = number_range(body)) {
    return *range + std::u32string(end);
  }
  if (const auto words = number_words(word)) {
    return words->first + std::u32string(words->second);
  }
  // "1500-strong": the number before a hyphenated word.
  const size_t dash = word.find(U'-');
  if (dash != std::u32string_view::npos && dash + 1 < word.size() &&
      is_ascii_alpha(word[dash + 1])) {
    if (const auto words = number_words(word.substr(0, dash));
        words && words->second.empty()) {
      return words->first + std::u32string(word.substr(dash));
    }
  }
  return std::nullopt;
}

/// Three or more groups of digits (a phone number, an account number) are a
/// code and are spelled digit by digit, group by group.
std::optional<std::pair<std::u32string, size_t>> digit_run(
    const std::vector<std::u32string_view>& words, size_t start) {
  const auto group = [](std::u32string_view s) {
    if (s.empty() || s.size() > 8) {
      return false;
    }
    bool any_digit = false;
    for (const char32_t c : s) {
      if (is_ascii_digit(c)) {
        any_digit = true;
      } else if (!(c >= U'A' && c <= U'Z')) {
        return false;
      }
    }
    return any_digit;
  };
  const bool plus = !words[start].empty() && words[start][0] == U'+';
  std::vector<std::u32string_view> bodies;
  std::u32string_view end;
  for (size_t i = start; i < words.size(); ++i) {
    auto [body, suffix] = split_suffix(words[i]);
    if (bodies.empty() && plus) {
      body.remove_prefix(1);
    }
    if (!group(body)) {
      break;
    }
    bodies.push_back(body);
    if (!suffix.empty()) {
      end = suffix;
      break;
    }
  }
  const bool years =
      std::all_of(bodies.begin(), bodies.end(), [](std::u32string_view b) {
        return b.size() == 4 && ((b[0] == U'1' && b[1] == U'9') ||
                                 (b[0] == U'2' && b[1] == U'0'));
      });
  const bool is_long = bodies.size() >= 3 && !years;
  const bool leading_zero =
      std::any_of(bodies.begin(), bodies.end(),
                  [](std::u32string_view b) { return b[0] == U'0'; });
  if (!is_long && !(bodies.size() == 2 && (plus || leading_zero))) {
    return std::nullopt;
  }
  std::u32string spelled;
  for (const std::u32string_view body : bodies) {
    if (!spelled.empty()) {
      spelled += U", ";
    }
    spelled += spell_digits(body);
  }
  if (plus) {
    spelled = U"plus " + spelled;
  }
  spelled += std::u32string(end);
  return std::make_pair(std::move(spelled), bodies.size());
}

/// ``@`` and ``+`` in a word no rule claimed are read out, and the pieces they
/// leave get their own turn at the numbers rule.
std::u32string spell_symbols(std::u32string_view word) {
  if (word.find(U'@') == std::u32string_view::npos &&
      word.find(U'+') == std::u32string_view::npos) {
    return std::u32string(word);
  }
  Appender res;
  for (const char32_t c : word) {
    if (c == U'@') {
      res.push_spoken(U"at");
    } else if (c == U'+') {
      res.push_spoken(U"plus");
    } else {
      res.push(c);
    }
  }
  res.pop_space();
  std::u32string out;
  size_t i = 0;
  const std::u32string& text = res.buffer;
  while (i <= text.size()) {
    size_t j = text.find(U' ', i);
    if (j == std::u32string::npos) {
      j = text.size();
    }
    const std::u32string_view piece(text.data() + i, j - i);
    if (!out.empty()) {
      out.push_back(U' ');
    }
    if (const auto number = rewrite_number(piece)) {
      out += *number;
    } else {
      out += piece;
    }
    i = j + 1;
  }
  return out;
}

std::u32string rewrite_words(const std::u32string& text) {
  std::vector<std::u32string_view> words;
  size_t i = 0;
  while (i <= text.size()) {
    size_t j = text.find(U' ', i);
    if (j == std::u32string::npos) {
      j = text.size();
    }
    words.emplace_back(text.data() + i, j - i);
    i = j + 1;
  }
  std::u32string out;
  for (size_t w = 0; w < words.size();) {
    if (!out.empty()) {
      out.push_back(U' ');
    }
    if (const auto run = digit_run(words, w)) {
      out += run->first;
      w += run->second;
      continue;
    }
    if (const auto number = rewrite_number(words[w])) {
      out += *number;
    } else {
      out += spell_symbols(words[w]);
    }
    ++w;
  }
  return out;
}

/// A closing quote after the final punctuation is dropped: the model reads it
/// as a word.
void strip_closing_quote(std::u32string& text) {
  while (!text.empty() && text.back() == U'"') {
    text.pop_back();
  }
  if (text.size() >= 2 && text.back() == U'\'') {
    const char32_t before = text[text.size() - 2];
    if (before == U'.' || before == U'!' || before == U'?' || before == U',' ||
        before == U';') {
      text.pop_back();
    }
  }
}

std::string to_utf8(const std::u32string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char32_t c : s) {
    utf8_append_codepoint(out, c);
  }
  return out;
}

std::vector<std::string> split_words(std::string_view text) {
  std::vector<std::string> words;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() &&
           is_ascii_whitespace(static_cast<unsigned char>(text[i]))) {
      ++i;
    }
    size_t j = i;
    while (j < text.size() &&
           !is_ascii_whitespace(static_cast<unsigned char>(text[j]))) {
      ++j;
    }
    if (j > i) {
      words.emplace_back(text.substr(i, j - i));
    }
    i = j;
  }
  return words;
}

std::string join_words(const std::vector<std::string>& words, size_t first,
                       size_t last) {
  std::string out;
  for (size_t i = first; i < last; ++i) {
    if (!out.empty()) {
      out.push_back(' ');
    }
    out += words[i];
  }
  return out;
}

bool ends_with_any(std::string_view word, std::string_view marks) {
  // Closing quotes and brackets may follow the mark: `"yes,"`.
  size_t end = word.size();
  while (end > 0) {
    const char c = word[end - 1];
    if (c == '"' || c == '\'' || c == ')' || c == ']') {
      --end;
    } else {
      break;
    }
  }
  const std::string_view trimmed = word.substr(0, end);
  for (const char m : marks) {
    if (!trimmed.empty() && trimmed.back() == m) {
      return true;
    }
  }
  return false;
}

/// Where to cut ``words``: after the sentence end nearest the middle in tokens
/// that leaves ``min_side`` tokens either side, else after such a clause mark,
/// else after the word nearest the middle. Returns the number of words before
/// the cut, at least one and at most ``words.size() - 1``.
size_t cut_point(const std::vector<std::string>& words, int min_side,
                 const std::function<int(std::string_view)>& count_tokens) {
  std::vector<int> ends;
  int total = 0;
  for (const std::string& word : words) {
    total += count_tokens(word);
    ends.push_back(total);
  }
  const auto imbalance = [&](size_t i) {
    return std::abs(total - 2 * ends[i]);
  };
  const auto after = [&](std::string_view marks) -> std::optional<size_t> {
    std::optional<size_t> best;
    for (size_t i = 0; i + 1 < words.size(); ++i) {
      if (!ends_with_any(words[i], marks) || ends[i] < min_side ||
          total - ends[i] < min_side) {
        continue;
      }
      if (!best || imbalance(i) < imbalance(*best)) {
        best = i;
      }
    }
    return best;
  };
  std::optional<size_t> i = after(".!?");
  if (!i) {
    i = after(",;:");
  }
  if (!i) {
    size_t best = 0;
    for (size_t k = 1; k + 1 < words.size(); ++k) {
      if (imbalance(k) < imbalance(best)) {
        best = k;
      }
    }
    i = best;
  }
  return *i + 1;
}

}  // namespace

std::string phonon_normalize_text(std::string_view text) {
  const std::u32string in = utf8_str_to_u32(std::string(text));
  std::u32string folded = fold_characters(in);
  std::u32string rewritten = rewrite_words(folded);
  strip_closing_quote(rewritten);
  return to_utf8(rewritten);
}

PhononPrompt phonon_prepare_prompt(std::string_view text) {
  PhononPrompt out;
  const std::vector<std::string> words = split_words(text);
  out.words = static_cast<int>(words.size());
  if (words.empty()) {
    return out;
  }
  std::u32string joined = utf8_str_to_u32(join_words(words, 0, words.size()));
  if (joined[0] >= U'a' && joined[0] <= U'z') {
    joined[0] = static_cast<char32_t>(joined[0] - U'a' + U'A');
  } else if (joined[0] >= 0xE0 && joined[0] <= 0xFE && joined[0] != 0xF7) {
    // Latin-1 lower-case letters sit 0x20 above their capitals.
    joined[0] = static_cast<char32_t>(joined[0] - 0x20);
  }
  if (is_alnum(joined.back())) {
    joined.push_back(U'.');
  }
  out.text = to_utf8(joined);
  return out;
}

std::vector<std::string> phonon_plan_chunks(
    std::string_view text,
    const std::function<int(std::string_view)>& count_tokens, int max_tokens,
    int hard_max_tokens) {
  SentenceSplitOptions split_options;
  split_options.language = "en";
  split_options.split_on_colon = false;
  std::vector<std::string> sentences;
  for (const std::string& unit : split_sentences(text, split_options)) {
    const std::string trimmed = trim_ascii_ws_copy(unit);
    if (!trimmed.empty()) {
      sentences.push_back(trimmed);
    }
  }

  // Greedily group whole sentences up to max_tokens.
  std::vector<std::string> chunks;
  std::string current;
  int current_tokens = 0;
  for (const std::string& sentence : sentences) {
    const int n = count_tokens(phonon_prepare_prompt(sentence).text);
    if (current.empty()) {
      current = sentence;
      current_tokens = n;
    } else if (current_tokens + n > max_tokens) {
      chunks.push_back(current);
      current = sentence;
      current_tokens = n;
    } else {
      current += ' ';
      current += sentence;
      current_tokens += n;
    }
  }
  if (!current.empty()) {
    chunks.push_back(current);
  }

  // Cut anything over the hard limit, keeping order.
  std::vector<std::string> fitted;
  std::vector<std::string> todo(chunks.rbegin(), chunks.rend());
  const int min_side = (hard_max_tokens + 3) / 4;
  while (!todo.empty()) {
    const std::string chunk = std::move(todo.back());
    todo.pop_back();
    if (count_tokens(phonon_prepare_prompt(chunk).text) <= hard_max_tokens) {
      fitted.push_back(chunk);
      continue;
    }
    const std::vector<std::string> words = split_words(chunk);
    if (words.size() < 2) {
      // One word: cut it between its characters.
      const std::u32string word = utf8_str_to_u32(chunk);
      const size_t mid = word.size() / 2;
      if (mid == 0) {
        fitted.push_back(chunk);
        continue;
      }
      todo.push_back(to_utf8(word.substr(mid)));
      todo.push_back(to_utf8(word.substr(0, mid)));
      continue;
    }
    const size_t at = cut_point(words, min_side, count_tokens);
    todo.push_back(join_words(words, at, words.size()));
    todo.push_back(join_words(words, 0, at));
  }
  return fitted;
}

}  // namespace moonshine_tts
