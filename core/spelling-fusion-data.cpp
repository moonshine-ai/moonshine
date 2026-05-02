#include "spelling-fusion-data.h"

#include <algorithm>

#include "spelling-fusion.h"

namespace spelling_fusion_data {

namespace {

// Helper: build a normalized lookup table from a list of {spoken, char}
// pairs. Uses ``spelling_normalize`` so apostrophe-containing source
// keys (like ``"that's it"``) become the same key the matcher sees at
// runtime (``"thats it"``).
std::unordered_map<std::string, std::string> build_lookup(
    std::initializer_list<std::pair<const char *, const char *>> pairs) {
  std::unordered_map<std::string, std::string> result;
  result.reserve(pairs.size());
  for (const auto &pair : pairs) {
    std::string key = spelling_normalize(pair.first);
    if (!key.empty()) {
      result[key] = pair.second;
    }
  }
  return result;
}

std::unordered_set<std::string> build_set(
    std::initializer_list<const char *> phrases) {
  std::unordered_set<std::string> result;
  result.reserve(phrases.size());
  for (const char *phrase : phrases) {
    std::string key = spelling_normalize(phrase);
    if (!key.empty()) {
      result.insert(std::move(key));
    }
  }
  return result;
}

}  // namespace

const std::unordered_map<std::string, std::string> &lookup_table() {
  // Combined letter + digit + special-char lookup. The order here
  // doesn't matter (it's a hash map at the end), but we keep the same
  // grouping as the Python source so a side-by-side diff stays
  // readable.
  static const std::unordered_map<std::string, std::string> table =
      build_lookup({
          // ---- Letters: plain spellings + most common STT variants ----
          {"a", "a"}, {"ay", "a"}, {"hey", "a"}, {"aye", "a"},
          {"b", "b"}, {"bee", "b"},
          {"c", "c"}, {"see", "c"}, {"sea", "c"},
          {"d", "d"}, {"dee", "d"},
          {"e", "e"},
          {"f", "f"}, {"ef", "f"}, {"eff", "f"},
          {"g", "g"}, {"gee", "g"},
          {"h", "h"}, {"aitch", "h"},
          {"i", "i"}, {"eye", "i"},
          {"j", "j"}, {"jay", "j"},
          {"k", "k"}, {"kay", "k"}, {"okay", "k"}, {"ok", "k"},
          {"l", "l"}, {"el", "l"}, {"ell", "l"},
          {"m", "m"}, {"em", "m"},
          {"n", "n"}, {"en", "n"}, {"and", "n"},
          {"o", "o"}, {"oh", "o"},
          {"p", "p"}, {"pee", "p"},
          {"q", "q"}, {"queue", "q"}, {"cue", "q"},
          {"r", "r"}, {"are", "r"}, {"ar", "r"}, {"ah", "r"},
          {"uh-huh", "r"}, {"aww", "r"}, {"awe", "r"},
          {"s", "s"}, {"es", "s"}, {"ess", "s"},
          {"t", "t"}, {"tee", "t"},
          {"u", "u"}, {"you", "u"},
          {"v", "v"}, {"vee", "v"},
          {"w", "w"}, {"double u", "w"}, {"double you", "w"},
          {"x", "x"}, {"ex", "x"},
          {"y", "y"}, {"why", "y"}, {"wye", "y"},
          {"z", "z"}, {"zee", "z"}, {"zed", "z"}, {"zet", "z"},
          // ---- NATO / ICAO phonetic alphabet ----
          {"alpha", "a"}, {"alfa", "a"},
          {"bravo", "b"},
          {"charlie", "c"},
          {"delta", "d"},
          {"echo", "e"},
          {"foxtrot", "f"}, {"fox trot", "f"},
          {"golf", "g"},
          {"hotel", "h"},
          {"india", "i"},
          {"juliet", "j"}, {"juliett", "j"},
          {"kilo", "k"},
          {"lima", "l"},
          {"mike", "m"},
          {"november", "n"},
          {"oscar", "o"},
          {"papa", "p"},
          {"quebec", "q"},
          {"romeo", "r"},
          {"sierra", "s"},
          {"tango", "t"},
          {"uniform", "u"},
          {"victor", "v"},
          {"whiskey", "w"}, {"whisky", "w"},
          {"x-ray", "x"}, {"xray", "x"}, {"x ray", "x"},
          {"yankee", "y"},
          {"zulu", "z"},
          // ---- Digits + spoken-form homophones ----
          {"zero", "0"}, {"0", "0"},
          {"one", "1"}, {"won", "1"}, {"1", "1"},
          {"two", "2"}, {"to", "2"}, {"too", "2"}, {"2", "2"},
          {"three", "3"}, {"3", "3"},
          {"four", "4"}, {"for", "4"}, {"4", "4"},
          {"five", "5"}, {"5", "5"},
          {"six", "6"}, {"6", "6"},
          {"seven", "7"}, {"7", "7"},
          {"eight", "8"}, {"ate", "8"}, {"8", "8"},
          {"nine", "9"}, {"niner", "9"}, {"9", "9"},
          // ---- Punctuation ----
          {"period", "."}, {"dot", "."}, {"full stop", "."}, {"point", "."},
          {"comma", ","},
          {"colon", ":"},
          {"semicolon", ";"}, {"semi colon", ";"},
          {"exclamation mark", "!"}, {"exclamation point", "!"},
          {"exclamation", "!"}, {"bang", "!"},
          {"question mark", "?"},
          // ---- Brackets / parens ----
          {"open parenthesis", "("}, {"left parenthesis", "("},
          {"open paren", "("}, {"left paren", "("},
          {"close parenthesis", ")"}, {"right parenthesis", ")"},
          {"close paren", ")"}, {"right paren", ")"},
          {"open bracket", "["}, {"left bracket", "["},
          {"close bracket", "]"}, {"right bracket", "]"},
          {"open brace", "{"}, {"left brace", "{"},
          {"open curly", "{"}, {"left curly", "{"},
          {"close brace", "}"}, {"right brace", "}"},
          {"close curly", "}"}, {"right curly", "}"},
          // ---- Common password / code characters ----
          {"at sign", "@"}, {"at", "@"}, {"at symbol", "@"},
          {"hash", "#"}, {"hashtag", "#"}, {"pound sign", "#"},
          {"number sign", "#"}, {"pound", "#"},
          {"dollar sign", "$"}, {"dollar", "$"},
          {"percent", "%"}, {"percent sign", "%"}, {"per cent", "%"},
          {"caret", "^"}, {"carrot", "^"}, {"hat", "^"},
          {"ampersand", "&"}, {"and sign", "&"},
          {"asterisk", "*"}, {"star", "*"},
          {"hyphen", "-"}, {"dash", "-"}, {"minus", "-"},
          {"underscore", "_"}, {"under score", "_"},
          {"plus", "+"}, {"plus sign", "+"},
          {"equals", "="}, {"equal sign", "="}, {"equals sign", "="},
          {"pipe", "|"}, {"vertical bar", "|"},
          {"backslash", "\\"}, {"back slash", "\\"},
          {"forward slash", "/"}, {"slash", "/"},
          {"tilde", "~"},
          {"grave", "`"}, {"backtick", "`"}, {"back tick", "`"},
          {"apostrophe", "'"}, {"single quote", "'"},
          {"quote", "\""}, {"double quote", "\""}, {"quotation mark", "\""},
          // ---- Whitespace / control ----
          {"space", " "},
      });
  return table;
}

const std::unordered_set<std::string> &upper_modifiers() {
  static const std::unordered_set<std::string> set = build_set({
      "upper case", "uppercase", "upper", "capital", "cap", "big", "shift",
  });
  return set;
}

const std::vector<std::string> &upper_modifiers_by_length() {
  // Sorted descending by length so longest-prefix-first matching works
  // ("upper case" wins over "upper" when both could fire).
  static const std::vector<std::string> by_len = []() {
    const auto &set = upper_modifiers();
    std::vector<std::string> v(set.begin(), set.end());
    std::sort(v.begin(), v.end(), [](const std::string &a, const std::string &b) {
      return a.size() > b.size();
    });
    return v;
  }();
  return by_len;
}

const std::unordered_set<std::string> &undo_words() {
  static const std::unordered_set<std::string> set = build_set({
      "undo", "delete", "backspace", "back space", "erase",
      "scratch that", "remove",
  });
  return set;
}

const std::unordered_set<std::string> &clear_words() {
  static const std::unordered_set<std::string> set = build_set({
      "clear", "clear all", "reset", "start over",
  });
  return set;
}

const std::unordered_set<std::string> &stop_words() {
  static const std::unordered_set<std::string> set = build_set({
      "stop", "end", "finish", "finished", "done",
      "complete", "that's it", "submit", "confirm",
      "i'm done", "all done", "go", "enter",
  });
  return set;
}

const std::unordered_set<std::string> &default_weak_homonyms() {
  static const std::unordered_set<std::string> set = build_set({
      "okay", "ok", "you",
  });
  return set;
}

const std::unordered_map<std::string, std::string> &spell_class_to_char() {
  static const std::unordered_map<std::string, std::string> table = {
      {"zero", "0"},  {"one", "1"},  {"two", "2"},   {"three", "3"},
      {"four", "4"},  {"five", "5"}, {"six", "6"},   {"seven", "7"},
      {"eight", "8"}, {"nine", "9"},
  };
  return table;
}

const DefaultSpellingMeta &default_meta() {
  // Class list mirrored from
  // https://download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json
  // (kept as a hard-coded vector rather than parsing JSON at startup; the
  // model is small and the labels are stable).
  static const std::vector<std::string> classes = {
      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
      "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
      "u", "v", "w", "x", "y", "z",
      "zero", "one", "two", "three", "four",
      "five", "six", "seven", "eight", "nine",
  };
  static const DefaultSpellingMeta meta = {
      /*sample_rate=*/16000,
      /*clip_seconds=*/1.0f,
      /*input_name=*/"waveform",
      /*output_name=*/"logits",
      /*classes=*/classes,
  };
  return meta;
}

}  // namespace spelling_fusion_data
