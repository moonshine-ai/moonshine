#include "context-extractor.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

#include "string-utils.h"

namespace {

// Punctuation that behaves like its ASCII counterpart but arrives as multi-byte
// UTF-8 in real prose, which is where context passages come from. Folding it
// first lets the splitter below stay a byte scan: untouched, a curly apostrophe
// would cut "Tellson's" into two candidates and an em dash would glue two words
// into one.
struct PunctuationMapping {
  const char *utf8;
  const char *replacement;
};

const PunctuationMapping kPunctuationMappings[] = {
    {"\xe2\x80\x99", "'"},  // U+2019 right single quotation mark
    {"\xe2\x80\x98", "'"},  // U+2018 left single quotation mark
    {"\xe2\x80\x9c", " "},  // U+201C left double quotation mark
    {"\xe2\x80\x9d", " "},  // U+201D right double quotation mark
    {"\xe2\x80\x93", " "},  // U+2013 en dash
    {"\xe2\x80\x94", " "},  // U+2014 em dash
    {"\xe2\x80\xa6", " "},  // U+2026 horizontal ellipsis
    {"\xc2\xa0", " "},      // U+00A0 no-break space
};

// Everything outside ASCII is taken to be part of a word. The mappings above
// have already pulled out the non-ASCII punctuation that turns up in prose, and
// treating the remainder as letters is what makes accented Latin, Cyrillic and
// CJK work here without carrying a Unicode character table.
bool is_letter_byte(unsigned char byte) {
  return byte >= 0x80 || std::isalpha(byte) != 0;
}

bool is_digit_byte(unsigned char byte) { return std::isdigit(byte) != 0; }

// Only ever joins two halves of a word, never starts one.
bool is_joiner_byte(unsigned char byte) { return byte == '\'' || byte == '-'; }

size_t character_count(const std::string &word) {
  size_t count = 0;
  for (const char c : word) {
    // Continuation bytes carry on the character before them rather than
    // starting one of their own.
    if ((static_cast<unsigned char>(c) & 0xc0) != 0x80) {
      count++;
    }
  }
  return count;
}

bool contains_digit(const std::string &word) {
  for (const char c : word) {
    if (is_digit_byte(static_cast<unsigned char>(c))) {
      return true;
    }
  }
  return false;
}

std::string trim_joiners(const std::string &word) {
  size_t begin = 0;
  size_t end = word.size();
  while (begin < end &&
         is_joiner_byte(static_cast<unsigned char>(word[begin]))) {
    begin++;
  }
  while (end > begin &&
         is_joiner_byte(static_cast<unsigned char>(word[end - 1]))) {
    end--;
  }
  return word.substr(begin, end - begin);
}

// Case folding for grouping only, never for output. ASCII-only on purpose: the
// bytes above 0x7f are a UTF-8 encoding rather than characters, so lowercasing
// them one byte at a time would corrupt them. The cost is that two spellings of
// an accented word differing only in case stay separate candidates, which
// wastes a slot at worst.
std::string ascii_fold(const std::string &word) {
  std::string folded = word;
  for (char &c : folded) {
    const unsigned char byte = static_cast<unsigned char>(c);
    if (byte < 0x80) {
      c = static_cast<char>(std::tolower(byte));
    }
  }
  return folded;
}

// One distinct spelling and where it was first seen.
struct SurfaceForm {
  size_t count = 0;
  size_t first_index = 0;
};

// A word and its case variants, which are one term as far as biasing goes.
struct Candidate {
  std::string term;
  // Occurrences of ``term`` alone, kept only to judge a rival spelling against
  // it, as against ``occurrences`` which counts every spelling in the group.
  size_t term_count = 0;
  size_t occurrences = 0;
  size_t subwords = 0;
  size_t first_index = 0;
};

}  // namespace

std::string ContextExtractor::strip_possessive(const std::string &word) {
  // "Tellson's" and, for a name already ending in s, "Jones'".
  if (ends_with(word, "'s") || ends_with(word, "'S")) {
    return word.substr(0, word.size() - 2);
  }
  if (ends_with(word, "'")) {
    return word.substr(0, word.size() - 1);
  }
  return word;
}

std::vector<std::string> ContextExtractor::candidate_words(
    const std::string &text) {
  std::string normalized = text;
  for (const PunctuationMapping &mapping : kPunctuationMappings) {
    normalized = replace_all(normalized, mapping.utf8, mapping.replacement);
  }

  std::vector<std::string> words;
  std::string current;
  // Digits are collected as part of a word but disqualify it below, so that
  // "IPv6" is read as one word and then dropped rather than being handed on as
  // a truncated "IPv".
  auto flush = [&words, &current]() {
    if (current.empty()) {
      return;
    }
    const std::string word =
        trim_joiners(ContextExtractor::strip_possessive(trim_joiners(current)));
    current.clear();
    if (character_count(word) < ContextExtractor::kMinCharacters) {
      return;
    }
    if (contains_digit(word)) {
      return;
    }
    words.push_back(word);
  };

  for (const char c : normalized) {
    const unsigned char byte = static_cast<unsigned char>(c);
    if (is_letter_byte(byte) || is_digit_byte(byte)) {
      current.push_back(c);
      continue;
    }
    // A joiner only stays inside a word that has already started, which keeps
    // a hyphen used as a dash from opening one.
    if (is_joiner_byte(byte) && !current.empty()) {
      current.push_back(c);
      continue;
    }
    flush();
  }
  flush();
  return words;
}

std::vector<std::string> ContextExtractor::extract(
    const std::string &context, int32_t max_terms,
    const SubwordCountFn &subword_count) {
  const size_t limit = static_cast<size_t>(
      max_terms > 0 ? max_terms : ContextExtractor::kDefaultMaxTerms);
  if (!subword_count) {
    return {};
  }

  // Count each spelling, then group the spellings that differ only in case.
  // Both steps are needed: the group decides how much the passage leans on a
  // word, while the spellings decide which capitalization to ask for, and a
  // passage that says "Madame" thirty times and "madame" twice wants one term
  // spelled the first way rather than two terms competing.
  std::unordered_map<std::string, SurfaceForm> surface_forms;
  const std::vector<std::string> words =
      ContextExtractor::candidate_words(context);
  for (size_t index = 0; index < words.size(); index++) {
    auto inserted =
        surface_forms.emplace(words.at(index), SurfaceForm{0, index});
    inserted.first->second.count++;
  }

  std::unordered_map<std::string, Candidate> grouped;
  for (const auto &[form, stats] : surface_forms) {
    auto inserted = grouped.emplace(ascii_fold(form), Candidate{});
    Candidate &candidate = inserted.first->second;
    candidate.occurrences += stats.count;
    // The spelling the passage uses most often wins, and an earlier first
    // appearance breaks a tie so that the result does not depend on the hash
    // order of the map above.
    const bool is_better = inserted.second ||
                           stats.count > candidate.term_count ||
                           (stats.count == candidate.term_count &&
                            stats.first_index < candidate.first_index);
    if (is_better) {
      candidate.term = form;
      candidate.term_count = stats.count;
      candidate.first_index = stats.first_index;
    }
  }

  std::vector<Candidate> candidates;
  candidates.reserve(grouped.size());
  for (auto &[folded, candidate] : grouped) {
    (void)folded;
    // The mid-sentence spelling, which is the form the tokenizer's
    // word-boundary marker makes the common case in running text.
    candidate.subwords = subword_count(" " + candidate.term);
    if (candidate.subwords < ContextExtractor::kMinSubwordTokens) {
      continue;
    }
    candidates.push_back(candidate);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &left, const Candidate &right) {
              if (left.occurrences != right.occurrences) {
                return left.occurrences > right.occurrences;
              }
              // Among words the passage says equally often, the one the
              // tokenizer finds strangest is the one biasing can help most.
              // This is what orders a short passage, where almost everything
              // is said exactly once.
              if (left.subwords != right.subwords) {
                return left.subwords > right.subwords;
              }
              return left.first_index < right.first_index;
            });

  std::vector<std::string> terms;
  terms.reserve(std::min(limit, candidates.size()));
  for (const Candidate &candidate : candidates) {
    if (terms.size() >= limit) {
      break;
    }
    terms.push_back(candidate.term);
  }
  return terms;
}
