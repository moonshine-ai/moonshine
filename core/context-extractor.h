#ifndef CONTEXT_EXTRACTOR_H
#define CONTEXT_EXTRACTOR_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Picks key terms out of a free-form passage of text, so a caller can hand over
// whatever context it already has — the document on screen, a meeting agenda,
// the names in a thread — instead of curating a list by hand.
//
// The selection rests on a property of the model's own tokenizer: its
// vocabulary is frequency-ordered, so a word that is common in the language the
// tokenizer was built for has a token to itself, while an unusual one has to be
// spelled out of several subwords. Needing two or more subwords is the model
// telling us it has no single symbol for this word, which is exactly the case
// where biasing earns its keep. That keeps the whole judgment inside data we
// already load, follows whichever language the loaded tokenizer covers, and
// needs no word lists shipped alongside the models.
//
// Words that pass are ranked by how often the passage says them and the list is
// capped, because length costs accuracy on everything else (see
// docs/models/domain-customization.md): a short list of the terms a passage
// leans on beats every unusual word it happens to contain.
class ContextExtractor {
 public:
  // Cap applied when the caller does not ask for one. The list-length
  // measurements in docs/models/domain-customization.md put a hundred terms at
  // about a third of a point of word error rate on the words you did not list,
  // growing gently either side of that, so this sits where a generous list is
  // still close to free.
  static constexpr int32_t kDefaultMaxTerms = 200;

  // How many subword tokens a word needs before it counts as unusual enough to
  // bias towards. Two is deliberately the lowest bar that means anything: at
  // three, ordinary words are excluded more cleanly but genuine names start
  // going with them, and the cap above is a better tool for precision than a
  // stricter threshold here.
  static constexpr size_t kMinSubwordTokens = 2;

  // Words shorter than this are skipped. Short words are almost all function
  // words, and biasing towards one is the cheapest way to damage a transcript,
  // since a two-letter fragment can fire almost anywhere.
  static constexpr size_t kMinCharacters = 3;

  // Returns how many subword tokens the loaded tokenizer needs to spell
  // ``word``, or 0 if it cannot spell it at all. Injected rather than reached
  // for through the model so that the ranking can be tested against a stub
  // vocabulary.
  using SubwordCountFn = std::function<size_t(const std::string &word)>;

  // Returns the chosen terms, most important first, at most ``max_terms`` of
  // them. Zero or negative asks for kDefaultMaxTerms. The result is
  // deterministic for a given passage and tokenizer.
  static std::vector<std::string> extract(const std::string &context,
                                          int32_t max_terms,
                                          const SubwordCountFn &subword_count);

  // Splits ``text`` into the words worth considering, in the order they appear
  // and with duplicates kept. Case is preserved, since a key term has to carry
  // the spelling the caller wants to see in the transcript. Words containing a
  // digit are left out: a passage has far more dates and quantities in it than
  // it has names like "IPv6", and alphanumerics are the spelling model's
  // problem rather than something biasing is good at. Callers who want one can
  // still name it outright alongside the passage. Exposed for tests.
  static std::vector<std::string> candidate_words(const std::string &text);

  // Removes a trailing possessive, so that "Tellson's" is considered as
  // "Tellson". Without this the two spellings compete as separate candidates
  // and the possessive usually wins the rarity test on its extra subword,
  // spending a slot on a form that only matches half the time it is said.
  // Exposed for tests.
  static std::string strip_possessive(const std::string &word);
};

#endif
