#ifndef CONTEXT_BIASER_H
#define CONTEXT_BIASER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Decode-time contextual biasing towards a caller-supplied list of key terms
// (jargon, product names, proper nouns) with no retraining.
//
// Each key term is tokenized once and stored in a prefix trie over subword
// token IDs. While decoding we track which trie nodes the emitted prefix has
// reached, and just before the argmax we add a bonus to the logits of the
// tokens that would continue one of those paths. A term that spans several
// subwords, like "Kubernetes" (``|Ku`` + ``bern`` + ``etes``), is therefore
// boosted piece by piece instead of having to win the whole way in one step.
//
// The bonus grows with depth, as ``boost * (1 + ln(depth))``. Starting down a
// path stays cheap while completing one is strongly rewarded. That ramp matters
// because greedy decoding cannot take back a token it has already emitted: with
// a flat bonus, a wrong first subword would be as attractive as a genuine
// completion, so short prefixes of key terms would fire on unrelated audio.
//
// Not thread-safe: the walk state is mutable and every call advances it.
// Decoding is already serialized per transcriber, so one biaser per transcriber
// is enough.
class ContextBiaser {
 public:
  // Added to the logit of a depth-1 token; deeper tokens are scaled up from
  // here. Moonshine logits are pre-softmax, and a value in this range shifts
  // close calls towards a key term without overriding confident predictions.
  //
  // Chosen by sweeping it on a biasing test set built from LibriSpeech
  // test-clean, where each utterance's rare words are its key terms and the
  // list is padded to a realistic size with rare words from elsewhere in the
  // corpus (see scripts/make-keyterm-testset.py). Tiny, Small and Medium
  // Streaming all put the best error rate on the key terms right here, and all
  // three get worse on both the terms and the surrounding words above it, so a
  // higher value is not a stronger version of this feature but a broken one.
  static constexpr float kDefaultBoost = 2.0f;

  // Registers one tokenization of a key term. Callers normally add more than
  // one tokenization per term (with and without a leading space) so that it
  // matches both mid-sentence and utterance-initial; see
  // ContextBiaser::variants_for_term.
  void add_token_sequence(const std::vector<int32_t> &tokens);

  // Returns the spellings that should be tokenized and registered for one
  // key term. The tokenizer marks word starts with U+2581, so the
  // mid-sentence and utterance-initial forms of a term have different first
  // tokens and both need to be in the trie.
  static std::vector<std::string> variants_for_term(const std::string &term);

  void set_boost(float boost) { this->boost = boost; }
  float get_boost() const { return this->boost; }

  bool empty() const { return this->sequence_count == 0; }
  size_t sequence_count_for_test() const { return this->sequence_count; }

  void clear();

  // Discards any partial match and returns the walk to the root. Call this
  // before each decode pass, which always restarts from BOS.
  void reset();

  // Adds the bonus for every token that continues an active path. ``logits``
  // is modified in place and must hold at least ``vocab_size`` floats.
  void apply(float *logits, int vocab_size);

  // Advances the walk over a token that has actually been emitted.
  void advance(int32_t token);

  // The bonus that apply() would currently add to ``token``. Exposed for
  // tests, which need to check the depth ramp without a decoder.
  float bonus_for_token(int32_t token) const;

 private:
  struct Node {
    std::unordered_map<int32_t, int32_t> children;
    int depth = 0;
  };

  float bonus_for_depth(int depth) const;

  // Refills depth_bonuses if the trie has grown deeper or the boost changed.
  // apply() runs on every decoded token, so the logarithm is worth hoisting out
  // of it: depths are bounded by the longest key term, a handful of subwords.
  void ensure_depth_bonuses();

  // nodes[0] is the root, which is always active so that a key term can start
  // at any point in the transcript.
  std::vector<Node> nodes{Node{}};
  std::vector<int32_t> active{0};
  // Scratch buffers reused across steps to keep the decode loop allocation
  // free. They hold no state between calls.
  std::vector<int32_t> next_active;
  std::vector<std::pair<int32_t, float>> pending_bonuses;
  // bonus_for_depth() indexed by depth, valid up to max_depth for
  // depth_bonuses_boost.
  std::vector<float> depth_bonuses;
  float depth_bonuses_boost = 0.0f;
  int max_depth = 0;
  size_t sequence_count = 0;
  float boost = kDefaultBoost;
};

#endif
