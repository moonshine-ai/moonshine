#include "context-biaser.h"

#include <algorithm>
#include <cmath>

#include "string-utils.h"

namespace {

// The word-boundary marker the Moonshine tokenizers use (U+2581). A term that
// appears mid-sentence starts with a token carrying this prefix, so it does not
// share a first token with the same term at the very start of an utterance.
const char *kWordStartMarker = "\xe2\x96\x81";

}  // namespace

void ContextBiaser::add_token_sequence(const std::vector<int32_t> &tokens) {
  if (tokens.empty()) {
    return;
  }
  int32_t node_index = 0;
  for (const int32_t token : tokens) {
    const auto existing = this->nodes.at(node_index).children.find(token);
    if (existing != this->nodes.at(node_index).children.end()) {
      node_index = existing->second;
      continue;
    }
    // Read the parent depth and record the edge before growing the vector,
    // since push_back invalidates any reference into it.
    const int child_depth = this->nodes.at(node_index).depth + 1;
    const int32_t child_index = static_cast<int32_t>(this->nodes.size());
    this->nodes.at(node_index).children.emplace(token, child_index);
    this->nodes.push_back(Node{});
    this->nodes.at(child_index).depth = child_depth;
    this->max_depth = std::max(this->max_depth, child_depth);
    node_index = child_index;
  }
  this->sequence_count++;
}

std::vector<std::string> ContextBiaser::variants_for_term(
    const std::string &term) {
  const std::string trimmed = trim(term);
  if (trimmed.empty()) {
    return {};
  }
  // Already explicitly anchored to a word start by the caller.
  if (starts_with(trimmed, kWordStartMarker)) {
    return {trimmed};
  }
  return {trimmed, " " + trimmed};
}

void ContextBiaser::clear() {
  this->nodes.assign(1, Node{});
  this->sequence_count = 0;
  this->max_depth = 0;
  this->depth_bonuses.clear();
  this->reset();
}

void ContextBiaser::reset() { this->active.assign(1, 0); }

float ContextBiaser::bonus_for_depth(int depth) const {
  if (depth <= 0) {
    return 0.0f;
  }
  return this->boost * (1.0f + std::log(static_cast<float>(depth)));
}

void ContextBiaser::ensure_depth_bonuses() {
  // One slot past the deepest node, so apply() can read the bonus for a leaf's
  // children (there are none, and the value goes unused) without a bounds test
  // inside its loop.
  const size_t wanted = static_cast<size_t>(this->max_depth) + 2;
  if (this->depth_bonuses.size() == wanted &&
      this->depth_bonuses_boost == this->boost) {
    return;
  }
  this->depth_bonuses.resize(wanted);
  for (size_t depth = 0; depth < wanted; ++depth) {
    this->depth_bonuses.at(depth) =
        this->bonus_for_depth(static_cast<int>(depth));
  }
  this->depth_bonuses_boost = this->boost;
}

void ContextBiaser::apply(float *logits, int vocab_size) {
  if (logits == nullptr || this->sequence_count == 0) {
    return;
  }
  this->ensure_depth_bonuses();
  this->pending_bonuses.clear();
  for (const int32_t node_index : this->active) {
    // Overlapping key terms can propose the same next token from different
    // depths, and we keep the largest bonus rather than stacking them so a
    // token shared by many terms is not boosted out of all proportion. Only
    // tokens from the second active node onwards can collide, though: a single
    // node lists each token once. Skipping the scan for the first node keeps
    // this loop linear in the number of candidates, which matters because the
    // root alone offers one candidate per key term on every decoded token.
    const bool needs_dedup = !this->pending_bonuses.empty();
    const Node &node = this->nodes.at(node_index);
    // Every child of a node sits one level below it, so the bonus is the same
    // for all of them and there is no need to visit the child nodes at all.
    const float bonus = this->depth_bonuses.at(node.depth + 1);
    for (const auto &child : node.children) {
      const int32_t token = child.first;
      if (token < 0 || token >= vocab_size) {
        continue;
      }
      bool already_pending = false;
      if (needs_dedup) {
        for (auto &pending : this->pending_bonuses) {
          if (pending.first == token) {
            pending.second = std::max(pending.second, bonus);
            already_pending = true;
            break;
          }
        }
      }
      if (!already_pending) {
        this->pending_bonuses.emplace_back(token, bonus);
      }
    }
  }
  // Unchecked indexing is deliberate here: this runs on every decoded token,
  // and every entry was range-checked against vocab_size above.
  for (const auto &[token, bonus] : this->pending_bonuses) {
    logits[token] += bonus;
  }
}

void ContextBiaser::advance(int32_t token) {
  if (this->sequence_count == 0) {
    return;
  }
  this->next_active.clear();
  // The root stays active so a key term can begin at the next token even in
  // the middle of matching another one.
  this->next_active.push_back(0);
  for (const int32_t node_index : this->active) {
    const auto child = this->nodes.at(node_index).children.find(token);
    if (child != this->nodes.at(node_index).children.end()) {
      this->next_active.push_back(child->second);
    }
  }
  this->active.swap(this->next_active);
}

float ContextBiaser::bonus_for_token(int32_t token) const {
  float best = 0.0f;
  for (const int32_t node_index : this->active) {
    const auto child = this->nodes.at(node_index).children.find(token);
    if (child != this->nodes.at(node_index).children.end()) {
      best =
          std::max(best, bonus_for_depth(this->nodes.at(child->second).depth));
    }
  }
  return best;
}
