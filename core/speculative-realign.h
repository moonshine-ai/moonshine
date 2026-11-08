#ifndef MOONSHINE_SPECULATIVE_REALIGN_H
#define MOONSHINE_SPECULATIVE_REALIGN_H

// The control loop of speculative decoding with draft realignment, kept free
// of ONNX Runtime so it can be tested against a scripted decoder.
//
// The draft is last update's token sequence. Verification feeds it through
// the decoder in one call and keeps the prefix the model agrees with. When the
// model rejects a token it is nearly always one word that was respelled,
// inserted or dropped, with the rest of the line unchanged, so instead of
// regenerating the rest one token per call this loop works out where the
// model's new token sits in the old draft and verifies the remainder in one
// more call. Every attempt costs one decoder call over the remaining draft,
// so the number of attempts per update is capped.
//
// The output is unchanged by realignment: every emitted token is still the
// model's argmax given the tokens before it.

#include <algorithm>
#include <cstdint>
#include <functional>

struct SpeculativeDecodeHooks {
  // Run the decoder on ``head`` followed by ``cand[0, cand_len)``, where
  // ``head`` is already accepted (or BOS). Accept the longest prefix of the
  // candidate the model agrees with (calling ``accept`` for each token in
  // order), leave the decoder state as if only head + accepted tokens had
  // been fed, store the model's token for the first disagreeing position (or
  // the position after the whole candidate) in ``*next``, and return the
  // number of candidate tokens accepted. Return -1 on error. With cand_len 0
  // this is an ordinary single-token step.
  std::function<int(int head, const int *cand, int cand_len, int *next)> verify;
  // Emit a token the model chose. Return false when decoding is finished
  // (EOS or the token budget), in which case the token is not emitted.
  std::function<bool(int token)> accept;
};

struct SpeculativeDecodeOutcome {
  bool ok = true;
  int realign_attempts = 0;
};

// How far ahead in the remaining draft to look for the model's new token
// when deciding whether words were dropped.
constexpr int kRealignSearchWindow = 4;

inline SpeculativeDecodeOutcome run_speculative_decode(
    const int *draft, int draft_len, int bos_id, int max_realign_attempts,
    const SpeculativeDecodeHooks &hooks) {
  SpeculativeDecodeOutcome outcome;

  // Draft tokens not yet accepted or discarded. tail[0] is the token the
  // model most recently rejected.
  const int *tail = draft;
  int tail_len = draft_len;
  // Set when the previous attempt assumed tail[0] had been respelled. If that
  // attempt accepted nothing, tail[0] is kept in case the model's token was an
  // insertion and tail[0] is still to come.
  bool substitution_pending = false;

  // The first pass verifies the whole draft from BOS (or just runs BOS when
  // there is no draft). Later passes continue from the newest accepted token.
  int head = bos_id;
  const int *cand = tail;
  int cand_len = tail_len;
  bool first_pass = true;

  while (true) {
    int next = 0;
    const int accepted = hooks.verify(head, cand, cand_len, &next);
    if (accepted < 0) {
      outcome.ok = false;
      return outcome;
    }
    if (first_pass) {
      first_pass = false;
      tail += accepted;
      tail_len -= accepted;
    } else if (substitution_pending) {
      if (accepted > 0) {
        // The respelling verified: tail[0] really was replaced.
        tail += 1 + accepted;
        tail_len -= 1 + accepted;
        substitution_pending = false;
      }
      // accepted == 0: keep tail[0]; the search below may find the model's
      // token was inserted ahead of it.
    } else {
      tail += accepted;
      tail_len -= accepted;
    }

    if (!hooks.accept(next)) {
      return outcome;
    }
    head = next;

    // Choose what to verify next. No draft left, or realignment disabled or
    // exhausted: an empty candidate, which is a plain auto-regressive step.
    cand = nullptr;
    cand_len = 0;
    if (tail_len > 0 && outcome.realign_attempts < max_realign_attempts) {
      int found = -1;
      for (int k = 0; k < std::min(kRealignSearchWindow, tail_len); ++k) {
        if (tail[k] == next) {
          found = k;
          break;
        }
      }
      if (found >= 0) {
        // ``next`` is the draft token at offset ``found``: the ones before it
        // were dropped (or, at found == 0 after a failed substitution, the
        // inserted token is now behind us). Continue after it.
        tail += found + 1;
        tail_len -= found + 1;
        substitution_pending = false;
        cand = tail;
        cand_len = tail_len;
      } else {
        if (substitution_pending) {
          // Two unknown tokens in a row where the draft had tail[0]: give up
          // on tail[0] and try one more respelling.
          tail += 1;
          tail_len -= 1;
        }
        if (tail_len > 1) {
          // ``next`` replaced tail[0]; the rest of the draft should follow.
          substitution_pending = true;
          cand = tail + 1;
          cand_len = tail_len - 1;
        }
      }
      if (cand_len > 0) {
        ++outcome.realign_attempts;
      }
    }
  }
}

#endif  // MOONSHINE_SPECULATIVE_REALIGN_H
