#include <cmath>

#include "stt/stt.h"

namespace spelling {

int Argmax(const float* logits, int n_logits) {
  if (n_logits <= 0) return 0;
  int best = 0;
  float best_v = logits[0];
  for (int i = 1; i < n_logits; ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  return best;
}

float SoftmaxProb(const float* logits, int n_logits, int index) {
  if (n_logits <= 0 || index < 0 || index >= n_logits) return 0.0f;
  float m = logits[0];
  for (int i = 1; i < n_logits; ++i) {
    if (logits[i] > m) m = logits[i];
  }
  // Accumulate in double; the exponent range stays small after
  // subtracting the max so float would also work, but double is free
  // on the Pico 2's M33 (no FPU penalty) for this one-shot call.
  double s = 0.0;
  for (int i = 0; i < n_logits; ++i) {
    s += std::exp(static_cast<double>(logits[i] - m));
  }
  if (s <= 0.0) return 0.0f;
  return static_cast<float>(std::exp(static_cast<double>(logits[index] - m)) /
                            s);
}

}  // namespace spelling
