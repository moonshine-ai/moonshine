package ai.moonshine.voice;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Matches an utterance to one of several key/phrases groups by meaning.
 *
 * <p>Each phrase is embedded once and cached, the utterance is embedded once per
 * call, and the key of the best-scoring phrase at or above the threshold wins.
 * Without an {@link EmbeddingModel} it falls back to case-insensitive substring
 * matching, which is what keeps dialogs working before {@link AgentFlow#load}.
 */
final class PhraseMatcher {
  /** A key and the phrases that select it. */
  static final class Group {
    final String key;
    final List<String> phrases;

    Group(String key, List<String> phrases) {
      this.key = key;
      this.phrases = phrases;
    }
  }

  @Nullable private final EmbeddingModel model;
  private final Map<String, float[]> cache = new HashMap<>();

  PhraseMatcher(@Nullable EmbeddingModel model) {
    this.model = model;
  }

  /** The best-matching key, or null when nothing clears {@code threshold}. */
  @Nullable
  String match(String utterance, List<Group> groups, float threshold) {
    if (utterance == null || utterance.isEmpty() || groups.isEmpty()) {
      return null;
    }
    if (model == null) {
      return matchSubstring(utterance, groups);
    }

    float[] utteranceEmbedding;
    try {
      utteranceEmbedding = model.calculateEmbedding(utterance);
    } catch (RuntimeException e) {
      return null;
    }
    String bestKey = null;
    float bestScore = -1;
    for (Group group : groups) {
      for (String phrase : group.phrases) {
        if (phrase == null || phrase.isEmpty()) {
          continue;
        }
        try {
          float score = model.distance(utteranceEmbedding, embeddingFor(phrase));
          if (score > bestScore) {
            bestScore = score;
            bestKey = group.key;
          }
        } catch (RuntimeException e) {
          // A phrase we cannot embed simply does not match.
        }
      }
    }
    return bestScore >= threshold ? bestKey : null;
  }

  /** The best-matching phrase, treating each phrase as its own key. */
  @Nullable
  String match(String utterance, String[] phrases, float threshold) {
    List<Group> groups = new ArrayList<>(phrases.length);
    for (String phrase : phrases) {
      groups.add(new Group(phrase, Collections.singletonList(phrase)));
    }
    return match(utterance, groups, threshold);
  }

  @Nullable
  private String matchSubstring(String utterance, List<Group> groups) {
    String lower = utterance.toLowerCase();
    for (Group group : groups) {
      for (String phrase : group.phrases) {
        if (phrase == null || phrase.isEmpty()) {
          continue;
        }
        String needle = phrase.toLowerCase();
        if (lower.equals(needle) || lower.contains(needle)) {
          return group.key;
        }
      }
    }
    return null;
  }

  private float[] embeddingFor(String phrase) {
    synchronized (cache) {
      float[] cached = cache.get(phrase);
      if (cached != null) {
        return cached;
      }
    }
    float[] computed = model.calculateEmbedding(phrase);
    synchronized (cache) {
      cache.put(phrase, computed);
    }
    return computed;
  }
}
