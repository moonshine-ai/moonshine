package ai.moonshine.voice;

/**
 * One ranked intent from {@link IntentRecognizer#getClosestIntents}.
 *
 * <p>Internal to the library; intent matching is reached through {@link DialogFlow}.
 */
final class IntentMatch {
  final String canonicalPhrase;
  final float similarity;

  IntentMatch(String canonicalPhrase, float similarity) {
    this.canonicalPhrase = canonicalPhrase;
    this.similarity = similarity;
  }
}
