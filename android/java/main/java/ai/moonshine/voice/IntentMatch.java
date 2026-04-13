package ai.moonshine.voice;

/** One ranked intent from {@link IntentRecognizer#getClosestIntents}. */
public final class IntentMatch {
  public final String canonicalPhrase;
  public final float similarity;

  public IntentMatch(String canonicalPhrase, float similarity) {
    this.canonicalPhrase = canonicalPhrase;
    this.similarity = similarity;
  }
}
