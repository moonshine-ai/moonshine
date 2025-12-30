# Moonshine Voice

The fast, accurate, on-device AI library for building interactive voice applications.

## About

If you've ever tried to build your own voice interface, or even just capture transcriptions in real time, you'll know how hard and frustrating it can be. Moonshine Voice is the batteries-included framework that makes it easy.

 - Cross-platform: Runs on Android, iOS, Linux, MacOS, Windows, and Raspberry Pis.
 - Full Stack: Includes voice-activity detection, diarization, transcription, speech understanding, and text-to-speech in one package.
 - Easy API: Your app will be notified whenever someone talks, using an event-based interface that mirrors how button presses and other GUI interactions are traditionally handled.
 - High Level: You don't have to worry about the intricacies of sound drivers and sample rates, the library takes care of the details.
 - Open Source: The code is released under an MIT license, so you can rely on it long term.
 - On Device: Everything happens locally, so there are no cloud usage fees, and you can guarantee availability.
 - Cutting Edge Technology: We train our own models from scratch, so we can offer you the world's fastest and most accurate speech to text models.


 ## Android

 The easiest way to get started is to download this repo and open the project in `examples/android/Transcriber` in Android Studio. This shows how you can build a simple transcription app using the library.

 The key code is in MainActivity.java, here is a breakdown of what it's doing:

```java
    transcriber = new MicTranscriber(this);
    transcriber.loadFromAssets(this, "base-en", JNI.MOONSHINE_MODEL_ARCH_BASE);
    transcriber.addListener(
        event -> event.accept(new TranscriptEventListener() {
          @Override
          public void onLineStarted(TranscriptEvent.LineStarted e) {
            runOnUiThread(() -> {
              adapter.addLine("...");
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
          @Override
          public void onLineTextChanged(TranscriptEvent.LineTextChanged e) {
            runOnUiThread(() -> {
              adapter.updateLastLine(e.line.text);
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
          @Override
          public void onLineCompleted(TranscriptEvent.LineCompleted e) {
            runOnUiThread(() -> {
              adapter.updateLastLine(e.line.text);
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
        }));
```