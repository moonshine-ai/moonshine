package ai.moonshine.voice;

import ai.moonshine.voice.JNI;
import ai.moonshine.voice.TranscriberOption;
import android.content.res.AssetManager;
import androidx.appcompat.app.AppCompatActivity;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class Transcriber {
  private int transcriberHandle = -1;
  private int defaultStreamHandle = -1;
  private final List<Consumer<TranscriptEvent>> listeners =
      new CopyOnWriteArrayList<>();
  private final ExecutorService executor = Executors.newSingleThreadExecutor();

  private final List<TranscriberOption> options = new ArrayList<>();

  public Transcriber() {}

  public Transcriber(List<TranscriberOption> options) {
    this.options.addAll(options);
  }

  public void loadFromFiles(String modelRootDir, int modelArch) {
    JNI.ensureLibraryLoaded();
    this.transcriberHandle = JNI.moonshineLoadTranscriberFromFiles(
        modelRootDir, modelArch, options.toArray(new TranscriberOption[0]));
    if (this.transcriberHandle < 0) {
      throw new RuntimeException("Failed to load transcriber from files: '" +
                                 modelRootDir + "'");
    }
    this.getDefaultStreamHandle();
  }

  public void loadFromMemory(byte[] encoderModelData, byte[] decoderModelData,
                             byte[] tokenizerData, int modelArch) {
    JNI.ensureLibraryLoaded();
    this.transcriberHandle = JNI.moonshineLoadTranscriberFromMemory(
        encoderModelData, decoderModelData, tokenizerData, modelArch,
        options.toArray(new TranscriberOption[0]));
    if (this.transcriberHandle < 0) {
      throw new RuntimeException("Failed to load transcriber from memory");
    }
    this.getDefaultStreamHandle();
  }

  public void loadFromAssets(AppCompatActivity parentContext, String path,
                             int modelArch) {
    AssetManager assetManager = parentContext.getAssets();
    String encoderModelPath = path + "/encoder_model.ort";
    String decoderModelPath = path + "/decoder_model_merged.ort";
    String tokenizerPath = path + "/tokenizer.bin";

    byte[] encoderModelData = readAllBytes(assetManager, encoderModelPath);
    byte[] decoderModelData = readAllBytes(assetManager, decoderModelPath);
    byte[] tokenizerData = readAllBytes(assetManager, tokenizerPath);
    this.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                        modelArch);
    if (this.transcriberHandle < 0) {
      throw new RuntimeException("Failed to load transcriber from assets: '" +
                                 path + "'");
    }
    this.getDefaultStreamHandle();
  }

  protected void finalize() throws Throwable {
    if (this.transcriberHandle >= 0) {
      if (this.defaultStreamHandle >= 0) {
        JNI.moonshineFreeStream(this.transcriberHandle,
                                this.defaultStreamHandle);
        this.defaultStreamHandle = -1;
      }
      JNI.moonshineFreeTranscriber(this.transcriberHandle);
      this.transcriberHandle = -1;
    }
  }

  public Transcript transcribeWithoutStreaming(float[] audioData,
                                               int sampleRate) {
    return JNI.moonshineTranscribeWithoutStreaming(this.transcriberHandle,
                                                   audioData, sampleRate, 0);
  }

  public int createStream() {
    return JNI.moonshineCreateStream(this.transcriberHandle, 0);
  }

  public void freeStream(int streamHandle) {
    JNI.moonshineFreeStream(this.transcriberHandle, streamHandle);
  }

  public void startStream(int streamHandle) {
    JNI.moonshineStartStream(this.transcriberHandle, streamHandle);
  }

  public void stopStream(int streamHandle) {
    JNI.moonshineStopStream(this.transcriberHandle, streamHandle);
    // There may be some audio left in the stream, so we need to transcribe it
    // to get the final transcript.
    Transcript transcript = JNI.moonshineTranscribeStream(
        this.transcriberHandle, streamHandle, JNI.MOONSHINE_FLAG_FORCE_UPDATE);
    this.notifyFromTranscript(transcript, streamHandle);
  }

  public void start() { this.startStream(this.getDefaultStreamHandle()); }

  public void stop() { this.stopStream(this.getDefaultStreamHandle()); }

  public void addListener(Consumer<TranscriptEvent> listener) {
    this.listeners.add(listener);
  }

  public void removeListener(Consumer<TranscriptEvent> listener) {
    this.listeners.remove(listener);
  }

  public void removeAllListeners() { this.listeners.clear(); }

  public void addAudio(float[] audioData, int sampleRate) {
    int streamHandle = this.getDefaultStreamHandle();
    this.addAudioToStream(streamHandle, audioData, sampleRate);
  }

  public void addAudioToStream(int streamHandle, float[] audioData,
                               int sampleRate) {
    JNI.moonshineAddAudioToStream(this.transcriberHandle, streamHandle,
                                  audioData, sampleRate);
    Transcript transcript =
        JNI.moonshineTranscribeStream(this.transcriberHandle, streamHandle, 0);
    if (transcript == null) {
      throw new RuntimeException("Failed to transcribe stream: " +
                                 streamHandle);
    }
    this.notifyFromTranscript(transcript, streamHandle);
  }

  private void notifyFromTranscript(Transcript transcript, int streamHandle) {
    for (TranscriptLine line : transcript.lines) {
      if (line.isNew) {
        this.emit(new TranscriptEvent.LineStarted(line, streamHandle));
      }
      if (line.isUpdated && !line.isNew && !line.isComplete) {
        this.emit(new TranscriptEvent.LineUpdated(line, streamHandle));
      }
      if (line.hasTextChanged) {
        this.emit(new TranscriptEvent.LineTextChanged(line, streamHandle));
      }
      if (line.isComplete && line.isUpdated) {
        this.emit(new TranscriptEvent.LineCompleted(line, streamHandle));
      }
    }
  }

  private void emit(TranscriptEvent event) {
    for (Consumer<TranscriptEvent> listener : this.listeners) {
      listener.accept(event);
    }
  }

  private int getDefaultStreamHandle() {
    if (this.defaultStreamHandle >= 0) {
      return this.defaultStreamHandle;
    }
    this.defaultStreamHandle = this.createStream();
    return this.defaultStreamHandle;
  }

  private static byte[] readAllBytes(AssetManager assetManager, String path) {
    try {
      InputStream is = assetManager.open(path);
      int size = is.available();
      byte[] buffer = new byte[size];
      is.read(buffer);
      is.close();
      return buffer;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
