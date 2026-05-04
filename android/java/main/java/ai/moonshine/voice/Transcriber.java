package ai.moonshine.voice;

import ai.moonshine.voice.JNI;
import ai.moonshine.voice.TranscriberOption;
import android.content.res.AssetManager;
import android.util.Log;

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
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

public class Transcriber {
  private int transcriberHandle = -1;
  private int defaultStreamHandle = -1;
  private final List<Consumer<TranscriptEvent>> listeners =
      new CopyOnWriteArrayList<>();
  private final ExecutorService executor = Executors.newSingleThreadExecutor();

  private final List<TranscriberOption> options = new ArrayList<>();

  /**
   * Default flags applied to every transcription call (i.e. updates triggered
   * by {@link #addAudio}, {@link #stop}, and {@link #transcribeWithoutStreaming}).
   * Set to {@link JNI#MOONSHINE_FLAG_SPELLING_MODE} via
   * {@link #setTranscribeFlags(int)} to enable the spelling-fusion path.
   */
  private int transcribeFlags = 0;

  private final Map<Long, TranscriptLine> completedLines = new ConcurrentHashMap<>();
  private final Lock completedLinesLock = new ReentrantLock();

  public Transcriber() {}

  public Transcriber(List<TranscriberOption> options) {
    this.options.addAll(options);
  }

  /**
   * Sets the flags applied to subsequent transcription calls. Callers can
   * {@code OR} {@link JNI#MOONSHINE_FLAG_SPELLING_MODE} to drive the
   * spelling-fusion path on completed lines (provided a spelling model has
   * been bound through {@code spelling_model_path} or a non-null spelling
   * buffer in {@link #loadFromMemory}).
   */
  public void setTranscribeFlags(int flags) { this.transcribeFlags = flags; }

  public int getTranscribeFlags() { return this.transcribeFlags; }

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
    this.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                        /*spellingModelData=*/null, modelArch);
  }

  /**
   * Loads a transcriber from in-memory model buffers, with an optional
   * spelling-CNN payload for the alphanumeric fusion path.
   *
   * @param spellingModelData Optional spelling {@code .ort} bytes; pass
   *                          {@code null} to skip spelling fusion.
   */
  public void loadFromMemory(byte[] encoderModelData, byte[] decoderModelData,
                             byte[] tokenizerData, byte[] spellingModelData,
                             int modelArch) {
    JNI.ensureLibraryLoaded();
    this.transcriberHandle = JNI.moonshineLoadTranscriberFromMemory(
        encoderModelData, decoderModelData, tokenizerData, spellingModelData,
        modelArch, options.toArray(new TranscriberOption[0]));
    if (this.transcriberHandle < 0) {
      throw new RuntimeException("Failed to load transcriber from memory");
    }
    this.getDefaultStreamHandle();
  }

  public void loadFromAssets(AppCompatActivity parentContext, String path,
                             int modelArch) {
    this.loadFromAssets(parentContext, path, /*spellingAssetPath=*/null,
                        modelArch);
  }

  /**
   * Loads a transcriber from APK assets, optionally including a spelling
   * model so the spelling-fusion path can be used.
   *
   * @param spellingAssetPath Path to the spelling {@code .ort} asset
   *                          (e.g. {@code "spelling_cnn.ort"}). May be
   *                          {@code null} to skip spelling fusion.
   */
  public void loadFromAssets(AppCompatActivity parentContext, String path,
                             String spellingAssetPath, int modelArch) {
    AssetManager assetManager = parentContext.getAssets();
    String encoderModelPath = path + "/encoder_model.ort";
    String decoderModelPath = path + "/decoder_model_merged.ort";
    String tokenizerPath = path + "/tokenizer.bin";

    byte[] encoderModelData = readAllBytes(assetManager, encoderModelPath);
    byte[] decoderModelData = readAllBytes(assetManager, decoderModelPath);
    byte[] tokenizerData = readAllBytes(assetManager, tokenizerPath);
    byte[] spellingModelData = null;
    if (spellingAssetPath != null && !spellingAssetPath.isEmpty()) {
      spellingModelData = readAllBytes(assetManager, spellingAssetPath);
    }
    this.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                        spellingModelData, modelArch);
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
    return this.transcribeWithoutStreaming(audioData, sampleRate,
                                           this.transcribeFlags);
  }

  public Transcript transcribeWithoutStreaming(float[] audioData, int sampleRate,
                                               int flags) {
    return JNI.moonshineTranscribeWithoutStreaming(this.transcriberHandle,
                                                   audioData, sampleRate, flags);
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
    // to get the final transcript. We OR in the configured transcribeFlags
    // (e.g. MOONSHINE_FLAG_SPELLING_MODE) so the trailing segment runs
    // through the same path as live updates.
    Transcript transcript = JNI.moonshineTranscribeStream(
        this.transcriberHandle, streamHandle,
        JNI.MOONSHINE_FLAG_FORCE_UPDATE | this.transcribeFlags);
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
                                  audioData, sampleRate, this.transcribeFlags);
    Transcript transcript = JNI.moonshineTranscribeStream(
        this.transcriberHandle, streamHandle, this.transcribeFlags);
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
        // There's a potential race condition when stop() is called from a different thread
        // than the one that called addAudioToStream(), which is fairly common since the
        // former is triggered by user interface.
        // Most of the events are idempotent, so we can just ignore them if they're already
        // emitted, but many applications need to be sure they only receive a single event
        // for each completed line, so we do some checking to ensure that here.
        completedLinesLock.lock();
        TranscriptLine previousLine = completedLines.get(line.id);
        if (previousLine == null) {
          completedLines.put(line.id, line);
        }
        completedLinesLock.unlock();
        if (previousLine == null) {
          this.emit(new TranscriptEvent.LineCompleted(line, streamHandle));
        }
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
