package ai.moonshine.voice;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.lang.ref.WeakReference;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

public class MicTranscriber extends Transcriber {
  private boolean isRunning = false;
  private boolean isMicCaptureLoopStarted = false;
  private MicCaptureProcessor micCaptureProcessor;
  private CompletableFuture<Void> isLoadedSignal = new CompletableFuture<>();
  private CompletableFuture<Void> hasMicPermissionSignal =
      new CompletableFuture<>();

  public MicTranscriber() {
    super();
    // When both isLoadedSignal and hasMicPermissionSignal are complete, run
    // startProcessing()
    CompletableFuture.allOf(isLoadedSignal, hasMicPermissionSignal)
        .thenRun(this::startProcessing);
  }

  // These load* methods are overridden to complete the CompletableFuture when
  // the transcriber is loaded, so we can continue with other post-loading
  // actions.
  public void loadFromAssets(AppCompatActivity parentContext,
                             String modelRootDir, int modelArch) {
    super.loadFromAssets(parentContext, modelRootDir, modelArch);
    this.isLoadedSignal.complete(null);
  }

  public void loadFromAssets(AppCompatActivity parentContext,
                             String modelRootDir, String spellingAssetPath,
                             int modelArch) {
    super.loadFromAssets(parentContext, modelRootDir, spellingAssetPath,
                         modelArch);
    this.isLoadedSignal.complete(null);
  }

  public void loadFromFiles(String modelRootDir, int modelArch) {
    super.loadFromFiles(modelRootDir, modelArch);
    this.isLoadedSignal.complete(null);
  }

  /**
   * Downloads the speech-to-text model for {@code language} / {@code modelArch} (if not already
   * present, into a managed {@link ModelCache} directory) on a background thread, then builds and
   * returns a ready {@link MicTranscriber} through {@code callback} on the main thread.
   *
   * <p>The same {@code modelArch} drives both the download manifest and the load, so the caller
   * specifies it once. Call {@link #cancel()} on the returned handle to abort.
   *
   * @param context any {@link Context}; the application context is retained.
   * @param language language code (e.g. {@code "en"}).
   * @param modelArch a {@code MOONSHINE_MODEL_ARCH_*} value.
   */
  public static Cancellable loadFromCatalog(Context context, String language, int modelArch,
                                            LoadCallback<MicTranscriber> callback) {
    ModelSpec spec = ModelSpec.stt(language, modelArch, false);
    return CatalogLoader.load(context, Collections.singletonList(spec), directories -> {
      MicTranscriber transcriber = new MicTranscriber();
      transcriber.loadFromFiles(directories.get(spec).getAbsolutePath(), modelArch);
      return transcriber;
    }, callback);
  }

  public void loadFromMemory(byte[] encoderModelData, byte[] decoderModelData,
                             byte[] tokenizerData, int modelArch) {
    super.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                         modelArch);
    this.isLoadedSignal.complete(null);
  }

  public void loadFromMemory(byte[] encoderModelData, byte[] decoderModelData,
                             byte[] tokenizerData, byte[] spellingModelData,
                             int modelArch) {
    super.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                         spellingModelData, modelArch);
    this.isLoadedSignal.complete(null);
  }

  public void onMicPermissionGranted() {
    this.hasMicPermissionSignal.complete(null);
  }

  private void startProcessing() {
    startMicCaptureLoop();
    startAudioProcessingLoop();
  }

  private void startAudioProcessingLoop() {
    Thread audioProcessingThread = new Thread(new Runnable() {
      @Override
      public void run() {
        Log.d("MainActivity", "Starting audio processing thread");
        audioProcessingLoop();
      }
    });
    audioProcessingThread.start();
  }

  private void startMicCaptureLoop() {
    if (isMicCaptureLoopStarted) {
      return;
    }
    isMicCaptureLoopStarted = true;
    micCaptureProcessor = new MicCaptureProcessor();
    Thread micThread = new Thread(micCaptureProcessor);
    micThread.start();
  }

  public void stop() {
    super.stop();
    this.isRunning = false;
  }

  public void start() {
    super.start();
    this.isRunning = true;
  }

  private void audioProcessingLoop() {
    int streamHandle = createStream();
    startStream(streamHandle);
    this.isRunning = true;
    boolean wasRunning = this.isRunning;
    while (!Thread.currentThread().isInterrupted()) {
      float[] audioData = micCaptureProcessor.consumeAudio();
      if (!this.isRunning && !wasRunning) {
        continue;
      }
      if (this.isRunning && !wasRunning) {
        startStream(streamHandle);
      }
      if (this.isRunning || wasRunning) {
        addAudioToStream(streamHandle, audioData, 16000);
      }
      if (!this.isRunning && wasRunning) {
        stopStream(streamHandle);
      }
      wasRunning = this.isRunning;
    }
    stopStream(streamHandle);
    freeStream(streamHandle);
  }
}
