package ai.moonshine.voice;

import android.Manifest;
import android.content.pm.PackageManager;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

public class MicTranscriber extends Transcriber {
  private boolean isRunning = false;
  private AppCompatActivity parentContext;
  private boolean isMicCaptureLoopStarted = false;
  private MicCaptureProcessor micCaptureProcessor;
  private CompletableFuture<Void> isLoadedSignal = new CompletableFuture<>();
  private CompletableFuture<Void> hasMicPermissionSignal =
      new CompletableFuture<>();

  public MicTranscriber(AppCompatActivity parentContext) {
    super();
    this.parentContext = parentContext;
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

  public void loadFromFiles(String modelRootDir, int modelArch) {
    super.loadFromFiles(modelRootDir, modelArch);
    this.isLoadedSignal.complete(null);
  }

  public void loadFromMemory(byte[] encoderModelData, byte[] decoderModelData,
                             byte[] tokenizerData, int modelArch) {
    super.loadFromMemory(encoderModelData, decoderModelData, tokenizerData,
                         modelArch);
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
