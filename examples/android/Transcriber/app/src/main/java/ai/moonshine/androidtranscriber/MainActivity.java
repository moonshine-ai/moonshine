package ai.moonshine.androidtranscriber;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import ai.moonshine.voice.DownloadProgress;
import ai.moonshine.voice.JNI;
import ai.moonshine.voice.LoadCallback;
import ai.moonshine.voice.MicTranscriber;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;

/**
 * Minimal microphone transcription sample. Downloads the Medium Streaming English model on first
 * run into a managed cache directory (nothing is bundled in the APK) and constructs the
 * transcriber in one call with {@link MicTranscriber#loadFromCatalog}. Progress and the ready
 * engine are delivered on the main thread, so no {@code Thread}/{@code runOnUiThread} plumbing is
 * needed here.
 */
public class MainActivity extends AppCompatActivity {

    private MicTranscriber transcriber;
    private TextView statusText;
    private TextView transcriptText;
    private boolean listening;
    private boolean pendingStartAfterPermission;

    private final ActivityResultLauncher<String> micPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (!Boolean.TRUE.equals(granted) || transcriber == null) {
                    pendingStartAfterPermission = false;
                    statusText.setText("Microphone permission is required.");
                    return;
                }
                transcriber.onMicPermissionGranted();
                if (pendingStartAfterPermission) {
                    pendingStartAfterPermission = false;
                    try {
                        transcriber.start();
                        listening = true;
                        statusText.setText("Listening…");
                    } catch (RuntimeException e) {
                        statusText.setText("Start failed: " + e.getMessage());
                    }
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        transcriptText = findViewById(R.id.transcriptText);

        statusText.setText("Downloading model (first run only)…");
        bootstrapTranscriber();

        findViewById(R.id.startButton)
                .setOnClickListener(
                        v -> {
                            if (transcriber == null) {
                                return;
                            }
                            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                                    != PackageManager.PERMISSION_GRANTED) {
                                pendingStartAfterPermission = true;
                                micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO);
                                return;
                            }
                            transcriber.onMicPermissionGranted();
                            try {
                                transcriber.start();
                                listening = true;
                                statusText.setText("Listening…");
                            } catch (RuntimeException e) {
                                statusText.setText("Start failed: " + e.getMessage());
                            }
                        });

        findViewById(R.id.stopButton)
                .setOnClickListener(
                        v -> {
                            if (transcriber == null || !listening) {
                                return;
                            }
                            try {
                                transcriber.stop();
                            } catch (RuntimeException ignored) {
                            }
                            listening = false;
                            statusText.setText("Stopped.");
                        });
    }

    /**
     * Downloads the Medium Streaming English model on first run and constructs the transcriber. All
     * callbacks land on the main thread; the per-line transcript events, however, arrive on the
     * transcriber's own thread and are still marshalled with {@code runOnUiThread}.
     */
    private void bootstrapTranscriber() {
        MicTranscriber.loadFromCatalog(
                this,
                "en",
                JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
                new LoadCallback<MicTranscriber>() {
                    @Override
                    public void onProgress(DownloadProgress progress) {
                        int pct =
                                progress.bytesTotal > 0
                                        ? (int) (progress.bytesDownloaded * 100 / progress.bytesTotal)
                                        : 0;
                        statusText.setText(
                                "Downloading model " + progress.fileIndex + "/" + progress.totalFiles
                                        + " (" + pct + "%)…");
                    }

                    @Override
                    public void onSuccess(MicTranscriber loaded) {
                        loaded.addListener(
                                event ->
                                        event.accept(
                                                new TranscriptEventListener() {
                                                    @Override
                                                    public void onLineTextChanged(
                                                            @NonNull TranscriptEvent.LineTextChanged e) {
                                                        runOnUiThread(
                                                                () ->
                                                                        transcriptText.setText(
                                                                                e.line.text != null
                                                                                        ? e.line.text
                                                                                        : ""));
                                                    }

                                                    @Override
                                                    public void onLineCompleted(
                                                            @NonNull TranscriptEvent.LineCompleted e) {
                                                        runOnUiThread(
                                                                () ->
                                                                        transcriptText.append(
                                                                                (e.line.text != null
                                                                                                ? e.line.text
                                                                                                : "")
                                                                                        + "\n"));
                                                    }
                                                }));
                        transcriber = loaded;
                        if (ContextCompat.checkSelfPermission(
                                        MainActivity.this, Manifest.permission.RECORD_AUDIO)
                                == PackageManager.PERMISSION_GRANTED) {
                            transcriber.onMicPermissionGranted();
                        }
                        statusText.setText("Ready. Tap Start to transcribe from the microphone.");
                    }

                    @Override
                    public void onError(Throwable error) {
                        statusText.setText("Failed to load models: " + error.getMessage());
                        transcriber = null;
                    }
                });
    }

    @Override
    protected void onDestroy() {
        if (transcriber != null) {
            try {
                transcriber.stop();
            } catch (RuntimeException ignored) {
            }
        }
        super.onDestroy();
    }
}
