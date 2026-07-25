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
import ai.moonshine.voice.AssetDownloader;
import ai.moonshine.voice.JNI;
import ai.moonshine.voice.MicTranscriber;
import ai.moonshine.voice.ModelSpec;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;
import java.io.File;

/**
 * Minimal microphone transcription sample. Downloads the Medium Streaming English model on first
 * run into {@code filesDir} via {@link AssetDownloader} (nothing is bundled in the APK) and loads
 * it with {@link MicTranscriber#loadFromFiles(String, int)}.
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
     * Downloads the Medium Streaming English model into {@code filesDir} on first run (off the main
     * thread) and loads it. Subsequent launches reuse the cached files.
     */
    private void bootstrapTranscriber() {
        new Thread(() -> {
            try {
                MicTranscriber t = new MicTranscriber();
                t.addListener(
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
                                                                        (e.line.text != null ? e.line.text : "")
                                                                                + "\n"));
                                            }
                                        }));

                File root = new File(getFilesDir(), "medium-streaming-en");
                //noinspection ResultOfMethodCallIgnored
                root.mkdirs();
                new AssetDownloader().ensureModelPresent(
                        root,
                        ModelSpec.stt("en", JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, false),
                        (relativePath, fileIndex, totalFiles, bytesDownloaded, bytesTotal) -> {
                            int pct = bytesTotal > 0 ? (int) (bytesDownloaded * 100 / bytesTotal) : 0;
                            runOnUiThread(
                                    () ->
                                            statusText.setText(
                                                    "Downloading model " + fileIndex + "/" + totalFiles
                                                            + " (" + pct + "%)…"));
                        });
                t.loadFromFiles(root.getAbsolutePath(), JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING);

                runOnUiThread(
                        () -> {
                            transcriber = t;
                            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                                    == PackageManager.PERMISSION_GRANTED) {
                                transcriber.onMicPermissionGranted();
                            }
                            statusText.setText("Ready. Tap Start to transcribe from the microphone.");
                        });
            } catch (Exception e) {
                runOnUiThread(
                        () -> {
                            statusText.setText("Failed to load models: " + e.getMessage());
                            transcriber = null;
                        });
            }
        }).start();
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
