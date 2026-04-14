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
import ai.moonshine.voice.JNI;
import ai.moonshine.voice.MicTranscriber;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;

/**
 * Minimal microphone transcription sample. Bundles {@code base-en} under {@code assets/base-en/}
 * (release packaging). {@link MicTranscriber} has a no-arg constructor; pass {@code this} only to
 * {@link MicTranscriber#loadFromAssets(AppCompatActivity, String, int)}.
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

        try {
            transcriber = new MicTranscriber();
            transcriber.addListener(
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
            transcriber.loadFromAssets(this, "base-en", JNI.MOONSHINE_MODEL_ARCH_BASE);
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED) {
                transcriber.onMicPermissionGranted();
            }
            statusText.setText("Ready. Tap Start to transcribe from the microphone.");
        } catch (RuntimeException e) {
            statusText.setText("Failed to load models: " + e.getMessage());
            transcriber = null;
        }

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
