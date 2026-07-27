package ai.moonshine.androidtranscriber;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import ai.moonshine.voice.MicTranscriber;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal microphone transcription sample.
 *
 * <p>Nothing is bundled in the APK: the transcriber downloads the streaming English model on first
 * run into a managed cache directory and reuses it thereafter. It also asks for the recording
 * permission itself, and delivers transcript callbacks on the main thread, so the only plumbing
 * left here is the background thread its two blocking calls need.
 */
public class MainActivity extends AppCompatActivity {

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private MicTranscriber mic;
    private TextView statusText;
    private TextView transcriptText;
    private boolean listening;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        transcriptText = findViewById(R.id.transcriptText);
        StringBuilder finishedLines = new StringBuilder();

        mic = new MicTranscriber(this)
                .onText(text -> transcriptText.setText(finishedLines + text))
                .onLine(line -> {
                    finishedLines.append(line.text == null ? "" : line.text).append('\n');
                    transcriptText.setText(finishedLines.toString());
                })
                .onError(error -> statusText.setText("Microphone error: " + error.getMessage()))
                .onProgress((fraction, file) ->
                        statusText.setText("Downloading " + file + " (" + (int) (fraction * 100)
                                + "%)…"));

        statusText.setText("Downloading model (first run only)…");
        worker.execute(() -> {
            try {
                mic.load();
                runOnUiThread(() ->
                        statusText.setText("Ready. Tap Start to transcribe from the microphone."));
            } catch (RuntimeException e) {
                runOnUiThread(() -> statusText.setText("Failed to load model: " + e.getMessage()));
            }
        });

        findViewById(R.id.startButton).setOnClickListener(v -> {
            if (listening) {
                return;
            }
            // start() puts up the permission dialog if it needs to, and waits.
            worker.execute(() -> {
                try {
                    mic.start();
                    runOnUiThread(() -> {
                        listening = true;
                        statusText.setText("Listening…");
                    });
                } catch (RuntimeException e) {
                    runOnUiThread(() -> statusText.setText("Start failed: " + e.getMessage()));
                }
            });
        });

        findViewById(R.id.stopButton).setOnClickListener(v -> {
            if (!listening) {
                return;
            }
            mic.stop();
            listening = false;
            statusText.setText("Stopped.");
        });
    }

    @Override
    protected void onDestroy() {
        mic.close();
        worker.shutdown();
        super.onDestroy();
    }
}
