package ai.moonshine.examples.agentflow;

import ai.moonshine.examples.agentflow.databinding.ActivityMainBinding;
import ai.moonshine.voice.AgentFlow;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import androidx.appcompat.app.AppCompatActivity;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * A voice-driven wifi setup, the whole thing.
 *
 * <p>Say "set up wifi" to start it, "cancel" to abandon it, or "start over" to run it again. The
 * last two are built in, so this file never mentions them. You can also type an utterance, which
 * takes exactly the same path and is the only one that works on an emulator with no microphone.
 *
 * <p>Nothing is bundled in the APK: one {@code load()} call downloads the speech, embedding and
 * voice models into a managed cache and wires the three engines together.
 */
public class MainActivity extends AppCompatActivity {

    /** AgentFlow's blocking calls (load, start, handleUtterance) run here. */
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private ActivityMainBinding binding;
    private AgentFlow agent;

    private final StringBuilder conversation = new StringBuilder();
    private boolean ready;
    private boolean listening;

    /**
     * The conversation, as straight-line code. Every {@code ask} speaks and then waits, so there is
     * no state machine to write and no callbacks to thread together.
     */
    private void wifiSetup(AgentFlow.Dialog d) {
        String ssid = d.ask("What's the name of your wifi network?");
        if (!d.confirm("I heard " + ssid + ". Is that right?")) {
            d.say("No problem, let's start over.");
            d.restart();
        }

        Map<String, List<String>> options = new LinkedHashMap<>();
        options.put("open", Arrays.asList("open", "no password", "none"));
        options.put("password", Arrays.asList("password", "secured", "protected", "wpa"));
        String security = d.choose("Is the network open, or does it use a password?", options);
        if ("password".equals(security)) {
            String password = d.ask("What's the password? Spell it out one letter at a time.");
            d.say("Got it, " + AgentFlow.spellOut(password) + ".");
        }

        if (d.confirm("Apply these changes?")) {
            d.say("Done. Connecting to " + ssid + ".");
        } else {
            d.say("Okay, nothing changed.");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        agent =
                new AgentFlow(this)
                        .onHeard(text -> runOnUiThread(() -> append("you", text)))
                        .onSaid(text -> runOnUiThread(() -> append("assistant", text)))
                        .onError(
                                error ->
                                        runOnUiThread(
                                                () ->
                                                        binding.statusText.setText(
                                                                "Flow error: "
                                                                        + error.getMessage())))
                        .onProgress((fraction, file) -> runOnUiThread(() -> showProgress(fraction, file)));

        agent.listenFor("set up wifi", this::wifiSetup);

        binding.startButton.setOnClickListener(v -> startListening());
        binding.stopButton.setOnClickListener(v -> stopListening());
        binding.sendButton.setOnClickListener(v -> sendTypedUtterance());
        binding.inputText.setOnEditorActionListener(
                (v, actionId, event) -> {
                    if (actionId == EditorInfo.IME_ACTION_SEND) {
                        sendTypedUtterance();
                        return true;
                    }
                    return false;
                });

        binding.statusText.setText(R.string.loading);
        updateUiState();
        worker.execute(
                () -> {
                    try {
                        agent.load();
                        runOnUiThread(
                                () -> {
                                    ready = true;
                                    binding.downloadProgress.setVisibility(View.GONE);
                                    binding.statusText.setText(R.string.prompt);
                                    updateUiState();
                                });
                    } catch (RuntimeException e) {
                        runOnUiThread(
                                () -> {
                                    binding.downloadProgress.setVisibility(View.GONE);
                                    binding.statusText.setText(
                                            "Failed to load: " + e.getMessage());
                                });
                    }
                });
    }

    @Override
    protected void onDestroy() {
        agent.close();
        worker.shutdown();
        super.onDestroy();
    }

    private void startListening() {
        if (!ready || listening) {
            return;
        }
        // startListening() puts up the permission dialog if it needs to, and waits.
        worker.execute(
                () -> {
                    try {
                        agent.startListening();
                        runOnUiThread(
                                () -> {
                                    listening = true;
                                    binding.statusText.setText(R.string.prompt);
                                    updateUiState();
                                });
                    } catch (RuntimeException e) {
                        runOnUiThread(
                                () ->
                                        binding.statusText.setText(
                                                "Couldn't open the microphone: " + e.getMessage()));
                    }
                });
    }

    private void stopListening() {
        if (!listening) {
            return;
        }
        worker.execute(
                () -> {
                    agent.stopListening();
                    runOnUiThread(
                            () -> {
                                listening = false;
                                binding.statusText.setText("Stopped.");
                                updateUiState();
                            });
                });
    }

    /** Feeds typed text in as though it had been heard. */
    private void sendTypedUtterance() {
        CharSequence entered = binding.inputText.getText();
        String text = entered == null ? "" : entered.toString().trim();
        if (text.isEmpty() || !ready) {
            return;
        }
        binding.inputText.setText("");
        // handleUtterance runs the flow, which blocks on each prompt until the
        // answer arrives, so it cannot run on the main thread.
        worker.execute(() -> agent.handleUtterance(text));
    }

    private void append(String speaker, String text) {
        conversation.append(speaker).append(": ").append(text).append('\n');
        binding.transcriptText.setText(conversation.toString());
        binding.transcriptScroll.post(
                () -> binding.transcriptScroll.fullScroll(View.FOCUS_DOWN));
    }

    private void showProgress(float fraction, String file) {
        int slash = file.lastIndexOf('/');
        String name = slash < 0 ? file : file.substring(slash + 1);
        binding.statusText.setText(
                "Downloading " + name + " (" + (int) (fraction * 100) + "%)…");
        binding.downloadProgress.setVisibility(View.VISIBLE);
        binding.downloadProgress.setProgress(Math.max(0, Math.min(100, (int) (fraction * 100))));
    }

    private void updateUiState() {
        binding.startButton.setEnabled(ready && !listening);
        binding.stopButton.setEnabled(listening);
        binding.sendButton.setEnabled(ready);
        binding.inputText.setEnabled(ready);
    }
}
