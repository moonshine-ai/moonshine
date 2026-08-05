package ai.moonshine.examples.texttospeech;

import ai.moonshine.examples.texttospeech.databinding.ActivityMainBinding;
import ai.moonshine.voice.ModelCache;
import ai.moonshine.voice.ModelSpec;
import ai.moonshine.voice.TextToSpeech;
import ai.moonshine.voice.TranscriberOption;
import ai.moonshine.voice.VoiceClone;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Speech synthesis with a voice picker and zero-shot voice cloning.
 *
 * <p>Nothing is bundled in the APK. The synthesizer downloads the Kokoro base model, the language's
 * G2P assets, and the selected voice on first use, into a managed per-language cache directory
 * ({@link ModelCache}) that it reuses thereafter.
 */
public class MainActivity extends AppCompatActivity {

    /** A language offered in the picker, covering both the Kokoro and Piper engines. */
    private static final class Language {
        final String id;
        final String displayName;

        Language(String id, String displayName) {
            this.id = id;
            this.displayName = displayName;
        }
    }

    /** One voice in the picker. */
    private static final class Voice {
        final String id;
        final String displayName;
        /** True when the voice's asset files are not on disk yet and must be downloaded. */
        final boolean needsDownload;

        Voice(String id, String displayName, boolean needsDownload) {
            this.id = id;
            this.displayName = displayName;
            this.needsDownload = needsDownload;
        }
    }

    private static final List<Language> LANGUAGES =
            Collections.unmodifiableList(
                    Arrays.asList(
                            new Language("ar_msa", "Arabic (MSA)"),
                            new Language("de", "German"),
                            new Language("en_us", "English (US)"),
                            new Language("en_gb", "English (UK)"),
                            new Language("es_ar", "Spanish (AR)"),
                            new Language("es_es", "Spanish (ES)"),
                            new Language("es_mx", "Spanish (MX)"),
                            new Language("fr", "French"),
                            new Language("hi", "Hindi"),
                            new Language("it", "Italian"),
                            new Language("ja", "Japanese"),
                            new Language("ko", "Korean"),
                            new Language("nl", "Dutch"),
                            new Language("pt_br", "Portuguese (BR)"),
                            new Language("pt_pt", "Portuguese (PT)"),
                            new Language("ru", "Russian"),
                            new Language("tr", "Turkish"),
                            new Language("uk", "Ukrainian"),
                            new Language("vi", "Vietnamese"),
                            new Language("zh_hans", "Chinese (Mandarin)")));

    private static final String DEFAULT_VOICE = "kokoro_af_alloy";

    private ActivityMainBinding binding;

    /** Moonshine's blocking calls (load, say, cloning) run here. */
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    /**
     * Volatile because the cloning path swaps this in from the worker thread, having just loaded
     * it, while the main thread reads it to decide what the buttons do.
     */
    @Nullable private volatile TextToSpeech tts;

    private Language selectedLanguage = languageFor("en_us");
    private List<Voice> availableVoices = new ArrayList<>();
    @Nullable private Voice selectedVoice;

    private boolean engineReady;
    private boolean isSpeaking;
    private boolean isLoading;
    private boolean spokenWelcome;

    /**
     * Whether {@link #tts} was built with {@code cloning(true)}. Cloning lives on the ZipVoice
     * engine, so a synthesizer built for a preset voice cannot clone and has to be replaced before
     * recording.
     */
    private boolean hasCloningEngine;

    /** Recording or cloning is under way. */
    private boolean isCloning;

    /** The synthesizer is speaking in a voice taken from the microphone. */
    private boolean isCloned;

    private boolean suppressSpinnerCallbacks;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        setSupportActionBar(binding.toolbar);

        setupLanguageSpinner();
        setupVoiceSpinner();

        binding.speakButton.setOnClickListener(v -> speakCurrentText());
        binding.recordButton.setOnClickListener(v -> cloneFromMicrophone());

        binding.inputText.addTextChangedListener(
                new TextWatcher() {
                    @Override
                    public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

                    @Override
                    public void onTextChanged(CharSequence s, int start, int before, int count) {}

                    @Override
                    public void afterTextChanged(Editable s) {
                        updateUiState();
                    }
                });

        refreshVoices();
        repopulateVoiceSpinner();
        loadSynthesizer(DEFAULT_VOICE, false);
    }

    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.close();
            tts = null;
        }
        worker.shutdown();
        super.onDestroy();
    }

    private static Language languageFor(String id) {
        for (Language language : LANGUAGES) {
            if (language.id.equals(id)) {
                return language;
            }
        }
        return LANGUAGES.get(0);
    }

    // -- Spinners ------------------------------------------------------------

    private void setupLanguageSpinner() {
        List<String> labels = new ArrayList<>();
        for (Language language : LANGUAGES) {
            labels.add(language.displayName);
        }
        binding.languageSpinner.setAdapter(
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, labels));
        suppressSpinnerCallbacks = true;
        binding.languageSpinner.setSelection(Math.max(LANGUAGES.indexOf(selectedLanguage), 0));
        suppressSpinnerCallbacks = false;
        binding.languageSpinner.setOnItemSelectedListener(
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(
                            AdapterView<?> parent, View view, int position, long id) {
                        if (suppressSpinnerCallbacks || position >= LANGUAGES.size()) {
                            return;
                        }
                        Language language = LANGUAGES.get(position);
                        if (language.id.equals(selectedLanguage.id)) {
                            return;
                        }
                        selectedLanguage = language;
                        selectedVoice = null;
                        refreshVoices();
                        repopulateVoiceSpinner();
                        loadSynthesizer(selectedVoice == null ? null : selectedVoice.id, false);
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {}
                });
    }

    private void setupVoiceSpinner() {
        binding.voiceSpinner.setOnItemSelectedListener(
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(
                            AdapterView<?> parent, View view, int position, long id) {
                        if (suppressSpinnerCallbacks || position >= availableVoices.size()) {
                            return;
                        }
                        Voice voice = availableVoices.get(position);
                        if (selectedVoice != null
                                && voice.id.equals(selectedVoice.id)
                                && !isCloned) {
                            return;
                        }
                        selectedVoice = voice;
                        // Picking a preset is how you get your own voice back off the engine.
                        isCloned = false;
                        setCloneStatus(null);
                        loadSynthesizer(voice.id, false);
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {}
                });
    }

    private void repopulateVoiceSpinner() {
        List<String> labels = new ArrayList<>();
        for (Voice voice : availableVoices) {
            labels.add(voice.displayName);
        }
        binding.voiceSpinner.setAdapter(
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, labels));
        suppressSpinnerCallbacks = true;
        if (!availableVoices.isEmpty()) {
            int index = 0;
            for (int i = 0; i < availableVoices.size(); i++) {
                if (selectedVoice != null && availableVoices.get(i).id.equals(selectedVoice.id)) {
                    index = i;
                    break;
                }
            }
            binding.voiceSpinner.setSelection(index);
        }
        suppressSpinnerCallbacks = false;
    }

    // -- Loading -------------------------------------------------------------

    /**
     * Builds a synthesizer for the current language and {@code voiceId}, downloading whatever it
     * needs. {@code load()} blocks, so it runs on the worker; the {@code 0..1} progress fraction it
     * reports comes back here and drives the progress bar.
     *
     * <p>Pass {@code cloning} to build one that can take its voice from a recording. That one
     * ignores {@code voiceId}, because a cloned voice comes from the clip rather than from the
     * catalogue, and it cannot speak until {@link TextToSpeech#cloneFrom(VoiceClone)} has run.
     */
    private void loadSynthesizer(@Nullable String voiceId, boolean cloning) {
        if (isLoading) {
            return;
        }
        isLoading = true;
        engineReady = false;
        hideError();
        setProgressVisible(true);
        binding.loadingLabel.setText(cloning ? R.string.fetching_cloning : R.string.initializing);
        updateUiState();

        TextToSpeech synthesizer =
                new TextToSpeech(this)
                        .language(selectedLanguage.id)
                        .onProgress(
                                (fraction, file) ->
                                        runOnUiThread(() -> showProgress(fraction, file)));
        if (cloning) {
            // Create-time mode: ZipVoice + clone ASR download with load().
            synthesizer.cloning(true);
        } else if (voiceId != null) {
            synthesizer.voice(voiceId);
        }

        worker.execute(
                () -> {
                    try {
                        synthesizer.load();
                        runOnUiThread(
                                () -> {
                                    isLoading = false;
                                    setProgressVisible(false);
                                    if (tts != null) {
                                        tts.close();
                                    }
                                    tts = synthesizer;
                                    hasCloningEngine = cloning;
                                    // A cloning engine has no voice until a clip
                                    // has been recorded into it, so it is not ready
                                    // to speak yet.
                                    engineReady = !cloning;
                                    refreshVoices();
                                    repopulateVoiceSpinner();
                                    updateUiState();
                                    if (!cloning && !spokenWelcome) {
                                        spokenWelcome = true;
                                        speakUtterance("Welcome to Moonshine Text to Speech");
                                    }
                                });
                    } catch (Exception e) {
                        synthesizer.close();
                        runOnUiThread(
                                () -> {
                                    isLoading = false;
                                    setProgressVisible(false);
                                    showError("Failed to load voice: " + e.getMessage());
                                    updateUiState();
                                });
                    }
                });
    }

    /**
     * Lists the voices for the current language, marking the ones already on disk. All voices of a
     * language share one cache directory, so pointing {@code g2p_root} at it is enough.
     */
    private void refreshVoices() {
        File root = ModelCache.directoryFor(this, ModelSpec.tts(selectedLanguage.id, null), null);
        try {
            String json =
                    TextToSpeech.getTtsVoices(
                            selectedLanguage.id,
                            Collections.singletonList(
                                    new TranscriberOption("g2p_root", root.getAbsolutePath())));
            availableVoices = parseVoices(json, selectedLanguage);
        } catch (Exception e) {
            showError(
                    "Failed to list voices for "
                            + selectedLanguage.displayName
                            + ": "
                            + e.getMessage());
            availableVoices = new ArrayList<>();
        }
        String currentId = selectedVoice == null ? null : selectedVoice.id;
        boolean stillThere = false;
        for (Voice voice : availableVoices) {
            if (voice.id.equals(currentId)) {
                stillThere = true;
                break;
            }
        }
        if (!stillThere) {
            selectedVoice = null;
            for (Voice voice : availableVoices) {
                if (!voice.needsDownload) {
                    selectedVoice = voice;
                    break;
                }
            }
            if (selectedVoice == null && !availableVoices.isEmpty()) {
                selectedVoice = availableVoices.get(0);
            }
        }
    }

    private List<Voice> parseVoices(String json, Language language) {
        List<Voice> out = new ArrayList<>();
        JSONArray langVoices;
        try {
            langVoices = new JSONObject(json).optJSONArray(language.id);
        } catch (Exception e) {
            return out;
        }
        if (langVoices == null) {
            return out;
        }
        for (int i = 0; i < langVoices.length(); i++) {
            JSONObject entry = langVoices.optJSONObject(i);
            if (entry == null) {
                continue;
            }
            String voiceId = entry.optString("id", "");
            String state = entry.optString("state", "");
            boolean needsDownload;
            if ("found".equals(state)) {
                needsDownload = false;
            } else if ("missing".equals(state)) {
                needsDownload = true;
            } else {
                continue;
            }
            String base;
            if (voiceId.startsWith("kokoro_")) {
                base = formatKokoroName(voiceId.substring("kokoro_".length()));
            } else if (voiceId.startsWith("piper_")) {
                base = formatPiperName(voiceId.substring("piper_".length()));
            } else {
                continue;
            }
            String display =
                    needsDownload
                            ? base + " " + getString(R.string.voice_suffix_downloadable)
                            : base;
            out.add(new Voice(voiceId, display, needsDownload));
        }
        // Group Kokoro entries first, then Piper; within each group, already-downloaded
        // voices appear before downloadable ones so users see what they can play
        // immediately.
        Collections.sort(
                out,
                new Comparator<Voice>() {
                    @Override
                    public int compare(Voice a, Voice b) {
                        int engineA = a.id.startsWith("kokoro_") ? 0 : 1;
                        int engineB = b.id.startsWith("kokoro_") ? 0 : 1;
                        if (engineA != engineB) {
                            return Integer.compare(engineA, engineB);
                        }
                        return Integer.compare(
                                a.needsDownload ? 1 : 0, b.needsDownload ? 1 : 0);
                    }
                });
        return out;
    }

    /** {@code shortId} is like {@code af_heart} (same scheme as the iOS sample). */
    private String formatKokoroName(String shortId) {
        String[] parts = shortId.split("_", 2);
        if (parts.length < 2) {
            return shortId + " (Kokoro)";
        }
        String name = capitalize(parts[1]);
        String gender = "";
        if (parts[0].endsWith("f")) {
            gender = "Female";
        } else if (parts[0].endsWith("m")) {
            gender = "Male";
        }
        return gender.isEmpty() ? name + " (Kokoro)" : name + " (" + gender + ", Kokoro)";
    }

    /** {@code shortId} is a Piper stem like {@code en_US-saikat} or {@code de_DE-thorsten-medium}. */
    private String formatPiperName(String shortId) {
        int dash = shortId.indexOf('-');
        String afterLocale = dash < 0 ? shortId : shortId.substring(dash + 1);
        if (afterLocale.isEmpty()) {
            afterLocale = shortId;
        }
        StringBuilder pretty = new StringBuilder();
        for (String segment : afterLocale.replace('_', ' ').split("[- ]")) {
            if (segment.isEmpty()) {
                continue;
            }
            if (pretty.length() > 0) {
                pretty.append(' ');
            }
            pretty.append(capitalize(segment));
        }
        return (pretty.length() == 0 ? shortId : pretty.toString()) + " (Piper)";
    }

    private static String capitalize(String value) {
        if (value.isEmpty()) {
            return value;
        }
        return value.substring(0, 1).toUpperCase(Locale.ROOT) + value.substring(1);
    }

    // -- Speaking ------------------------------------------------------------

    private void speakCurrentText() {
        CharSequence entered = binding.inputText.getText();
        String text = entered == null ? "" : entered.toString().trim();
        speakUtterance(text.isEmpty() ? "Hello world" : text);
    }

    /** {@code say} synthesizes and plays, returning once the audio has finished. */
    private void speakUtterance(String text) {
        TextToSpeech synthesizer = tts;
        if (synthesizer == null) {
            return;
        }
        isSpeaking = true;
        updateUiState();
        worker.execute(
                () -> {
                    String errorMessage = null;
                    try {
                        synthesizer.say(text);
                    } catch (Exception e) {
                        errorMessage = "Speech failed: " + e.getMessage();
                    }
                    String message = errorMessage;
                    runOnUiThread(
                            () -> {
                                isSpeaking = false;
                                if (message != null) {
                                    showError(message);
                                }
                                updateUiState();
                            });
                });
    }

    // -- Voice cloning -------------------------------------------------------

    /**
     * Records a few seconds from the microphone and rebuilds the voice from it.
     *
     * <p>Everything here is the library's: it asks for the recording permission, finds the speech
     * in the recording, transcribes the clip to condition the model, and swaps the voice on the
     * synthesizer. The app supplies a button and a status line.
     */
    private void cloneFromMicrophone() {
        if (isCloning || isLoading || tts == null) {
            return;
        }
        isCloning = true;
        hideError();
        setCloneStatus(getString(R.string.clone_asking));
        updateUiState();

        worker.execute(
                () -> {
                    try {
                        // Cloning only exists on ZipVoice, so switch engines behind
                        // the scenes rather than making that the reader's problem.
                        if (!hasCloningEngine) {
                            loadCloningEngineBlocking();
                        }
                        TextToSpeech synthesizer = tts;
                        if (synthesizer == null) {
                            throw new IllegalStateException("The cloning engine failed to load.");
                        }

                        VoiceClone clone = synthesizer.startCloning();
                        clone.onProgress(
                                (recorded, speech) ->
                                        runOnUiThread(
                                                () ->
                                                        setCloneStatus(
                                                                String.format(
                                                                        Locale.US,
                                                                        "Listening… %.1fs recorded,"
                                                                            + " %.1fs of speech",
                                                                        recorded,
                                                                        speech))));
                        clone.onReady(
                                () ->
                                        runOnUiThread(
                                                () ->
                                                        setCloneStatus(
                                                                getString(R.string.clone_enough))));

                        clone.fromMicrophone();
                        runOnUiThread(() -> setCloneStatus(getString(R.string.clone_trimming)));

                        synthesizer.cloneFrom(clone);
                        float[] audio = clone.getAudio();
                        float seconds =
                                audio == null
                                        ? 0f
                                        : (float) audio.length / Math.max(clone.getSampleRate(), 1);
                        runOnUiThread(
                                () -> {
                                    isCloning = false;
                                    isCloned = true;
                                    engineReady = true;
                                    setCloneStatus(
                                            String.format(
                                                    Locale.US,
                                                    "Cloned from %.1fs of your speech. Type"
                                                        + " something and press Speak.",
                                                    seconds));
                                    updateUiState();
                                });
                    } catch (Exception e) {
                        runOnUiThread(
                                () -> {
                                    isCloning = false;
                                    setCloneStatus(null);
                                    showError("Cloning failed: " + e.getMessage());
                                    updateUiState();
                                    // A half-built cloning engine cannot speak, so
                                    // put a working preset voice back.
                                    if (hasCloningEngine && !isCloned) {
                                        loadSynthesizer(
                                                selectedVoice == null ? null : selectedVoice.id,
                                                false);
                                    }
                                });
                    }
                });
    }

    /**
     * Replaces {@link #tts} with one that can clone. Runs on the worker thread, inside the cloning
     * task, so it blocks rather than posting back like {@link #loadSynthesizer}.
     */
    private void loadCloningEngineBlocking() {
        runOnUiThread(
                () -> {
                    setProgressVisible(true);
                    binding.loadingLabel.setText(R.string.fetching_cloning);
                });
        TextToSpeech synthesizer =
                new TextToSpeech(this)
                        .language(selectedLanguage.id)
                        .cloning(true)
                        .onProgress(
                                (fraction, file) ->
                                        runOnUiThread(() -> showProgress(fraction, file)));
        try {
            synthesizer.load();
        } catch (RuntimeException e) {
            synthesizer.close();
            runOnUiThread(() -> setProgressVisible(false));
            throw e;
        }
        TextToSpeech previous = tts;
        tts = synthesizer;
        hasCloningEngine = true;
        if (previous != null) {
            previous.close();
        }
        runOnUiThread(
                () -> {
                    setProgressVisible(false);
                    // Not ready to speak until a clip has been recorded into it.
                    engineReady = false;
                    updateUiState();
                });
    }

    // -- UI ------------------------------------------------------------------

    private void updateUiState() {
        boolean busy = !engineReady || isSpeaking || isLoading || isCloning;
        binding.languageSpinner.setEnabled(!busy);
        binding.voiceSpinner.setEnabled(!busy && !availableVoices.isEmpty());
        binding.inputText.setEnabled(!busy);

        // When the input is empty, pressing Speak says "Hello world" (see
        // speakCurrentText), so we only need a usable engine to enable the button.
        binding.speakButton.setEnabled(engineReady && !isSpeaking && !isLoading && !isCloning);
        binding.speakButton.setText(isSpeaking ? R.string.speaking : R.string.speak);

        binding.recordButton.setEnabled(!isCloning && !isLoading && tts != null);
        binding.recordButton.setText(
                isCloning
                        ? R.string.recording
                        : (isCloned ? R.string.record_again : R.string.record));
    }

    private void setCloneStatus(@Nullable String message) {
        if (message == null) {
            binding.cloneStatus.setVisibility(View.GONE);
            return;
        }
        binding.cloneStatus.setText(message);
        binding.cloneStatus.setVisibility(View.VISIBLE);
    }

    private void showProgress(float fraction, String file) {
        binding.loadingLabel.setText(getString(R.string.downloading_asset, displayNameForKey(file)));
        binding.downloadProgress.setVisibility(View.VISIBLE);
        binding.downloadProgress.setIndeterminate(false);
        binding.downloadProgress.setProgress(Math.max(0, Math.min(100, (int) (fraction * 100))));
    }

    /** Keep the progress label short: show just the filename, not the full asset key. */
    private String displayNameForKey(String key) {
        int slash = key.lastIndexOf('/');
        return slash < 0 ? key : key.substring(slash + 1);
    }

    private void setProgressVisible(boolean visible) {
        binding.loadingIndicator.setVisibility(visible ? View.VISIBLE : View.GONE);
        binding.loadingLabel.setVisibility(visible ? View.VISIBLE : View.GONE);
        if (!visible) {
            binding.downloadProgress.setVisibility(View.GONE);
        }
    }

    private void showError(String message) {
        binding.errorText.setText(message);
        binding.errorText.setVisibility(View.VISIBLE);
    }

    private void hideError() {
        binding.errorText.setVisibility(View.GONE);
    }
}
