package ai.moonshine.examples.texttospeech

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import ai.moonshine.examples.texttospeech.databinding.ActivityMainBinding
import ai.moonshine.voice.ModelCache
import ai.moonshine.voice.ModelSpec
import ai.moonshine.voice.TextToSpeech
import ai.moonshine.voice.TranscriberOption
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject

/**
 * Moonshine TTS languages exposed in the sample, covering both Kokoro and Piper engines.
 * Display names follow the iOS TextToSpeech sample where applicable.
 */
private data class KokoroLanguage(
    val id: String,
    val displayName: String,
)

private data class KokoroVoice(
    val id: String,
    val displayName: String,
    /** `true` if the voice's asset files are not yet on disk and must be downloaded. */
    val needsDownload: Boolean,
)

private val kokoroLanguages: List<KokoroLanguage> =
    listOf(
        KokoroLanguage("ar_msa", "Arabic (MSA)"),
        KokoroLanguage("de", "German"),
        KokoroLanguage("en_us", "English (US)"),
        KokoroLanguage("en_gb", "English (UK)"),
        KokoroLanguage("es_ar", "Spanish (AR)"),
        KokoroLanguage("es_es", "Spanish (ES)"),
        KokoroLanguage("es_mx", "Spanish (MX)"),
        KokoroLanguage("fr", "French"),
        KokoroLanguage("hi", "Hindi"),
        KokoroLanguage("it", "Italian"),
        KokoroLanguage("ja", "Japanese"),
        KokoroLanguage("ko", "Korean"),
        KokoroLanguage("nl", "Dutch"),
        KokoroLanguage("pt_br", "Portuguese (BR)"),
        KokoroLanguage("pt_pt", "Portuguese (PT)"),
        KokoroLanguage("ru", "Russian"),
        KokoroLanguage("tr", "Turkish"),
        KokoroLanguage("uk", "Ukrainian"),
        KokoroLanguage("vi", "Vietnamese"),
        KokoroLanguage("zh_hans", "Chinese (Mandarin)"),
    )

class MainActivity : AppCompatActivity() {

    /**
     * Nothing is bundled in the APK. The synthesizer downloads the Kokoro base model, the
     * language's G2P assets, and the selected voice on first use, into a managed per-language
     * cache directory ([ModelCache]) that it reuses thereafter.
     */
    private lateinit var binding: ActivityMainBinding

    /** Moonshine's blocking calls (load, say) run here. */
    private val worker = Executors.newSingleThreadExecutor()

    private var tts: TextToSpeech? = null

    private var selectedLanguage: KokoroLanguage =
        kokoroLanguages.firstOrNull { it.id == "en_us" } ?: kokoroLanguages[0]
    private var availableVoices: List<KokoroVoice> = emptyList()
    private var selectedVoice: KokoroVoice? = null

    private var engineReady = false
    private var isSpeaking = false
    private var isLoading = false
    private var spokenWelcome = false

    private var suppressSpinnerCallbacks = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        setupLanguageSpinner()
        setupVoiceSpinner()

        binding.speakButton.setOnClickListener { speakCurrentText() }

        binding.inputText.addTextChangedListener(
            object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: Editable?) = updateUiState()
            },
        )

        refreshVoices()
        repopulateVoiceSpinner()
        loadSynthesizer("kokoro_af_alloy")
    }

    override fun onDestroy() {
        tts?.close()
        tts = null
        worker.shutdown()
        super.onDestroy()
    }

    private fun setupLanguageSpinner() {
        val labels = kokoroLanguages.map { it.displayName }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        binding.languageSpinner.adapter = adapter
        suppressSpinnerCallbacks = true
        binding.languageSpinner.setSelection(kokoroLanguages.indexOf(selectedLanguage).coerceAtLeast(0))
        suppressSpinnerCallbacks = false
        binding.languageSpinner.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long,
                ) {
                    if (suppressSpinnerCallbacks) return
                    val lang = kokoroLanguages.getOrNull(position) ?: return
                    if (lang.id == selectedLanguage.id) return
                    selectedLanguage = lang
                    selectedVoice = null
                    refreshVoices()
                    repopulateVoiceSpinner()
                    loadSynthesizer(selectedVoice?.id)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
    }

    private fun setupVoiceSpinner() {
        binding.voiceSpinner.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long,
                ) {
                    if (suppressSpinnerCallbacks) return
                    val voice = availableVoices.getOrNull(position) ?: return
                    if (voice.id == selectedVoice?.id) return
                    selectedVoice = voice
                    loadSynthesizer(voice.id)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
    }

    /**
     * Builds a synthesizer for the current language and `voiceId`, downloading whatever it needs.
     * `load()` blocks, so it runs on the worker; the `0..1` progress fraction it reports comes
     * back here and drives the progress bar.
     */
    private fun loadSynthesizer(voiceId: String?) {
        if (isLoading) return
        isLoading = true
        engineReady = false
        hideError()
        setProgressVisible(true)
        binding.loadingLabel.setText(R.string.initializing)
        updateUiState()

        val language = selectedLanguage
        val synthesizer = TextToSpeech(this)
            .language(language.id)
            .onProgress { fraction, file -> runOnUiThread { showProgress(fraction, file) } }
        if (voiceId != null) {
            synthesizer.voice(voiceId)
        }

        worker.execute {
            try {
                synthesizer.load()
                runOnUiThread {
                    isLoading = false
                    setProgressVisible(false)
                    tts?.close()
                    tts = synthesizer
                    engineReady = true
                    refreshVoices()
                    repopulateVoiceSpinner()
                    updateUiState()
                    if (!spokenWelcome) {
                        spokenWelcome = true
                        speakUtterance("Welcome to Moonshine Text to Speech")
                    }
                }
            } catch (e: Exception) {
                synthesizer.close()
                runOnUiThread {
                    isLoading = false
                    setProgressVisible(false)
                    showError("Failed to load voice: ${e.message}")
                    updateUiState()
                }
            }
        }
    }

    /**
     * Lists the voices for the current language, marking the ones already on disk. All voices of
     * a language share one cache directory, so pointing `g2p_root` at it is enough.
     */
    private fun refreshVoices() {
        val root = ModelCache.directoryFor(this, ModelSpec.tts(selectedLanguage.id, null), null)
        availableVoices = try {
            val json = TextToSpeech.getTtsVoices(
                selectedLanguage.id,
                listOf(TranscriberOption("g2p_root", root.absolutePath)),
            )
            parseVoices(json, selectedLanguage)
        } catch (e: Exception) {
            showError("Failed to list voices for ${selectedLanguage.displayName}: ${e.message}")
            emptyList()
        }
        val currentId = selectedVoice?.id
        if (currentId == null || availableVoices.none { it.id == currentId }) {
            selectedVoice = availableVoices.firstOrNull { !it.needsDownload }
                ?: availableVoices.firstOrNull()
        }
    }

    private fun parseVoices(json: String, language: KokoroLanguage): List<KokoroVoice> {
        val root = JSONObject(json)
        val langVoices: JSONArray = root.optJSONArray(language.id) ?: return emptyList()
        val out = ArrayList<KokoroVoice>()
        for (i in 0 until langVoices.length()) {
            val entry = langVoices.optJSONObject(i) ?: continue
            val voiceId = entry.optString("id", "")
            val state = entry.optString("state", "")
            val needsDownload =
                when (state) {
                    "found" -> false
                    "missing" -> true
                    else -> continue
                }
            val base =
                when {
                    voiceId.startsWith("kokoro_") ->
                        formatKokoroName(voiceId.removePrefix("kokoro_"))
                    voiceId.startsWith("piper_") ->
                        formatPiperName(voiceId.removePrefix("piper_"))
                    else -> continue
                }
            val display =
                if (needsDownload) {
                    "$base ${getString(R.string.voice_suffix_downloadable)}"
                } else {
                    base
                }
            out.add(KokoroVoice(id = voiceId, displayName = display, needsDownload = needsDownload))
        }
        // Group Kokoro entries first, then Piper; within each group, already-downloaded voices
        // appear before downloadable ones so users see what they can play immediately.
        return out.sortedWith(
            compareBy(
                { if (it.id.startsWith("kokoro_")) 0 else 1 },
                { if (it.needsDownload) 1 else 0 },
            ),
        )
    }

    /** [shortId] is like `af_heart` (same scheme as the iOS sample). */
    private fun formatKokoroName(shortId: String): String {
        val parts = shortId.split("_", limit = 2)
        if (parts.size < 2) return "$shortId (Kokoro)"
        val prefix = parts[0]
        val name = parts[1].replaceFirstChar { it.uppercaseChar() }
        val gender =
            when {
                prefix.endsWith("f") -> "Female"
                prefix.endsWith("m") -> "Male"
                else -> ""
            }
        return if (gender.isEmpty()) "$name (Kokoro)" else "$name ($gender, Kokoro)"
    }

    /** [shortId] is a Piper stem like `en_US-saikat` or `de_DE-thorsten-medium`. */
    private fun formatPiperName(shortId: String): String {
        val afterLocale = shortId.substringAfter('-', "").ifEmpty { shortId }
        val pretty =
            afterLocale
                .split('-')
                .filter { it.isNotEmpty() }
                .joinToString(" ") { segment ->
                    segment
                        .replace('_', ' ')
                        .split(' ')
                        .filter { it.isNotEmpty() }
                        .joinToString(" ") { it.replaceFirstChar { c -> c.uppercaseChar() } }
                }
                .ifEmpty { shortId }
        return "$pretty (Piper)"
    }

    private fun repopulateVoiceSpinner() {
        val labels = availableVoices.map { it.displayName }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        binding.voiceSpinner.adapter = adapter
        suppressSpinnerCallbacks = true
        val idx = availableVoices.indexOfFirst { it.id == selectedVoice?.id }
            .let { if (it >= 0) it else 0 }
        if (availableVoices.isNotEmpty()) {
            binding.voiceSpinner.setSelection(idx.coerceIn(0, availableVoices.lastIndex))
        }
        suppressSpinnerCallbacks = false
    }

    private fun updateUiState() {
        val busy = !engineReady || isSpeaking || isLoading
        binding.languageSpinner.isEnabled = !busy
        binding.voiceSpinner.isEnabled = !busy && availableVoices.isNotEmpty()
        binding.inputText.isEnabled = !busy

        // When the input is empty, pressing Speak says "Hello world" (see speakCurrentText),
        // so we only need a usable engine to enable the button.
        binding.speakButton.isEnabled = engineReady && !isSpeaking && !isLoading
        binding.speakButton.text =
            if (isSpeaking) {
                getString(R.string.speaking)
            } else {
                getString(R.string.speak)
            }
    }

    private fun speakCurrentText() {
        val text = binding.inputText.text?.toString()?.trim().orEmpty()
        speakUtterance(text.ifEmpty { "Hello world" })
    }

    /** `say` synthesizes and plays, returning once the audio has finished. */
    private fun speakUtterance(text: String) {
        val synthesizer = tts ?: return
        isSpeaking = true
        updateUiState()
        worker.execute {
            var errorMessage: String? = null
            try {
                synthesizer.say(text)
            } catch (e: Exception) {
                errorMessage = "Speech failed: ${e.message}"
            }
            val message = errorMessage
            runOnUiThread {
                isSpeaking = false
                if (message != null) {
                    showError(message)
                }
                updateUiState()
            }
        }
    }

    private fun showProgress(fraction: Float, file: String) {
        binding.loadingLabel.text =
            getString(R.string.downloading_asset, displayNameForKey(file))
        val indicator = binding.downloadProgress
        indicator.visibility = View.VISIBLE
        indicator.isIndeterminate = false
        indicator.progress = (fraction * 100).toInt().coerceIn(0, 100)
    }

    /** Keep the progress label short: show just the filename, not the full asset key. */
    private fun displayNameForKey(key: String): String {
        val slash = key.lastIndexOf('/')
        return if (slash < 0) key else key.substring(slash + 1)
    }

    private fun setProgressVisible(visible: Boolean) {
        binding.loadingIndicator.visibility = if (visible) View.VISIBLE else View.GONE
        binding.loadingLabel.visibility = if (visible) View.VISIBLE else View.GONE
        if (!visible) {
            binding.downloadProgress.visibility = View.GONE
        }
    }

    private fun showError(message: String) {
        binding.errorText.text = message
        binding.errorText.visibility = View.VISIBLE
    }

    private fun hideError() {
        binding.errorText.visibility = View.GONE
    }
}
