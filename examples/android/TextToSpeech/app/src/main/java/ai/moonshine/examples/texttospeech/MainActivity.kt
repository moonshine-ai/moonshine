package ai.moonshine.examples.texttospeech

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import ai.moonshine.examples.texttospeech.databinding.ActivityMainBinding
import ai.moonshine.voice.TextToSpeech
import ai.moonshine.voice.TranscriberOption
import java.io.File
import kotlin.concurrent.thread
import org.json.JSONArray
import org.json.JSONObject

/**
 * Kokoro-supported languages with display names (aligned with the iOS TextToSpeech sample).
 */
private data class KokoroLanguage(
    val id: String,
    val displayName: String,
)

private data class KokoroVoice(
    val id: String,
    val displayName: String,
)

private val kokoroLanguages: List<KokoroLanguage> =
    listOf(
        KokoroLanguage("en_us", "English (US)"),
        KokoroLanguage("en_gb", "English (UK)"),
        KokoroLanguage("es_mx", "Spanish"),
        KokoroLanguage("fr", "French"),
        KokoroLanguage("hi", "Hindi"),
        KokoroLanguage("it", "Italian"),
        KokoroLanguage("ja", "Japanese"),
        KokoroLanguage("pt_br", "Portuguese (BR)"),
        KokoroLanguage("zh_hans", "Chinese (Mandarin)"),
    )

class MainActivity : AppCompatActivity() {

    /** Bundled under `app/src/main/assets/` (Git LFS). Same tree as the iOS `tts-data` folder. */
    private val ttsAssetDir = "tts-data"

    private lateinit var binding: ActivityMainBinding
    private val mainHandler = Handler(Looper.getMainLooper())

    private var g2pRoot: String = ""
    private var tts: TextToSpeech? = null

    private var selectedLanguage: KokoroLanguage = kokoroLanguages[0]
    private var availableVoices: List<KokoroVoice> = emptyList()
    private var selectedVoice: KokoroVoice? = null

    private var engineReady = false
    private var isSpeaking = false

    private var suppressSpinnerCallbacks = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        setupLanguageSpinner()
        setupVoiceSpinner()

        binding.speakButton.setOnClickListener { speakCurrentText() }

        bootstrapEngine()
    }

    override fun onDestroy() {
        tts?.close()
        tts = null
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
                    onLanguageChanged()
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
                    recreateSynthesizer(voice.id)
                    updateUiState()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
    }

    private fun bootstrapEngine() {
        if (!hasTtsAssetsInApk()) {
            showError(
                "TTS assets not found under assets/$ttsAssetDir/. " +
                    "Clone this repository with Git LFS so Kokoro weights under " +
                    "app/src/main/assets/tts-data/ are present.",
            )
            return
        }

        binding.loadingIndicator.visibility = View.VISIBLE
        binding.loadingLabel.visibility = View.VISIBLE
        hideError()

        thread {
            try {
                val destRoot = File(filesDir, ttsAssetDir)
                AssetDirectoryCopy.copyDirIfNeeded(
                    this,
                    ttsAssetDir,
                    destRoot,
                    "kokoro/config.json",
                )
                g2pRoot = destRoot.absolutePath

                mainHandler.post {
                    try {
                        refreshVoices()
                        recreateSynthesizer(selectedVoice?.id)
                        repopulateVoiceSpinner()
                        engineReady = true
                        hideError()
                    } catch (e: Exception) {
                        showError("Failed to initialize TTS: ${e.message}")
                    } finally {
                        binding.loadingIndicator.visibility = View.GONE
                        binding.loadingLabel.visibility = View.GONE
                        updateUiState()
                        if (engineReady) {
                            speakWelcome()
                        }
                    }
                }
            } catch (e: Exception) {
                mainHandler.post {
                    binding.loadingIndicator.visibility = View.GONE
                    binding.loadingLabel.visibility = View.GONE
                    showError("Failed to copy assets: ${e.message}")
                    updateUiState()
                }
            }
        }
    }

    private fun hasTtsAssetsInApk(): Boolean {
        val kokoro = assets.list("$ttsAssetDir/kokoro") ?: return false
        return kokoro.contains("config.json") && kokoro.contains("model.onnx")
    }

    private fun onLanguageChanged() {
        if (g2pRoot.isEmpty()) return
        try {
            refreshVoices()
            recreateSynthesizer(selectedVoice?.id)
            repopulateVoiceSpinner()
            updateUiState()
        } catch (e: Exception) {
            showError("Failed to switch language: ${e.message}")
            updateUiState()
        }
    }

    private fun refreshVoices() {
        val json =
            TextToSpeech.getTtsVoices(
                selectedLanguage.id,
                listOf(TranscriberOption("g2p_root", g2pRoot)),
            )
        availableVoices = parseVoices(json, selectedLanguage)
        if (availableVoices.isNotEmpty()) {
            val currentId = selectedVoice?.id
            if (currentId == null || availableVoices.none { it.id == currentId }) {
                selectedVoice = availableVoices[0]
            }
        } else {
            selectedVoice = null
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
            if (state != "found" || !voiceId.startsWith("kokoro_")) continue
            val shortId = voiceId.removePrefix("kokoro_")
            out.add(KokoroVoice(id = voiceId, displayName = formatVoiceName(shortId)))
        }
        return out
    }

    /** [shortId] is like `af_heart` (same as iOS). */
    private fun formatVoiceName(shortId: String): String {
        val parts = shortId.split("_", limit = 2)
        if (parts.size < 2) return shortId
        val prefix = parts[0]
        val name = parts[1].replaceFirstChar { it.uppercaseChar() }
        val gender =
            when {
                prefix.endsWith("f") -> "Female"
                prefix.endsWith("m") -> "Male"
                else -> ""
            }
        return if (gender.isEmpty()) name else "$name ($gender)"
    }

    private fun recreateSynthesizer(voiceId: String?) {
        tts?.close()
        tts = null
        val opts = ArrayList<TranscriberOption>()
        if (voiceId != null) {
            opts.add(TranscriberOption("voice", voiceId))
        }
        tts =
            TextToSpeech(
                selectedLanguage.id,
                g2pRoot,
                opts,
            )
    }

    private fun repopulateVoiceSpinner() {
        val labels = availableVoices.map { it.displayName }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        binding.voiceSpinner.adapter = adapter
        suppressSpinnerCallbacks = true
        val idx = availableVoices.indexOfFirst { it.id == selectedVoice?.id }.let { if (it >= 0) it else 0 }
        if (availableVoices.isNotEmpty()) {
            binding.voiceSpinner.setSelection(idx.coerceIn(0, availableVoices.lastIndex))
        }
        suppressSpinnerCallbacks = false
    }

    private fun updateUiState() {
        binding.languageSpinner.isEnabled = engineReady && !isSpeaking
        binding.voiceSpinner.isEnabled = engineReady && !isSpeaking && availableVoices.isNotEmpty()
        binding.inputText.isEnabled = engineReady && !isSpeaking

        val canSpeak =
            engineReady &&
                !isSpeaking &&
                binding.inputText.text?.toString()?.trim()?.isNotEmpty() == true
        binding.speakButton.isEnabled = canSpeak
        binding.speakButton.text =
            if (isSpeaking) {
                getString(R.string.speaking)
            } else {
                getString(R.string.speak)
            }
    }

    private fun speakWelcome() {
        speakUtterance("Welcome to Moonshine Text to Speech")
    }

    private fun speakCurrentText() {
        val text = binding.inputText.text?.toString()?.trim().orEmpty()
        if (text.isEmpty()) return
        speakUtterance(text)
    }

    private fun speakUtterance(text: String) {
        val t = tts ?: return
        isSpeaking = true
        updateUiState()
        thread {
            try {
                t.say(this@MainActivity, text)
                t.waitUntilDone()
            } finally {
                mainHandler.post {
                    isSpeaking = false
                    updateUiState()
                }
            }
        }
    }

    private fun showError(message: String) {
        engineReady = false
        binding.errorText.text = message
        binding.errorText.visibility = View.VISIBLE
    }

    private fun hideError() {
        binding.errorText.visibility = View.GONE
    }
}
