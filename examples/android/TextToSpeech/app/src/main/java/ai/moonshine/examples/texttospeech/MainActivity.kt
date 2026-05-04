package ai.moonshine.examples.texttospeech

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
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
     * Only the Kokoro model (`kokoro/model.onnx` + `kokoro/config.json`) and the default voice
     * (`kokoro/voices/af_alloy.kokorovoice`) are bundled in the APK. Everything else (G2P assets,
     * other Kokoro voices, all Piper voices) is fetched from `download.moonshine.ai/tts/` the
     * first time a voice that needs it is selected.
     */
    private val ttsAssetDir = "tts-data"

    private lateinit var binding: ActivityMainBinding
    private val mainHandler = Handler(Looper.getMainLooper())

    private var g2pRoot: String = ""
    private var tts: TextToSpeech? = null

    private var selectedLanguage: KokoroLanguage =
        kokoroLanguages.firstOrNull { it.id == "en_us" } ?: kokoroLanguages[0]
    private var availableVoices: List<KokoroVoice> = emptyList()
    private var selectedVoice: KokoroVoice? = null

    private var engineReady = false
    private var isSpeaking = false
    private var isDownloading = false
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
                    ensureAssetsThenRecreate(selectedLanguage, voice)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
    }

    private fun bootstrapEngine() {
        if (!hasBundledKokoroAssets()) {
            showError(
                "Bundled Kokoro model not found under assets/$ttsAssetDir/kokoro/. " +
                    "Clone this repository with Git LFS so the model and the Alloy voice are present.",
            )
            return
        }

        setProgressVisible(true)
        binding.loadingLabel.setText(R.string.initializing)
        binding.downloadProgress.visibility = View.GONE
        hideError()

        val initialLang = selectedLanguage
        val initialVoice = "kokoro_af_alloy"

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

                TtsAssetDownloader.ensureAssetsPresent(
                    destRoot,
                    initialLang.id,
                    initialVoice,
                    ::postDownloadProgress,
                )

                mainHandler.post {
                    try {
                        refreshVoices()
                        selectedVoice =
                            availableVoices.firstOrNull { it.id == initialVoice }
                                ?: availableVoices.firstOrNull { !it.needsDownload }
                                ?: availableVoices.firstOrNull()
                        recreateSynthesizer(selectedVoice?.id)
                        repopulateVoiceSpinner()
                        engineReady = true
                        hideError()
                    } catch (e: Exception) {
                        showError("Failed to initialize TTS: ${e.message}")
                    } finally {
                        setProgressVisible(false)
                        updateUiState()
                        if (engineReady && !spokenWelcome) {
                            spokenWelcome = true
                            speakWelcome()
                        }
                    }
                }
            } catch (e: Exception) {
                mainHandler.post {
                    setProgressVisible(false)
                    showError("Failed to prepare assets: ${e.message}")
                    updateUiState()
                }
            }
        }
    }

    private fun hasBundledKokoroAssets(): Boolean {
        val kokoro = assets.list("$ttsAssetDir/kokoro") ?: return false
        return kokoro.contains("config.json") && kokoro.contains("model.onnx")
    }

    private fun onLanguageChanged() {
        if (g2pRoot.isEmpty()) return
        try {
            refreshVoices()
        } catch (e: Exception) {
            showError("Failed to list voices for ${selectedLanguage.displayName}: ${e.message}")
            updateUiState()
            return
        }
        val preferred =
            availableVoices.firstOrNull { !it.needsDownload }
                ?: availableVoices.firstOrNull()
        selectedVoice = preferred
        repopulateVoiceSpinner()
        if (preferred == null) {
            recreateSynthesizer(null)
            updateUiState()
        } else {
            ensureAssetsThenRecreate(selectedLanguage, preferred)
        }
    }

    /**
     * Download whatever `voice` needs (if anything), then rebuild the synthesizer for
     * `language` + `voice`. Progress is shown in the download UI; UI controls stay disabled
     * until the operation finishes.
     */
    private fun ensureAssetsThenRecreate(language: KokoroLanguage, voice: KokoroVoice) {
        if (isDownloading) return
        val root = File(g2pRoot)

        if (!voice.needsDownload) {
            try {
                selectedLanguage = language
                selectedVoice = voice
                recreateSynthesizer(voice.id)
            } catch (e: Exception) {
                showError("Failed to load voice: ${e.message}")
            }
            updateUiState()
            return
        }

        isDownloading = true
        hideError()
        setProgressVisible(true)
        binding.loadingLabel.text = getString(R.string.initializing)
        updateUiState()

        thread {
            var errorMessage: String? = null
            try {
                TtsAssetDownloader.ensureAssetsPresent(
                    root,
                    language.id,
                    voice.id,
                    ::postDownloadProgress,
                )
            } catch (e: Exception) {
                errorMessage = "Download failed: ${e.message}"
            }
            mainHandler.post {
                isDownloading = false
                setProgressVisible(false)
                if (errorMessage != null) {
                    showError(errorMessage)
                } else {
                    try {
                        refreshVoices()
                        val stillSelected =
                            availableVoices.firstOrNull { it.id == voice.id }
                                ?: availableVoices.firstOrNull { !it.needsDownload }
                        selectedVoice = stillSelected
                        repopulateVoiceSpinner()
                        recreateSynthesizer(selectedVoice?.id)
                        hideError()
                    } catch (e: Exception) {
                        showError("Failed to load voice: ${e.message}")
                    }
                }
                updateUiState()
            }
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
                selectedVoice = availableVoices.firstOrNull { !it.needsDownload }
                    ?: availableVoices.first()
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
        val idx = availableVoices.indexOfFirst { it.id == selectedVoice?.id }
            .let { if (it >= 0) it else 0 }
        if (availableVoices.isNotEmpty()) {
            binding.voiceSpinner.setSelection(idx.coerceIn(0, availableVoices.lastIndex))
        }
        suppressSpinnerCallbacks = false
    }

    private fun updateUiState() {
        val busy = !engineReady || isSpeaking || isDownloading
        binding.languageSpinner.isEnabled = !busy
        binding.voiceSpinner.isEnabled = !busy && availableVoices.isNotEmpty()
        binding.inputText.isEnabled = !busy

        // When the input is empty, pressing Speak says "Hello world" (see speakCurrentText),
        // so we only need a usable engine/voice to enable the button.
        val canSpeak =
            engineReady &&
                !isSpeaking &&
                !isDownloading &&
                selectedVoice?.needsDownload == false
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
        speakUtterance(text.ifEmpty { "Hello world" })
    }

    /**
     * Synthesize PCM and play it via a plain [AudioTrack].
     *
     * We deliberately do not use `TextToSpeech.say()` / `AudioTrack.Builder.setContext()` here
     * because on some Android 15 emulators `audioserver` fails `validateUidPackagePair` for the
     * caller's UID and `AudioTrack` fails to initialize (see `logcat`: `AudioTrack-JNI ... Error`
     * `-22 initializing AudioTrack`). Building the track without `setContext` sidesteps that path.
     */
    private fun speakUtterance(text: String) {
        val t = tts ?: return
        isSpeaking = true
        updateUiState()
        thread {
            var errorMessage: String? = null
            try {
                val result = t.synthesize(text)
                val samples = result.samples ?: FloatArray(0)
                val sampleRate = result.sampleRateHz
                if (samples.isNotEmpty() && sampleRate > 0) {
                    playSamples(samples, sampleRate)
                }
            } catch (e: Exception) {
                errorMessage = "Speech failed: ${e.message}"
            } finally {
                mainHandler.post {
                    isSpeaking = false
                    if (errorMessage != null) {
                        binding.errorText.text = errorMessage
                        binding.errorText.visibility = View.VISIBLE
                    }
                    updateUiState()
                }
            }
        }
    }

    private fun playSamples(samples: FloatArray, sampleRate: Int) {
        val minBuf =
            AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
            )
        if (minBuf <= 0) {
            throw RuntimeException("AudioTrack.getMinBufferSize failed for sampleRate=$sampleRate")
        }
        val attrs =
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
        val format =
            AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build()
        val track =
            AudioTrack.Builder()
                .setAudioAttributes(attrs)
                .setAudioFormat(format)
                .setBufferSizeInBytes(minBuf.coerceAtLeast(samples.size * 4).coerceAtMost(minBuf * 8))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        try {
            if (track.state != AudioTrack.STATE_INITIALIZED) {
                throw RuntimeException("AudioTrack failed to initialize (state=${track.state})")
            }
            track.play()
            var offset = 0
            while (offset < samples.size) {
                val wrote =
                    track.write(
                        samples,
                        offset,
                        samples.size - offset,
                        AudioTrack.WRITE_BLOCKING,
                    )
                if (wrote <= 0) {
                    throw RuntimeException("AudioTrack.write returned $wrote")
                }
                offset += wrote
            }
            val deadline = System.nanoTime() + 60_000_000_000L
            while (System.nanoTime() < deadline) {
                if (track.playbackHeadPosition >= samples.size - 1) break
                try {
                    Thread.sleep(10)
                } catch (_: InterruptedException) {
                    Thread.currentThread().interrupt()
                    break
                }
            }
        } finally {
            try {
                track.stop()
            } catch (_: Exception) {
            }
            track.release()
        }
    }

    /** Called from the download thread; posts updates to the UI thread. */
    private fun postDownloadProgress(
        key: String,
        fileIndex: Int,
        totalFiles: Int,
        bytesDownloaded: Long,
        bytesTotal: Long,
    ) {
        mainHandler.post {
            binding.loadingLabel.text =
                getString(R.string.downloading_asset, displayNameForKey(key), fileIndex, totalFiles)
            val indicator = binding.downloadProgress
            indicator.visibility = View.VISIBLE
            if (bytesTotal > 0) {
                indicator.isIndeterminate = false
                val pct = ((bytesDownloaded * 100L) / bytesTotal).coerceIn(0L, 100L).toInt()
                indicator.progress = pct
            } else {
                indicator.isIndeterminate = true
            }
        }
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
        engineReady = tts != null && g2pRoot.isNotEmpty()
        binding.errorText.text = message
        binding.errorText.visibility = View.VISIBLE
    }

    private fun hideError() {
        binding.errorText.visibility = View.GONE
    }
}
