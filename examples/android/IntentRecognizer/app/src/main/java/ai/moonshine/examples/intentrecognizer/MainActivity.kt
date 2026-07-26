package ai.moonshine.examples.intentrecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import ai.moonshine.examples.intentrecognizer.databinding.ActivityMainBinding
import ai.moonshine.voice.CatalogLoader
import ai.moonshine.voice.DownloadProgress
import ai.moonshine.voice.IntentRecognizer
import ai.moonshine.voice.JNI
import ai.moonshine.voice.LoadCallback
import ai.moonshine.voice.MicTranscriber
import ai.moonshine.voice.ModelSpec
import ai.moonshine.voice.TranscriptEvent
import ai.moonshine.voice.TranscriptEventListener

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var adapter: PhraseAdapter
    private val debounceHandler = Handler(Looper.getMainLooper())
    private var debounceRunnable: Runnable? = null

    private var intentRecognizer: IntentRecognizer? = null
    private var mic: MicTranscriber? = null
    private var engineReady = false
    private var listening = false
    private var pendingListenAfterPermission = false

    private var lastScoredMinimum: Float? = null
    private var lastTopSimilarity: Float? = null

    private val defaultPhrases = listOf(
        "turn on the lights",
        "turn off the lights",
        "what is the weather",
        "set a timer",
        "play some music",
        "stop the music",
    )

    private val transcriptListener = java.util.function.Consumer<TranscriptEvent> { event ->
        event.accept(
            object : TranscriptEventListener() {
                override fun onLineStarted(e: TranscriptEvent.LineStarted) {
                    runOnUiThread { binding.liveTranscript.text = e.line.text.orEmpty() }
                }

                override fun onLineTextChanged(e: TranscriptEvent.LineTextChanged) {
                    runOnUiThread { binding.liveTranscript.text = e.line.text.orEmpty() }
                }

                override fun onLineCompleted(e: TranscriptEvent.LineCompleted) {
                    runOnUiThread {
                        binding.liveTranscript.text = e.line.text.orEmpty()
                        handleCompletedTranscriptLine(e.line.text.orEmpty())
                    }
                }
            },
        )
    }

    private val micPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                mic?.onMicPermissionGranted()
                if (pendingListenAfterPermission) {
                    pendingListenAfterPermission = false
                    startListeningInternal()
                }
            } else {
                pendingListenAfterPermission = false
                binding.statusText.text = "Microphone permission is required to listen."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        adapter = PhraseAdapter(
            onScheduleIntentSync = { scheduleDebouncedIntentSync() },
            onRemoveRow = { id ->
                adapter.removeRow(id)
                commitIntentPhrases()
            },
        )
        adapter.resetToDefaults(defaultPhrases)
        binding.phraseRecycler.layoutManager = LinearLayoutManager(this)
        binding.phraseRecycler.adapter = adapter

        binding.addPhraseButton.setOnClickListener { adapter.addEmptyRow() }

        binding.thresholdSlider.addOnChangeListener { _, value, fromUser ->
            binding.thresholdLabel.text = String.format("Threshold: %.2f", value)
            if (fromUser) {
                updateDiagnostics()
            }
        }
        binding.thresholdLabel.text =
            String.format("Threshold: %.2f", binding.thresholdSlider.value)

        binding.listenButton.setOnClickListener { toggleListen() }

        bootstrapEngine()
    }

    override fun onPause() {
        super.onPause()
        if (listening) {
            try {
                mic?.stop()
            } catch (_: Exception) {
            }
            listening = false
            binding.liveTranscript.text = ""
            lastScoredMinimum = null
            lastTopSimilarity = null
            binding.listenButton.setText(R.string.listen)
            binding.statusText.text = "Paused (activity in background)."
            updateDiagnostics()
        }
    }

    override fun onDestroy() {
        debounceRunnable?.let { debounceHandler.removeCallbacks(it) }
        mic?.removeListener(transcriptListener)
        if (listening) {
            try {
                mic?.stop()
            } catch (_: Exception) {
            }
        }
        intentRecognizer?.close()
        intentRecognizer = null
        mic = null
        super.onDestroy()
    }

    private fun bootstrapEngine() {
        // Download the Medium Streaming English speech model and the embedding model on first run
        // into a managed cache directory (nothing is bundled in the APK), then construct both
        // engines. CatalogLoader does the download off the main thread and delivers progress and
        // the result back on the main thread, so this activity no longer needs any Thread /
        // runOnUiThread plumbing for bootstrap.
        binding.statusText.text = "Downloading models (first run only)…"
        val sttSpec = ModelSpec.stt("en", JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, false)
        val intentSpec = ModelSpec.intent(null, "q4")

        CatalogLoader.load(
            this,
            listOf(sttSpec, intentSpec),
            CatalogLoader.Builder<Pair<IntentRecognizer, MicTranscriber>> { directories ->
                val recognizer = IntentRecognizer(
                    directories[intentSpec]!!.absolutePath,
                    JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
                    "q4",
                )
                val micTranscriber = MicTranscriber()
                micTranscriber.addListener(transcriptListener)
                micTranscriber.loadFromFiles(
                    directories[sttSpec]!!.absolutePath,
                    JNI.MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
                )
                Pair(recognizer, micTranscriber)
            },
            object : LoadCallback<Pair<IntentRecognizer, MicTranscriber>> {
                override fun onProgress(progress: DownloadProgress) {
                    val pct =
                        if (progress.bytesTotal > 0)
                            (progress.bytesDownloaded * 100 / progress.bytesTotal)
                        else 0
                    binding.statusText.text = "Downloading ${progress.relativePath} ($pct%)…"
                }

                override fun onSuccess(engines: Pair<IntentRecognizer, MicTranscriber>) {
                    intentRecognizer = engines.first
                    mic = engines.second
                    applyRegisteredIntents()
                    if (ContextCompat.checkSelfPermission(
                            this@MainActivity, Manifest.permission.RECORD_AUDIO,
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        engines.second.onMicPermissionGranted()
                    }
                    engineReady = true
                    binding.statusText.text = "Ready. Tap Listen to use the microphone."
                    updateDiagnostics()
                }

                override fun onError(error: Throwable) {
                    binding.statusText.text = "Failed to load models: ${error.message}"
                    intentRecognizer?.close()
                    intentRecognizer = null
                    mic = null
                }
            },
        )
    }

    private fun applyRegisteredIntents() {
        val ir = intentRecognizer ?: return
        ir.clearIntents()
        for (p in adapter.currentPhrases()) {
            if (p.isNotEmpty()) {
                ir.registerIntent(p)
            }
        }
    }

    private fun scheduleDebouncedIntentSync() {
        debounceRunnable?.let { debounceHandler.removeCallbacks(it) }
        debounceRunnable = Runnable { commitIntentPhrases() }
        debounceHandler.postDelayed(debounceRunnable!!, 450L)
    }

    private fun commitIntentPhrases() {
        if (!engineReady) return
        try {
            applyRegisteredIntents()
            if (!listening && !binding.statusText.text.contains("Missing")) {
                binding.statusText.text = "Intents updated."
            }
        } catch (e: Exception) {
            binding.statusText.text = "Could not update intents: ${e.message}"
        }
    }

    private fun toggleListen() {
        if (!engineReady) return
        val m = mic ?: return
        try {
            if (listening) {
                m.stop()
                listening = false
                binding.liveTranscript.text = ""
                lastScoredMinimum = null
                lastTopSimilarity = null
                binding.listenButton.setText(R.string.listen)
                binding.statusText.text = "Stopped."
                updateDiagnostics()
            } else {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
                    PackageManager.PERMISSION_GRANTED
                ) {
                    pendingListenAfterPermission = true
                    micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    return
                }
                startListeningInternal()
            }
        } catch (e: Exception) {
            binding.statusText.text = "Microphone error: ${e.message}"
            listening = false
            pendingListenAfterPermission = false
            binding.listenButton.setText(R.string.listen)
        }
    }

    private fun startListeningInternal() {
        val m = mic ?: return
        m.onMicPermissionGranted()
        binding.liveTranscript.text = ""
        lastScoredMinimum = null
        lastTopSimilarity = null
        binding.statusText.text = ""
        m.start()
        listening = true
        binding.listenButton.setText(R.string.stop)
        updateDiagnostics()
    }

    private fun handleCompletedTranscriptLine(raw: String) {
        val utterance = raw.trim()
        if (utterance.isEmpty()) return
        val ir = intentRecognizer ?: return

        val minimumSimilarity = binding.thresholdSlider.value
        lastScoredMinimum = minimumSimilarity

        val matches = try {
            ir.getClosestIntents(utterance, minimumSimilarity)
        } catch (e: Exception) {
            binding.statusText.text = "Intent match error: ${e.message}"
            return
        }

        val top = matches.firstOrNull()
        if (top == null) {
            lastTopSimilarity = null
            updateDiagnostics()
            return
        }

        if (top.similarity + 1e-5f < minimumSimilarity) {
            lastTopSimilarity = top.similarity
            updateDiagnostics()
            return
        }

        lastTopSimilarity = top.similarity
        val rowId = adapter.rowIdMatchingCanonical(top.canonicalPhrase)
        if (rowId != null) {
            adapter.flashHighlight(rowId)
        }
        updateDiagnostics()
    }

    private fun updateDiagnostics() {
        val minStr = lastScoredMinimum?.let { String.format("%.2f", it) } ?: "—"
        val topStr = lastTopSimilarity?.let { String.format("%.3f", it) } ?: "—"
        binding.diagnosticsText.text =
            "Last scored minimum: $minStr | Top match: $topStr"
    }
}
