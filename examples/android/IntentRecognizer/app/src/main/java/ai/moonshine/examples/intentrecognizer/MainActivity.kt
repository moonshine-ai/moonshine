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
import ai.moonshine.voice.IntentRecognizer
import ai.moonshine.voice.JNI
import ai.moonshine.voice.MicTranscriber
import ai.moonshine.voice.TranscriptEvent
import ai.moonshine.voice.TranscriptEventListener
import java.io.File

class MainActivity : AppCompatActivity() {

    /**
     * Bundled under `app/src/main/assets/` (Git LFS). Streaming ASR must be loaded from
     * disk via [MicTranscriber.loadFromFiles], so we mirror assets into `filesDir` first.
     */
    private val asrAssetDir = "small-streaming-en"
    private val embedAssetDir = "embeddinggemma-300m"

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
        if (!hasSmallStreamingAssetsInApk()) {
            binding.statusText.text =
                "Missing bundled ASR weights under assets/$asrAssetDir/ " +
                    "(small English streaming: frontend, encoder, adapter, cross_kv, " +
                    "decoder_kv, streaming_config.json, tokenizer.bin). " +
                    "Clone this folder with Git LFS so ONNX assets are present."
            return
        }
        if (!hasEmbeddingAssetsInApk()) {
            binding.statusText.text =
                "Missing bundled embedding weights under assets/$embedAssetDir/ " +
                    "(model_q4.onnx, model_q4.onnx_data, tokenizer.bin). Use Git LFS."
            return
        }

        try {
            val asrRoot = File(filesDir, asrAssetDir)
            AssetDirectoryCopy.copyDirIfNeeded(
                this,
                asrAssetDir,
                asrRoot,
                "streaming_config.json",
            )

            val embedRoot = File(filesDir, embedAssetDir)
            AssetDirectoryCopy.copyDirIfNeeded(
                this,
                embedAssetDir,
                embedRoot,
                "tokenizer.bin",
            )

            val ir = IntentRecognizer(
                embedRoot.absolutePath,
                JNI.MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
                "q4",
            )
            intentRecognizer = ir
            applyRegisteredIntents()

            val m = MicTranscriber()
            mic = m
            m.addListener(transcriptListener)
            m.loadFromFiles(asrRoot.absolutePath, JNI.MOONSHINE_MODEL_ARCH_SMALL_STREAMING)
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
            ) {
                m.onMicPermissionGranted()
            }

            engineReady = true
            binding.statusText.text = "Ready. Tap Listen to use the microphone."
            updateDiagnostics()
        } catch (e: Exception) {
            binding.statusText.text = "Failed to load models: ${e.message}"
            intentRecognizer?.close()
            intentRecognizer = null
            mic = null
        }
    }

    private fun hasSmallStreamingAssetsInApk(): Boolean {
        val n = assets.list(asrAssetDir)?.toSet() ?: return false
        return n.containsAll(
            setOf(
                "frontend.ort",
                "encoder.ort",
                "adapter.ort",
                "cross_kv.ort",
                "decoder_kv.ort",
                "streaming_config.json",
                "tokenizer.bin",
            ),
        )
    }

    private fun hasEmbeddingAssetsInApk(): Boolean {
        val n = assets.list(embedAssetDir)?.toSet() ?: return false
        return n.contains("tokenizer.bin") && n.contains("model_q4.onnx")
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
