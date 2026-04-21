package ai.moonshine.examples.texttospeech

import ai.moonshine.voice.TextToSpeech
import ai.moonshine.voice.TranscriberOption
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONArray

/**
 * Resolves TTS asset dependencies via `moonshine_get_tts_dependencies` and downloads any missing
 * files from the Moonshine CDN (`https://download.moonshine.ai/tts/<relative/key>`) into
 * `<g2pRoot>/<relative/key>`.
 */
object TtsAssetDownloader {

    private const val CDN_BASE = "https://download.moonshine.ai/tts/"
    private const val TMP_SUFFIX = ".download"

    fun interface ProgressListener {
        /**
         * @param key current asset key (e.g. `en_us/dict.tsv`).
         * @param fileIndex 1-based index of the file being downloaded.
         * @param totalFiles total number of files that will be downloaded this pass.
         * @param bytesDownloaded bytes written for this file so far.
         * @param bytesTotal total bytes for this file (may be `-1` if the server did not report `Content-Length`).
         */
        fun onProgress(
            key: String,
            fileIndex: Int,
            totalFiles: Int,
            bytesDownloaded: Long,
            bytesTotal: Long,
        )
    }

    /**
     * Make sure every asset required for `language` + optional prefixed `voice` (e.g.
     * `kokoro_af_alloy`, `piper_en_US-ryan-low`) exists under `g2pRoot`. Downloads each missing
     * file synchronously; callers must invoke this off the main thread.
     */
    fun ensureAssetsPresent(
        g2pRoot: File,
        language: String,
        voice: String?,
        listener: ProgressListener,
    ) {
        val keys = resolveDependencyKeys(g2pRoot, language, voice)
        val missing =
            keys.filter { key ->
                key.isNotEmpty() && key.contains('/') && !File(g2pRoot, key).exists()
            }
        if (missing.isEmpty()) return

        missing.forEachIndexed { index, key ->
            val dest = File(g2pRoot, key)
            dest.parentFile?.mkdirs()
            downloadOne(key, dest, index + 1, missing.size, listener)
        }
    }

    private fun resolveDependencyKeys(
        g2pRoot: File,
        language: String,
        voice: String?,
    ): List<String> {
        val opts = ArrayList<TranscriberOption>()
        opts.add(TranscriberOption("g2p_root", g2pRoot.absolutePath))
        if (voice != null) {
            opts.add(TranscriberOption("voice", voice))
        }
        val json = TextToSpeech.getTtsDependencies(language, opts)
        val arr = JSONArray(json)
        val out = ArrayList<String>(arr.length())
        for (i in 0 until arr.length()) {
            out.add(arr.optString(i, ""))
        }
        return out
    }

    private fun downloadOne(
        key: String,
        dest: File,
        fileIndex: Int,
        totalFiles: Int,
        listener: ProgressListener,
    ) {
        val url = URL(CDN_BASE + encodeKey(key))
        val tmp = File(dest.parentFile, dest.name + TMP_SUFFIX)
        tmp.delete()

        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 60_000
        conn.instanceFollowRedirects = true
        try {
            conn.connect()
            val code = conn.responseCode
            if (code !in 200..299) {
                throw IOException("HTTP $code fetching $url")
            }
            val totalBytes = conn.contentLengthLong
            listener.onProgress(key, fileIndex, totalFiles, 0L, totalBytes)

            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var downloaded = 0L
                    var lastReported = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        downloaded += read
                        if (downloaded - lastReported >= 64 * 1024 || read == 0) {
                            listener.onProgress(key, fileIndex, totalFiles, downloaded, totalBytes)
                            lastReported = downloaded
                        }
                    }
                    listener.onProgress(key, fileIndex, totalFiles, downloaded, totalBytes)
                }
            }
        } finally {
            conn.disconnect()
        }

        if (!tmp.renameTo(dest)) {
            tmp.delete()
            throw IOException("Failed to move download into place: ${dest.absolutePath}")
        }
    }

    /** Percent-encode each path segment; the library expects canonical `/`-joined relative keys. */
    private fun encodeKey(key: String): String =
        key.split('/').joinToString("/") { segment ->
            java.net.URLEncoder.encode(segment, "UTF-8").replace("+", "%20")
        }
}
