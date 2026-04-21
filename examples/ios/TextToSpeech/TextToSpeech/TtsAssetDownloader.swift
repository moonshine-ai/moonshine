import Foundation
import MoonshineVoice

/// Resolves TTS asset dependencies via `TextToSpeech.getDependencies` and downloads any missing
/// files from the Moonshine CDN (`https://download.moonshine.ai/tts/<relative/key>`) into
/// `<g2pRoot>/<relative/key>`.
enum TtsAssetDownloader {
    private static let cdnBase = "https://download.moonshine.ai/tts/"
    private static let tmpSuffix = ".download"

    struct Progress {
        /// Relative asset key being fetched (e.g. `en_us/dict.tsv`).
        let key: String
        /// 1-based index of the file being downloaded this pass.
        let fileIndex: Int
        /// Total number of files that will be downloaded this pass.
        let totalFiles: Int
        /// Bytes written for the current file so far.
        let bytesDownloaded: Int64
        /// Total bytes for the current file (`-1` if the server did not report `Content-Length`).
        let bytesTotal: Int64
    }

    enum DownloadError: LocalizedError {
        case httpStatus(Int, URL)
        case badResponse(URL)
        case move(URL, Error)

        var errorDescription: String? {
            switch self {
            case .httpStatus(let code, let url):
                return "HTTP \(code) fetching \(url.lastPathComponent)"
            case .badResponse(let url):
                return "Invalid response for \(url.lastPathComponent)"
            case .move(let url, let error):
                return "Failed to save \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// Make sure every asset required for `language` + optional prefixed `voice`
    /// (e.g. `kokoro_af_alloy`, `piper_en_US-ryan-low`) exists under `g2pRoot`.
    /// Downloads each missing file synchronously; callers must invoke this off the main actor.
    static func ensureAssetsPresent(
        g2pRoot: URL,
        language: String,
        voice: String?,
        onProgress: @Sendable @escaping (Progress) -> Void
    ) async throws {
        let keys = try resolveDependencyKeys(
            g2pRoot: g2pRoot, language: language, voice: voice)
        let missing = keys.filter { key in
            !key.isEmpty && key.contains("/")
                && !FileManager.default.fileExists(
                    atPath: g2pRoot.appendingPathComponent(key).path)
        }
        if missing.isEmpty { return }

        for (index, key) in missing.enumerated() {
            try await downloadOne(
                key: key,
                dest: g2pRoot.appendingPathComponent(key),
                fileIndex: index + 1,
                totalFiles: missing.count,
                onProgress: onProgress
            )
        }
    }

    private static func resolveDependencyKeys(
        g2pRoot: URL, language: String, voice: String?
    ) throws -> [String] {
        var options: [TranscriberOption] = [
            TranscriberOption(name: "g2p_root", value: g2pRoot.path)
        ]
        if let voice = voice {
            options.append(TranscriberOption(name: "voice", value: voice))
        }
        let json = try TextToSpeech.getDependencies(
            languages: language, options: options)
        guard let data = json.data(using: .utf8),
            let array = try JSONSerialization.jsonObject(with: data) as? [String]
        else { return [] }
        return array
    }

    private static func downloadOne(
        key: String,
        dest: URL,
        fileIndex: Int,
        totalFiles: Int,
        onProgress: @Sendable @escaping (Progress) -> Void
    ) async throws {
        guard let url = cdnURL(for: key) else {
            throw URLError(.badURL)
        }

        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let tmp = dest.appendingPathExtension(String(tmpSuffix.dropFirst()))
        try? FileManager.default.removeItem(at: tmp)

        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.badResponse(url)
        }
        guard (200...299).contains(http.statusCode) else {
            throw DownloadError.httpStatus(http.statusCode, url)
        }
        let totalBytes = http.expectedContentLength

        onProgress(
            Progress(
                key: key, fileIndex: fileIndex, totalFiles: totalFiles,
                bytesDownloaded: 0, bytesTotal: totalBytes))

        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: tmp) else {
            throw DownloadError.badResponse(url)
        }
        defer { try? handle.close() }

        var buffer = Data()
        buffer.reserveCapacity(64 * 1024)
        var downloaded: Int64 = 0
        var lastReported: Int64 = 0

        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 64 * 1024 {
                try handle.write(contentsOf: buffer)
                downloaded += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if downloaded - lastReported >= 64 * 1024 {
                    onProgress(
                        Progress(
                            key: key, fileIndex: fileIndex, totalFiles: totalFiles,
                            bytesDownloaded: downloaded, bytesTotal: totalBytes))
                    lastReported = downloaded
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            downloaded += Int64(buffer.count)
        }
        try handle.close()

        onProgress(
            Progress(
                key: key, fileIndex: fileIndex, totalFiles: totalFiles,
                bytesDownloaded: downloaded, bytesTotal: totalBytes))

        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        do {
            try FileManager.default.moveItem(at: tmp, to: dest)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            throw DownloadError.move(dest, error)
        }
    }

    /// Percent-encode each path segment; the library expects canonical `/`-joined relative keys.
    private static func cdnURL(for key: String) -> URL? {
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "/"))
        let encoded =
            key
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { segment -> String in
                String(segment).addingPercentEncoding(withAllowedCharacters: allowed) ?? String(
                    segment)
            }
            .joined(separator: "/")
        return URL(string: cdnBase + encoded)
    }
}
