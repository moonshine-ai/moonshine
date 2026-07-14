import Foundation

/// Which model's files to resolve and download. Each case maps to one of the native dependency
/// APIs (`moonshine_get_*_dependencies`), so the file list always comes from the library rather
/// than being hardcoded here.
public enum ModelSpec: Sendable {
    /// Speech-to-text transcription model. `modelArch` selects the architecture (nil = the default
    /// for the language); `includeSpelling` also fetches the alphanumeric spelling model when one
    /// is published for the language.
    case stt(language: String, modelArch: ModelArch? = nil, includeSpelling: Bool = false)
    /// Text-to-speech voice. `voice` is a prefixed id (e.g. `kokoro_af_heart`,
    /// `piper_en_US-lessac-medium`); nil uses the language default.
    case tts(language: String, voice: String? = nil)
    /// Intent-recognition embedding model. `variant` is e.g. `q4` (nil = default).
    case intent(modelName: String = "embeddinggemma-300m", variant: String? = nil)
    /// Grapheme-to-phoneme assets for a language (lexicons / ONNX bundles).
    case g2p(language: String)
}

/// Progress for a single file within an ``AssetDownloader`` run.
public struct DownloadProgress: Sendable {
    /// Path of the file relative to the download root (e.g. `encoder_model.ort` or `en_us/dict.tsv`).
    public let relativePath: String
    /// 1-based index of the file being downloaded in this run.
    public let fileIndex: Int
    /// Total number of files that will be downloaded in this run.
    public let totalFiles: Int
    /// Bytes written for the current file so far.
    public let bytesDownloaded: Int64
    /// Total bytes for the current file, or `-1` if the server did not report a length.
    public let bytesTotal: Int64
}

/// Downloads the model/data files a Moonshine engine needs into an app-chosen directory, then hands
/// back that directory for loading with ``Transcriber``, ``TextToSpeech``, or ``IntentRecognizer``.
///
/// This is **opt-in**: apps that bundle their models never need it, and default behavior is
/// unchanged. Downloads are resolved from the native dependency catalog, written atomically (via a
/// `.part` file), resumable across interruptions, and reported through an optional progress
/// callback so the app can show UI and satisfy App Store expectations around consent and progress.
///
/// Files already present under the root are skipped, so calling ``ensureModelPresent(root:spec:onProgress:)``
/// repeatedly is cheap.
///
/// Requires iOS 15 / macOS 12 for the async byte-stream download API. Apps deploying to older OSes
/// should bundle their models instead.
@available(iOS 15.0, macOS 12.0, *)
public final class AssetDownloader: @unchecked Sendable {
    /// CDN root for TTS / G2P canonical asset keys (mirrors the STT/embedding host used by the
    /// native manifests, which embed their own absolute `base_url`).
    private static let ttsCdnBase = "https://download.moonshine.ai/tts/"
    private static let partSuffix = "part"
    private static let progressChunkBytes: Int64 = 256 * 1024

    private let session: URLSession
    private let api = MoonshineAPI.shared

    /// - Parameters:
    ///   - allowsCellularAccess: When false, downloads only proceed over unmetered (e.g. Wi-Fi)
    ///     connections. Defaults to true. Ignored when a custom `session` is supplied.
    ///   - timeout: Per-request and per-resource timeout in seconds. Ignored when a custom
    ///     `session` is supplied.
    ///   - session: An explicit `URLSession` (used by tests to inject a mock protocol). When nil, a
    ///     session is created from the parameters above.
    public init(
        allowsCellularAccess: Bool = true,
        timeout: TimeInterval = 120,
        session: URLSession? = nil
    ) {
        if let session = session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.default
            configuration.allowsCellularAccess = allowsCellularAccess
            configuration.waitsForConnectivity = true
            configuration.timeoutIntervalForRequest = timeout
            configuration.timeoutIntervalForResource = timeout
            self.session = URLSession(configuration: configuration)
        }
    }

    // MARK: - Public API

    /// Returns true when every file required by `spec` already exists under `root`.
    public func isModelPresent(root: URL, spec: ModelSpec) -> Bool {
        guard let files = try? resolveFiles(root: root, spec: spec), !files.isEmpty else {
            return false
        }
        let fileManager = FileManager.default
        return files.allSatisfy { file in
            fileManager.fileExists(atPath: root.appendingPathComponent(file.relativePath).path)
        }
    }

    /// Ensures every file required by `spec` is present under `root`, downloading any that are
    /// missing, and returns `root` (the directory to pass to the engine loader).
    ///
    /// Existing files are left untouched. Progress is reported per file through `onProgress`.
    /// Throws ``AssetDownloadError`` on download failures and ``MoonshineError`` if the manifest
    /// could not be resolved.
    @discardableResult
    public func ensureModelPresent(
        root: URL,
        spec: ModelSpec,
        onProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        let files = try resolveFiles(root: root, spec: spec)
        let fileManager = FileManager.default
        let missing = files.filter { file in
            !fileManager.fileExists(atPath: root.appendingPathComponent(file.relativePath).path)
        }
        guard !missing.isEmpty else { return root }

        for (index, file) in missing.enumerated() {
            try Task.checkCancellation()
            try await downloadOne(
                file: file,
                root: root,
                fileIndex: index + 1,
                totalFiles: missing.count,
                onProgress: onProgress
            )
        }
        return root
    }

    // MARK: - Manifest resolution

    private struct ResolvedFile {
        let url: URL
        let relativePath: String
    }

    private func resolveFiles(root: URL, spec: ModelSpec) throws -> [ResolvedFile] {
        switch spec {
        case .stt(let language, let modelArch, let includeSpelling):
            var options: [TranscriberOption] = []
            if let modelArch = modelArch {
                options.append(TranscriberOption(name: "model_arch", value: String(modelArch.rawValue)))
            }
            if includeSpelling {
                options.append(TranscriberOption(name: "include_spelling", value: "true"))
            }
            let json = try api.getSttDependencies(language: language, options: options)
            return try filesFromGroupManifest(json)

        case .intent(let modelName, let variant):
            var options: [TranscriberOption] = []
            if let variant = variant {
                options.append(TranscriberOption(name: "variant", value: variant))
            }
            let json = try api.getIntentDependencies(modelName: modelName, options: options)
            return try filesFromGroupManifest(json)

        case .tts(let language, let voice):
            var options: [TranscriberOption] = [
                TranscriberOption(name: "g2p_root", value: root.path)
            ]
            if let voice = voice {
                options.append(TranscriberOption(name: "voice", value: voice))
            }
            let json = try api.getTtsDependencies(languages: language, options: options)
            return try filesFromKeyArray(json)

        case .g2p(let language):
            let options: [TranscriberOption] = [
                TranscriberOption(name: "g2p_root", value: root.path)
            ]
            let csv = try api.getG2pDependencies(languages: language, options: options)
            return filesFromKeyList(csv.split(separator: ",").map(String.init))
        }
    }

    /// Parses the `{"groups":[{"base_url":..,"files":[..]}]}` manifest emitted by the STT and intent
    /// dependency APIs. Files are downloaded from `base_url + "/" + file` and stored under their bare
    /// filename in the root.
    private func filesFromGroupManifest(_ json: String) throws -> [ResolvedFile] {
        guard let data = json.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let groups = object["groups"] as? [[String: Any]]
        else {
            throw AssetDownloadError.invalidManifest(detail: "expected {\"groups\": [...]}: \(json)")
        }
        var result: [ResolvedFile] = []
        for group in groups {
            guard let baseURLString = group["base_url"] as? String,
                let files = group["files"] as? [String]
            else {
                throw AssetDownloadError.invalidManifest(detail: "malformed group in \(json)")
            }
            for file in files {
                guard let url = URL(string: baseURLString + "/" + file) else {
                    throw AssetDownloadError.invalidManifest(detail: "bad URL for \(file)")
                }
                result.append(ResolvedFile(url: url, relativePath: file))
            }
        }
        return result
    }

    /// Parses the flat JSON array of canonical keys emitted by the TTS dependency API.
    private func filesFromKeyArray(_ json: String) throws -> [ResolvedFile] {
        guard let data = json.data(using: .utf8),
            let keys = try? JSONSerialization.jsonObject(with: data) as? [String]
        else {
            throw AssetDownloadError.invalidManifest(detail: "expected JSON array of keys: \(json)")
        }
        return filesFromKeyList(keys)
    }

    /// Maps TTS/G2P canonical keys (e.g. `en_us/dict.tsv`) to CDN URLs, skipping in-memory override
    /// labels that have no path component.
    private func filesFromKeyList(_ keys: [String]) -> [ResolvedFile] {
        var result: [ResolvedFile] = []
        for rawKey in keys {
            let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty, key.contains("/"), let url = cdnURL(forKey: key) else {
                continue
            }
            result.append(ResolvedFile(url: url, relativePath: key))
        }
        return result
    }

    /// Percent-encodes each path segment of a canonical key under the TTS CDN base.
    private func cdnURL(forKey key: String) -> URL? {
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "/"))
        let encoded =
            key
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { segment -> String in
                String(segment).addingPercentEncoding(withAllowedCharacters: allowed)
                    ?? String(segment)
            }
            .joined(separator: "/")
        return URL(string: Self.ttsCdnBase + encoded)
    }

    // MARK: - Single-file download

    private func downloadOne(
        file: ResolvedFile,
        root: URL,
        fileIndex: Int,
        totalFiles: Int,
        onProgress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws {
        let fileManager = FileManager.default
        let destination = root.appendingPathComponent(file.relativePath)
        let directory = destination.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw AssetDownloadError.fileWrite(path: directory.path, underlying: error)
        }

        let partURL = destination.appendingPathExtension(Self.partSuffix)

        // Resume a prior partial download when possible.
        var existingBytes: Int64 = 0
        if let attributes = try? fileManager.attributesOfItem(atPath: partURL.path),
            let size = attributes[.size] as? Int64 {
            existingBytes = size
        }

        var request = URLRequest(url: file.url)
        if existingBytes > 0 {
            request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
        }

        let (byteStream, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AssetDownloadError.badResponse(url: file.url)
        }
        guard (200...299).contains(http.statusCode) else {
            throw AssetDownloadError.httpStatus(code: http.statusCode, url: file.url)
        }

        // 206 => server honored our Range and is sending the remainder; anything else (typically
        // 200) means it ignored the range, so start the file over.
        let isResuming = existingBytes > 0 && http.statusCode == 206
        if !isResuming {
            existingBytes = 0
            try? fileManager.removeItem(at: partURL)
        }

        let remainingBytes = http.expectedContentLength  // -1 if unknown
        let totalBytes = remainingBytes >= 0 ? existingBytes + remainingBytes : -1

        try ensureSpaceAvailable(forVolumeAt: directory, needBytes: remainingBytes, url: file.url)

        if !fileManager.fileExists(atPath: partURL.path) {
            fileManager.createFile(atPath: partURL.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: partURL) else {
            throw AssetDownloadError.fileWrite(
                path: partURL.path, underlying: AssetDownloadError.badResponse(url: file.url))
        }
        if isResuming {
            _ = try? handle.seekToEnd()
        }

        onProgress?(
            DownloadProgress(
                relativePath: file.relativePath, fileIndex: fileIndex, totalFiles: totalFiles,
                bytesDownloaded: existingBytes, bytesTotal: totalBytes))

        var buffer = Data()
        buffer.reserveCapacity(64 * 1024)
        var downloaded = existingBytes
        var lastReported = existingBytes

        do {
            for try await byte in byteStream {
                if Task.isCancelled {
                    try? handle.close()
                    throw AssetDownloadError.cancelled
                }
                buffer.append(byte)
                if buffer.count >= 64 * 1024 {
                    try handle.write(contentsOf: buffer)
                    downloaded += Int64(buffer.count)
                    buffer.removeAll(keepingCapacity: true)
                    if downloaded - lastReported >= Self.progressChunkBytes {
                        onProgress?(
                            DownloadProgress(
                                relativePath: file.relativePath, fileIndex: fileIndex,
                                totalFiles: totalFiles, bytesDownloaded: downloaded,
                                bytesTotal: totalBytes))
                        lastReported = downloaded
                    }
                }
            }
            if !buffer.isEmpty {
                try handle.write(contentsOf: buffer)
                downloaded += Int64(buffer.count)
            }
            try handle.close()
        } catch let error as AssetDownloadError {
            throw error
        } catch {
            try? handle.close()
            throw AssetDownloadError.fileWrite(path: partURL.path, underlying: error)
        }

        onProgress?(
            DownloadProgress(
                relativePath: file.relativePath, fileIndex: fileIndex, totalFiles: totalFiles,
                bytesDownloaded: downloaded, bytesTotal: totalBytes))

        // Atomically move the completed .part file into place.
        do {
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: partURL, to: destination)
        } catch {
            try? fileManager.removeItem(at: partURL)
            throw AssetDownloadError.fileWrite(path: destination.path, underlying: error)
        }
    }

    /// Best-effort free-space precheck before writing a file whose size the server reported.
    private func ensureSpaceAvailable(forVolumeAt directory: URL, needBytes: Int64, url: URL) throws {
        guard needBytes > 0 else { return }
        let values = try? directory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey
        ])
        guard let available = values?.volumeAvailableCapacityForImportantUsage else { return }
        // Keep a small safety margin so we do not fill the volume completely.
        let margin: Int64 = 8 * 1024 * 1024
        if available < needBytes + margin {
            throw AssetDownloadError.insufficientSpace(
                needBytes: needBytes, availableBytes: available, url: url)
        }
    }
}
