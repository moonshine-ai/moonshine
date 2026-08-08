import Foundation

/// Default, binding-managed on-disk location for models downloaded via the high-level
/// ``MicTranscriber/load(language:modelArch:cacheDirectory:includeSpelling:includeWordTimestamps:options:downloader:onProgress:)``
/// (and the sibling `load` factories on ``Transcriber`` and ``TextToSpeech``).
///
/// Apps that want to control where models live can pass an explicit directory to those factories
/// (or keep using ``AssetDownloader`` directly); this type just provides a sensible default so the
/// common case needs no filesystem boilerplate.
///
/// The default root is under Application Support on iOS (durable, not user-visible) and Caches on
/// macOS (models are re-downloadable, so purging is acceptable), both in a `MoonshineModels`
/// subdirectory. Each ``ModelSpec`` maps to a stable subdirectory so different models never clobber
/// one another and re-loading an already-downloaded model is instant.
@available(iOS 15.0, macOS 12.0, *)
public enum ModelCache {
    /// The root directory under which Moonshine stores downloaded models by default.
    public static func defaultRoot() -> URL {
        let fileManager = FileManager.default
        #if os(macOS)
        let searchPath: FileManager.SearchPathDirectory = .cachesDirectory
        #else
        let searchPath: FileManager.SearchPathDirectory = .applicationSupportDirectory
        #endif
        let base =
            (try? fileManager.url(
                for: searchPath, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fileManager.urls(for: searchPath, in: .userDomainMask)[0]
        return base.appendingPathComponent("MoonshineModels", isDirectory: true)
    }

    /// A stable, filesystem-safe subdirectory name uniquely identifying `spec`'s files.
    public static func key(for spec: ModelSpec) -> String {
        switch spec {
        case .stt(let language, let modelArch, let includeSpelling, let includeWordTimestamps):
            var parts = ["stt", language, modelArch.map { "\($0.rawValue)" } ?? "default"]
            if includeSpelling { parts.append("spelling") }
            if includeWordTimestamps { parts.append("wt") }
            return sanitize(parts.joined(separator: "-"))
        case .tts(let language, _):
            // Voices of a language share one G2P root (the manifest lays them out under distinct
            // key prefixes), so the directory is keyed by language only.
            return sanitize("tts-\(language)")
        case .embedding(let modelName, let variant):
            // Keeps the historical "intent-" prefix so caches downloaded by
            // earlier releases are still found rather than re-fetched.
            let parts = ["intent", modelName, variant ?? "default"]
            return sanitize(parts.joined(separator: "-"))
        case .g2p(let language):
            return sanitize("g2p-\(language)")
        case .diarization:
            return "diarization-community1"
        }
    }

    /// Returns (creating if needed) the directory that `spec`'s files should live in, under `root`
    /// (or ``defaultRoot()`` when `root` is nil).
    @discardableResult
    public static func directory(for spec: ModelSpec, under root: URL? = nil) throws -> URL {
        let base = root ?? defaultRoot()
        let directory = base.appendingPathComponent(key(for: spec), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Replaces characters that are awkward in path components with `-`.
    private static func sanitize(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let scalars = raw.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        return String(scalars)
    }
}
