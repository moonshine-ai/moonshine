import Foundation

// High-level "download the needed models, then construct" factories.
//
// These sit on top of the existing initializers and ``AssetDownloader``: each one derives the
// download ``ModelSpec`` from the *same* parameters used to construct the engine (so an app never
// duplicates `modelArch` / `variant` between a download call and a constructor), resolves a
// directory (a caller-supplied one, or a managed ``ModelCache`` location), downloads any missing
// files with optional progress reporting, and returns a ready-to-use engine.
//
// They are `async throws`: download failures surface as ``AssetDownloadError`` and load failures as
// ``MoonshineError``, so callers `try await` once and handle both in a single `catch`. All are
// gated to iOS 15 / macOS 12 because they rely on ``AssetDownloader``.

@available(iOS 15.0, macOS 12.0, *)
private func resolveDirectory(for spec: ModelSpec, override: URL?) throws -> URL {
    if let override = override {
        try FileManager.default.createDirectory(at: override, withIntermediateDirectories: true)
        return override
    }
    return try ModelCache.directory(for: spec)
}

@available(iOS 15.0, macOS 12.0, *)
extension Transcriber {
    /// Downloads the speech-to-text model for `language` (if not already present) and returns a
    /// ready ``Transcriber`` for file/batch transcription.
    public static func load(
        language: String,
        modelArch: ModelArch = .mediumStreaming,
        cacheDirectory: URL? = nil,
        includeSpelling: Bool = false,
        includeWordTimestamps: Bool = false,
        options: [TranscriberOption]? = nil,
        downloader: AssetDownloader = AssetDownloader(),
        onProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> Transcriber {
        let spec = ModelSpec.stt(
            language: language, modelArch: modelArch, includeSpelling: includeSpelling,
            includeWordTimestamps: includeWordTimestamps)
        let directory = try resolveDirectory(for: spec, override: cacheDirectory)
        _ = try await downloader.ensureModelPresent(root: directory, spec: spec, onProgress: onProgress)
        let resolvedOptions = try await withDiarizationModels(
            options, downloader: downloader, onProgress: onProgress)
        return try Transcriber(
            modelPath: directory.path, modelArch: modelArch, options: resolvedOptions)
    }
}

/// Whether `options` asks for speaker IDs, reading the value the way the C API does.
private func wantsSpeakerIds(_ options: [TranscriberOption]?) -> Bool {
    guard let option = options?.last(where: { $0.name == "identify_speakers" }) else {
        return false
    }
    let value = option.value.trimmingCharacters(in: .whitespaces).lowercased()
    return value == "true" || value == "1"
}

/// Downloads the diarization models and appends `diarization_model_dir` when the caller asked for
/// speaker IDs without saying where the models are. They are a download rather than part of the
/// library, and the transcriber refuses to construct without them.
@available(iOS 15.0, macOS 12.0, *)
func withDiarizationModels(
    _ options: [TranscriberOption]?,
    downloader: AssetDownloader,
    onProgress: (@Sendable (DownloadProgress) -> Void)?
) async throws -> [TranscriberOption]? {
    guard wantsSpeakerIds(options) else { return options }
    if options?.contains(where: { $0.name == "diarization_model_dir" }) == true { return options }
    let spec = ModelSpec.diarization
    let directory = try ModelCache.directory(for: spec)
    _ = try await downloader.ensureModelPresent(
        root: directory, spec: spec, onProgress: onProgress)
    return (options ?? [])
        + [TranscriberOption(name: "diarization_model_dir", value: directory.path)]
}

@available(iOS 15.0, macOS 12.0, *)
extension EmbeddingModel {
    /// Downloads the embedding model (if not already present) and returns a
    /// ready ``EmbeddingModel``.
    public static func load(
        modelName: String = "embeddinggemma-300m",
        modelArch: EmbeddingModelArch = .gemma300m,
        variant: String = "q4",
        cacheDirectory: URL? = nil,
        downloader: AssetDownloader = AssetDownloader(),
        onProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> EmbeddingModel {
        let spec = ModelSpec.embedding(modelName: modelName, variant: variant)
        let directory = try resolveDirectory(for: spec, override: cacheDirectory)
        _ = try await downloader.ensureModelPresent(root: directory, spec: spec, onProgress: onProgress)
        return try EmbeddingModel(
            modelPath: directory.path, modelArch: modelArch, modelVariant: variant)
    }
}

/// Namespace for cross-engine helpers.
@available(iOS 15.0, macOS 12.0, *)
public enum Moonshine {
    /// Downloads every model in `specs` (skipping files already present), reporting progress across
    /// the whole batch, and returns the directory each spec was written to.
    ///
    /// Useful for apps that need several models before they can start (e.g. an ASR model plus an
    /// embedding model): download them up front with a single progress handler, then
    /// construct each engine from the returned directory. Because ``ModelCache`` keys are stable,
    /// the per-engine `load` factories will find these files already present.
    @discardableResult
    public static func prepareModels(
        _ specs: [ModelSpec],
        into root: URL? = nil,
        downloader: AssetDownloader = AssetDownloader(),
        onProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> [ModelSpec: URL] {
        var result: [ModelSpec: URL] = [:]
        for spec in specs {
            let directory = try ModelCache.directory(for: spec, under: root)
            _ = try await downloader.ensureModelPresent(
                root: directory, spec: spec, onProgress: onProgress)
            result[spec] = directory
        }
        return result
    }
}
