import Foundation

/// Errors thrown by ``AssetDownloader`` while resolving manifests or fetching files.
///
/// Engine load / inference failures continue to surface as ``MoonshineError``; this type is scoped
/// to the download step so applications can distinguish "couldn't fetch the model" (retry, check
/// connectivity, free up space) from "the model failed to load".
public enum AssetDownloadError: LocalizedError {
    /// The server returned a non-2xx status for `url`.
    case httpStatus(code: Int, url: URL)
    /// The response was missing or not an HTTP response.
    case badResponse(url: URL)
    /// A manifest string could not be parsed into the expected shape.
    case invalidManifest(detail: String)
    /// Not enough free space to safely write `needBytes` (only `availableBytes` free) for `url`.
    case insufficientSpace(needBytes: Int64, availableBytes: Int64, url: URL)
    /// Writing or atomically moving a downloaded file failed.
    case fileWrite(path: String, underlying: Error)
    /// The download was cancelled.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .httpStatus(let code, let url):
            return "HTTP \(code) fetching \(url.lastPathComponent)"
        case .badResponse(let url):
            return "Invalid response for \(url.lastPathComponent)"
        case .invalidManifest(let detail):
            return "Invalid download manifest: \(detail)"
        case .insufficientSpace(let need, let available, let url):
            return
                "Not enough disk space for \(url.lastPathComponent): need \(need) bytes, \(available) available"
        case .fileWrite(let path, let underlying):
            return "Failed to write \(path): \(underlying.localizedDescription)"
        case .cancelled:
            return "Download cancelled"
        }
    }
}
