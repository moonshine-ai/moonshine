import Foundation

/// Adapts the per-file ``DownloadProgress`` the downloader emits to the `0..1`
/// fraction the high-level engines expose, spreading each file's own progress
/// across its slot in the batch so the fraction rises monotonically even when a
/// model is split over several files.
func fractionReporter(
    _ handler: (@Sendable (Double, String) -> Void)?
) -> (@Sendable (DownloadProgress) -> Void)? {
    guard let handler else { return nil }
    return { update in
        let withinFile =
            update.bytesTotal > 0
            ? Double(update.bytesDownloaded) / Double(update.bytesTotal) : 0
        let overall =
            update.totalFiles > 0
            ? (Double(update.fileIndex - 1) + withinFile) / Double(update.totalFiles)
            : withinFile
        handler(min(1, max(0, overall)), update.relativePath)
    }
}
