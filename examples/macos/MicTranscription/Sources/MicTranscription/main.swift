import Foundation
import MoonshineVoice

/// Downloads the Medium Streaming English model on first run into the user
/// caches directory and returns the directory to load it from. Subsequent runs
/// reuse the cached files (``AssetDownloader`` skips ones already present).
func downloadDefaultModel() async throws -> String {
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    let root = caches.appendingPathComponent(
        "moonshine-models/medium-streaming-en", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    let downloader = AssetDownloader()
    let spec = ModelSpec.stt(language: "en", modelArch: .mediumStreaming)
    if !downloader.isModelPresent(root: root, spec: spec) {
        fputs("Downloading Medium Streaming English model (first run only)…\n", stderr)
    }
    _ = try await downloader.ensureModelPresent(root: root, spec: spec) { progress in
        let pct =
            progress.bytesTotal > 0
            ? Int(progress.bytesDownloaded * 100 / progress.bytesTotal) : 0
        fputs(
            "\r  \(progress.relativePath) [\(progress.fileIndex)/\(progress.totalFiles)] \(pct)%      ",
            stderr)
    }
    fputs("\n", stderr)
    return root.path
}

// MARK: - Main

func main() async {
    let arguments = CommandLine.arguments

    var modelPath: String? = nil
    var modelArch: ModelArch? = nil
    for i in 1..<arguments.count {
        let argument = arguments[i]
        if argument.starts(with: "--model-path") {
            modelPath = argument.split(separator: "=").last.map(String.init)
        } else if argument.starts(with: "--model-arch") {
            let parts = argument.split(separator: "=")
            if parts.count > 1 {
                modelArch = ModelArch(rawValue: UInt32(parts[1]) ?? 0)
            }
        }
    }

    if modelPath == nil || modelArch == nil {
        // Default to the Medium Streaming English model, downloaded on first run
        // into the user caches directory (and reused thereafter). Override with
        // --model-path=/path/to/model --model-arch=<int>.
        do {
            modelPath = try await downloadDefaultModel()
            modelArch = .mediumStreaming
        } catch {
            fputs("Error: failed to download the default model: \(error)\n", stderr)
            exit(1)
        }
    }

    let micTranscriber = try! MicTranscriber(modelPath: modelPath!, modelArch: modelArch!)
    defer { micTranscriber.close() }

    class TestListener: TranscriptEventListener {
        func onLineStarted(_ event: LineStarted) {
            print(
                String(
                    format: "%.2fs: Line started: %@",
                    event.line.startTime, event.line.text))
        }

        func onLineTextChanged(_ event: LineTextChanged) {
            print(
                String(
                    format: "%.2fs: Line text changed: %@",
                    event.line.startTime, event.line.text))
        }

        func onLineCompleted(_ event: LineCompleted) {
            print(
                String(
                    format: "%.2fs: Line completed: %@",
                    event.line.startTime, event.line.text))
        }
    }

    let listener = TestListener()
    micTranscriber.addListener(listener)

    print("Listening to the microphone, press Ctrl+C to stop...")

    try! micTranscriber.start()

    while true {
        try! await Task.sleep(for: .seconds(1))
    }

    try! micTranscriber.stop()
}

await main()
