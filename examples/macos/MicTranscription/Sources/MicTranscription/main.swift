import Foundation
import MoonshineVoice

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

    let micTranscriber: MicTranscriber
    do {
        if let modelPath = modelPath {
            // Load from an explicit on-disk model directory.
            micTranscriber = try MicTranscriber(
                modelPath: modelPath, modelArch: modelArch ?? .mediumStreaming)
        } else {
            // Default to the Medium Streaming English model: download it on first
            // run (into a managed cache directory, reused thereafter) and
            // construct the transcriber in one call. Override with
            // --model-path=/path/to/model --model-arch=<int>.
            micTranscriber = try await MicTranscriber.load(
                language: "en",
                modelArch: modelArch ?? .mediumStreaming
            ) { progress in
                let pct =
                    progress.bytesTotal > 0
                    ? Int(progress.bytesDownloaded * 100 / progress.bytesTotal) : 0
                fputs(
                    "\r  \(progress.relativePath) [\(progress.fileIndex)/\(progress.totalFiles)] \(pct)%      ",
                    stderr)
            }
            fputs("\n", stderr)
        }
    } catch {
        fputs("Error: failed to prepare the transcriber: \(error)\n", stderr)
        exit(1)
    }
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
