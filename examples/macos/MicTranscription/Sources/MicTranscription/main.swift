import Foundation
import MoonshineVoice

// MARK: - Main

func main() async {
    var modelPath: String? = nil
    var modelArch: ModelArch? = nil
    for argument in CommandLine.arguments.dropFirst() {
        if argument.starts(with: "--model-path") {
            modelPath = argument.split(separator: "=").last.map(String.init)
        } else if argument.starts(with: "--model-arch") {
            let parts = argument.split(separator: "=")
            if parts.count > 1 {
                modelArch = ModelArch(rawValue: UInt32(parts[1]) ?? 0)
            }
        }
    }

    // Nothing here is required: with no configuration at all this transcribes
    // English from the default streaming model, downloaded on first run.
    let mic = MicTranscriber()
    if let modelArch { mic.modelArch(modelArch) }
    if let modelPath { mic.modelsFrom(URL(fileURLWithPath: modelPath)) }
    mic.onProgress { fraction, file in
        fputs("\r  \(file) \(Int(fraction * 100))%      ", stderr)
    }
    defer { mic.close() }

    do {
        try await mic.load()
        fputs("\n", stderr)
        print("Listening to the microphone, press Ctrl+C to stop...")
        try mic.start()
    } catch {
        fputs("Error: failed to start listening: \(error)\n", stderr)
        exit(1)
    }

    // `transcript` is an AsyncSequence of finished lines, so the interesting
    // part of the program is an ordinary for loop.
    do {
        for try await line in mic.transcript {
            print(String(format: "%.2fs: %@", line.startTime, line.text))
        }
    } catch {
        fputs("Error while transcribing: \(error)\n", stderr)
        exit(1)
    }
}

await main()
