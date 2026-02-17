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

    if modelPath == nil || modelArch == nil {
        guard let bundle = Transcriber.frameworkBundle else {
            fputs("Error: Could not find moonshine framework bundle\n", stderr)
            exit(1)
        }

        guard let resourcePath = bundle.resourcePath else {
            fputs("Error: Could not find resource path in bundle\n", stderr)
            exit(1)
        }
        let testAssetsPath = resourcePath.appending("/test-assets")
        modelPath = testAssetsPath.appending("/tiny-en")
        modelArch = .tiny
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
