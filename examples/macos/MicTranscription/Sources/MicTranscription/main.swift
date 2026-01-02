import Foundation
import MoonshineVoice

// MARK: - Main

func main() async {
    guard let bundle = Transcriber.frameworkBundle else {
        fputs("Error: Could not find moonshine framework bundle\n", stderr)
        exit(1)
    }

    guard let resourcePath = bundle.resourcePath else {
        fputs("Error: Could not find resource path in bundle\n", stderr)
        exit(1)
    }
    let testAssetsPath = resourcePath.appending("/test-assets")
    let modelPath = testAssetsPath.appending("/tiny-en")
    let modelArch: ModelArch = .tiny
    let micTranscriber = try! MicTranscriber(modelPath: modelPath, modelArch: modelArch)
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
