import Foundation
import MoonshineVoice

/// Transcribe audio data offline without streaming
func transcribeWithoutStreaming(
    transcriber: Transcriber,
    audioData: [Float],
    sampleRate: Int32
) throws {
    let transcript = try transcriber.transcribeWithoutStreaming(
        audioData: audioData,
        sampleRate: sampleRate,
        flags: 0
    )

    for line in transcript.lines {
        let endTime = line.startTime + line.duration
        print(
            String(
                format: "Transcript: [%.2fs - %.2fs] %@",
                line.startTime, endTime, line.text))
    }
}

/// Example of streaming transcription
func transcribeWithStreaming(
    transcriber: Transcriber,
    audioData: [Float],
    sampleRate: Int32
) throws {

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
    try transcriber.removeAllListeners()
    try transcriber.addListener(listener)

    try transcriber.start()

    let chunkDuration: Double = 0.1
    let chunkSize = Int(chunkDuration * Double(sampleRate))

    var offset = 0
    while offset < audioData.count {
        let endOffset = min(offset + chunkSize, audioData.count)
        let chunk = Array(audioData[offset..<endOffset])
        try transcriber.addAudio(chunk, sampleRate: sampleRate)
        offset = endOffset
    }

    try transcriber.stop()
}

// MARK: - Command Line Argument Parsing

struct Arguments {
    var language: String = "en"
    var modelArch: ModelArch? = nil
    var inputFiles: [String] = []
}

func parseArguments() -> Arguments {
    var args = Arguments()
    var remainingArgs = CommandLine.arguments.dropFirst()

    while !remainingArgs.isEmpty {
        let arg = remainingArgs.first!
        remainingArgs = remainingArgs.dropFirst()

        if arg == "--language" || arg == "-l" {
            guard let value = remainingArgs.first else {
                fputs("Error: --language requires a value\n", stderr)
                exit(1)
            }
            args.language = value
            remainingArgs = remainingArgs.dropFirst()
        } else if arg == "--model-arch" || arg == "-m" {
            guard let value = remainingArgs.first else {
                fputs("Error: --model-arch requires a value\n", stderr)
                exit(1)
            }
            switch value.lowercased() {
            case "tiny":
                args.modelArch = .tiny
            case "base":
                args.modelArch = .base
            case "tiny-streaming", "tiny_streaming":
                args.modelArch = .tinyStreaming
            case "base-streaming", "base_streaming":
                args.modelArch = .baseStreaming
            case "small-streaming", "small_streaming":
                args.modelArch = .smallStreaming
            case "medium-streaming", "medium_streaming":
                args.modelArch = .mediumStreaming
            default:
                fputs(
                    "Error: Invalid model architecture '\(value)'. Must be one of: tiny, base, tiny-streaming, base-streaming, small-streaming, medium-streaming\n",
                    stderr)
                exit(1)
            }
            remainingArgs = remainingArgs.dropFirst()
        } else if arg == "--help" || arg == "-h" {
            print(
                """
                Basic transcription example for Moonshine Voice

                Usage: BasicTranscription [options] [input .wavfiles...]

                Options:
                  --language, -l LANGUAGE    Language to use for transcription (default: en)
                  --model-arch, -m ARCH       Model architecture: tiny, base, tiny-streaming, base-streaming, small-streaming, medium-streaming
                  --help, -h                  Show this help message
                """)
            exit(0)
        } else {
            // Treat as input file
            args.inputFiles.append(arg)
        }
    }

    return args
}

// MARK: - Main

func main() {
    var args = parseArguments()

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

    if args.inputFiles.isEmpty {
        let testAssetsPath = resourcePath.appending("/test-assets")
        args.inputFiles = [testAssetsPath.appending("/two_cities.wav")]
    }

    // Create transcriber
    let transcriber: Transcriber
    do {
        transcriber = try Transcriber(modelPath: modelPath, modelArch: modelArch)
    } catch {
        fputs("Error: Failed to load transcriber: \(error)\n", stderr)
        fputs("Model path: \(modelPath)\n", stderr)
        exit(1)
    }

    // Process each input file
    for inputFile in args.inputFiles {
        // Load WAV file
        let wavData: WAVData
        do {
            wavData = try loadWAVFile(inputFile)
        } catch {
            fputs("Error: Failed to load WAV file '\(inputFile)': \(error)\n", stderr)
            continue
        }

        print(String(repeating: "*", count: 80))
        print("Transcribing \(inputFile) offline without streaming...")
        do {
            try transcribeWithoutStreaming(
                transcriber: transcriber,
                audioData: wavData.audioData,
                sampleRate: Int32(wavData.sampleRate)
            )
        } catch {
            fputs("Error during non-streaming transcription: \(error)\n", stderr)
        }

        print(String(repeating: "*", count: 80))
        print("Transcribing \(inputFile) with streaming...")
        do {
            try transcribeWithStreaming(
                transcriber: transcriber,
                audioData: wavData.audioData,
                sampleRate: Int32(wavData.sampleRate)
            )
        } catch {
            fputs("Error during streaming transcription: \(error)\n", stderr)
        }
    }
}

main()
