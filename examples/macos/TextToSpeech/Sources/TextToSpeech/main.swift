import AVFoundation
import Foundation
import MoonshineVoice

// MARK: - WAV Writing

func writeWav(path: String, samples: [Float], sampleRate: Int32) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample) / 8
    let blockAlign = numChannels * bitsPerSample / 8
    let dataSize = UInt32(samples.count) * UInt32(bitsPerSample / 8)
    let chunkSize: UInt32 = 36 + dataSize

    var data = Data()

    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: chunkSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)

    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    let fmtSize: UInt32 = 16
    let audioFormat: UInt16 = 1  // PCM
    data.append(contentsOf: withUnsafeBytes(of: fmtSize.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: audioFormat.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    let sr = UInt32(sampleRate)
    data.append(contentsOf: withUnsafeBytes(of: sr.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })
    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        let pcm = Int16(clamped * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: pcm.littleEndian) { Array($0) })
    }

    try data.write(to: url)
}

// MARK: - Command Line Argument Parsing

struct Arguments {
    var assetRoot: String = ""
    var language: String = "en_us"
    var voice: String? = nil
    var text: String = "Hello! This is a test of the Moonshine text to speech."
    var speed: String? = nil
    var device: String? = nil
    var output: String? = nil
    var listDevices: Bool = false
    var listVoices: Bool = false
}

func printUsage() {
    print("""
        Text-to-speech example for Moonshine Voice

        Usage: TextToSpeech [options]

        Options:
          --asset-root, -r PATH     Path to TTS assets directory (default: auto-detect)
          --language, -l LANG       Language tag, e.g. en_us, de, fr (default: en_us)
          --voice, -v VOICE         Voice ID, e.g. kokoro_af_heart (default: engine default)
          --text, -t TEXT           Text to synthesize (default: greeting)
          --speed, -s SPEED         Speech rate multiplier, e.g. 1.5 (default: 1.0)
          --device, -d DEVICE       Audio output device index or name substring
          --output, -o PATH         Write WAV file instead of playing audio
          --list-devices            List available audio output devices and exit
          --list-voices             List available voices for the language and exit
          --help, -h                Show this help message
        """)
}

func parseArguments() -> Arguments {
    var args = Arguments()
    let remaining = Array(CommandLine.arguments.dropFirst())
    var i = 0

    while i < remaining.count {
        let arg = remaining[i]
        switch arg {
        case "--asset-root", "-r":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.assetRoot = remaining[i]
        case "--language", "-l":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.language = remaining[i]
        case "--voice", "-v":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.voice = remaining[i]
        case "--text", "-t":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.text = remaining[i]
        case "--speed", "-s":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.speed = remaining[i]
        case "--device", "-d":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.device = remaining[i]
        case "--output", "-o":
            i += 1
            guard i < remaining.count else {
                fputs("Error: \(arg) requires a value\n", stderr)
                exit(1)
            }
            args.output = remaining[i]
        case "--list-devices":
            args.listDevices = true
        case "--list-voices":
            args.listVoices = true
        case "--help", "-h":
            printUsage()
            exit(0)
        default:
            fputs("Error: Unknown argument '\(arg)'\n", stderr)
            printUsage()
            exit(1)
        }
        i += 1
    }

    return args
}

// MARK: - Asset Root Resolution

func resolveAssetRoot(_ explicit: String, sourceFile: String = #filePath) -> String {
    if !explicit.isEmpty {
        return explicit
    }
    // Try the bundled tts-data directory next to the source file.
    // Source is at: examples/macos/TextToSpeech/Sources/TextToSpeech/main.swift
    // tts-data is at: examples/macos/TextToSpeech/tts-data/
    var dir = URL(fileURLWithPath: sourceFile).deletingLastPathComponent()
    for _ in 0..<10 {
        for name in ["tts-data", "core/moonshine-tts/data"] {
            let candidate = dir.appendingPathComponent(name)
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir),
                isDir.boolValue
            {
                return candidate.path
            }
        }
        dir = dir.deletingLastPathComponent()
    }
    // Fall back to relative paths from the current working directory
    for rel in [
        "tts-data",
        "../../../core/moonshine-tts/data",
        "../../core/moonshine-tts/data",
        "../core/moonshine-tts/data",
        "core/moonshine-tts/data",
    ] {
        let url = URL(fileURLWithPath: rel).standardized
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
            return url.path
        }
    }
    fputs(
        "Error: Could not find TTS assets. Use --asset-root to specify the path to tts-data/.\n",
        stderr)
    exit(1)
}

// MARK: - Device Resolution

func resolveDevice(_ spec: String) -> AudioDeviceID {
    let devices = MoonshineVoice.TextToSpeech.getAudioOutputDevices()
    if devices.isEmpty {
        fputs("Error: No audio output devices found.\n", stderr)
        exit(1)
    }

    // Try as numeric index first
    if let index = Int(spec) {
        // Match by position in our list
        if index >= 0 && index < devices.count {
            return devices[index].id
        }
        // Try matching by AudioDeviceID directly
        if let match = devices.first(where: { $0.id == AudioDeviceID(index) }) {
            return match.id
        }
        fputs("Error: Device index \(index) is out of range (0..\(devices.count - 1)).\n", stderr)
        fputs("Available devices:\n", stderr)
        for (i, dev) in devices.enumerated() {
            fputs("  [\(i)] \(dev.name) (id=\(dev.id))\n", stderr)
        }
        exit(1)
    }

    // Try as name substring (case-insensitive)
    let needle = spec.lowercased()
    if let match = devices.first(where: { $0.name.lowercased().contains(needle) }) {
        return match.id
    }

    fputs("Error: No output device name contains '\(spec)'.\n", stderr)
    fputs("Available devices:\n", stderr)
    for (i, dev) in devices.enumerated() {
        fputs("  [\(i)] \(dev.name) (id=\(dev.id))\n", stderr)
    }
    exit(1)
}

// MARK: - Main

func main() {
    let args = parseArguments()

    // Handle --list-devices
    if args.listDevices {
        let devices = MoonshineVoice.TextToSpeech.getAudioOutputDevices()
        if devices.isEmpty {
            print("No audio output devices found.")
        } else {
            print("Audio output devices:")
            for (i, dev) in devices.enumerated() {
                print("  [\(i)] \(dev.name) (id=\(dev.id))")
            }
        }
        return
    }

    let assetRoot = resolveAssetRoot(args.assetRoot)

    // Handle --list-voices
    if args.listVoices {
        do {
            let json = try MoonshineVoice.TextToSpeech.getVoices(
                languages: args.language,
                options: [TranscriberOption(name: "g2p_root", value: assetRoot)]
            )
            print("Voices for '\(args.language)':")
            print(json)
        } catch {
            fputs("Error listing voices: \(error)\n", stderr)
            exit(1)
        }
        return
    }

    // Create TTS synthesizer
    let tts: MoonshineVoice.TextToSpeech
    do {
        print("Creating TTS synthesizer for language '\(args.language)'...")
        tts = try MoonshineVoice.TextToSpeech(
            language: args.language,
            g2pRoot: assetRoot,
            voice: args.voice
        )
    } catch {
        fputs("Error: Failed to create TTS synthesizer: \(error)\n", stderr)
        exit(1)
    }
    defer { tts.close() }

    if let voice = args.voice {
        print("Voice: \(voice)")
    }

    // Build per-call options
    var synthOptions: [TranscriberOption]? = nil
    if let speed = args.speed {
        synthOptions = [TranscriberOption(name: "speed", value: speed)]
        print("Speed: \(speed)x")
    }

    if let outputPath = args.output {
        // Write to WAV file
        do {
            print("Synthesizing: \"\(args.text)\"")
            let result = try tts.synthesize(text: args.text, options: synthOptions)
            let duration = Double(result.samples.count) / Double(result.sampleRateHz)
            print(
                String(
                    format: "Got %d samples at %d Hz (%.2f seconds)",
                    result.samples.count, result.sampleRateHz, duration))

            try writeWav(path: outputPath, samples: result.samples, sampleRate: result.sampleRateHz)
            print("Written to \(outputPath)")
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    } else {
        // Play audio
        do {
            var deviceID: AudioDeviceID? = nil
            if let deviceSpec = args.device {
                deviceID = resolveDevice(deviceSpec)
                let devices = MoonshineVoice.TextToSpeech.getAudioOutputDevices()
                if let name = devices.first(where: { $0.id == deviceID })?.name {
                    print("Output device: \(name)")
                }
            }

            print("Synthesizing and playing: \"\(args.text)\"")
            try tts.say(args.text, device: deviceID, options: synthOptions)
            print("Playback complete.")
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }
}

main()
