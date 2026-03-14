#!/usr/bin/env swift
// Test the Swift API word timestamps by calling the C library directly.
// Build: swiftc -I core -L core/build -lmoonshine test_swift_api.swift -o test_swift_api

import Foundation

// Import the C API types directly
// transcript_word_t
struct CTranscriptWord {
    var text: UnsafePointer<CChar>?
    var start: Float
    var end: Float
    var confidence: Float
}

// transcript_line_t
struct CTranscriptLine {
    var text: UnsafePointer<CChar>?
    var audio_data: UnsafePointer<Float>?
    var audio_data_count: Int
    var start_time: Float
    var duration: Float
    var id: UInt64
    var is_complete: Int8
    var is_updated: Int8
    var is_new: Int8
    var has_text_changed: Int8
    var has_speaker_id: Int8
    var speaker_id: UInt64
    var speaker_index: UInt32
    var last_transcription_latency_ms: UInt32
    var words: UnsafePointer<CTranscriptWord>?
    var word_count: UInt64
}

// transcript_t
struct CTranscript {
    var lines: UnsafeMutablePointer<CTranscriptLine>?
    var line_count: UInt64
}

// transcriber_option_t
struct CTranscriberOption {
    var name: UnsafePointer<CChar>?
    var value: UnsafePointer<CChar>?
}

// C API function declarations
@_silgen_name("moonshine_get_version")
func moonshine_get_version() -> Int32

@_silgen_name("moonshine_load_transcriber_from_files")
func moonshine_load_transcriber_from_files(
    _ path: UnsafePointer<CChar>,
    _ model_arch: Int32,
    _ options: UnsafePointer<CTranscriberOption>?,
    _ options_count: Int32,
    _ version: Int32
) -> Int32

@_silgen_name("moonshine_transcribe_without_streaming")
func moonshine_transcribe_without_streaming(
    _ handle: Int32,
    _ audio_data: UnsafePointer<Float>,
    _ audio_length: UInt64,
    _ sample_rate: Int32,
    _ flags: UInt32,
    _ out_transcript: UnsafeMutablePointer<UnsafeMutablePointer<CTranscript>?>
) -> Int32

@_silgen_name("moonshine_free_transcriber")
func moonshine_free_transcriber(_ handle: Int32) -> Int32

// Paths
let scriptDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
let modelPath = scriptDir + "/test-assets/tiny-en"
let wavPath = scriptDir + "/test-assets/beckett.wav"

// Load WAV file
func loadWAV(_ path: String) -> [Float] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        print("FAIL: Cannot read \(path)")
        exit(1)
    }
    // Skip 44-byte WAV header
    let pcmData = data.subdata(in: 44..<data.count)
    let samples = pcmData.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [Float] in
        let int16Buffer = buffer.bindMemory(to: Int16.self)
        return int16Buffer.map { Float($0) / 32768.0 }
    }
    return samples
}

// Test
print("=== Swift API Test ===")
let version = moonshine_get_version()
print("Moonshine version: \(version)")

// Set up options
var optName1 = Array("word_timestamps".utf8CString)
var optVal1 = Array("true".utf8CString)
var optName2 = Array("identify_speakers".utf8CString)
var optVal2 = Array("false".utf8CString)

let options: [CTranscriberOption] = optName1.withUnsafeBufferPointer { n1 in
    optVal1.withUnsafeBufferPointer { v1 in
        optName2.withUnsafeBufferPointer { n2 in
            optVal2.withUnsafeBufferPointer { v2 in
                [
                    CTranscriberOption(name: n1.baseAddress, value: v1.baseAddress),
                    CTranscriberOption(name: n2.baseAddress, value: v2.baseAddress),
                ]
            }
        }
    }
}

print("Loading model from \(modelPath)...")
let handle = modelPath.withCString { pathPtr in
    options.withUnsafeBufferPointer { optsPtr in
        moonshine_load_transcriber_from_files(pathPtr, 0, optsPtr.baseAddress, 2, version)
    }
}

if handle < 0 {
    print("FAIL: Load failed with code \(handle)")
    exit(1)
}
print("Model loaded (handle=\(handle))")

// Load audio
var audio = loadWAV(wavPath)
let audioDuration = Float(audio.count) / 16000.0
print("\nAudio: \(wavPath) (\(String(format: "%.2f", audioDuration))s)")

// Transcribe
var transcriptPtr: UnsafeMutablePointer<CTranscript>? = nil
let err = moonshine_transcribe_without_streaming(
    handle, &audio, UInt64(audio.count), 16000, 0, &transcriptPtr
)

if err != 0 {
    print("FAIL: Transcribe failed with code \(err)")
    let _ = moonshine_free_transcriber(handle)
    exit(1)
}

guard let transcript = transcriptPtr?.pointee else {
    print("FAIL: No transcript")
    let _ = moonshine_free_transcriber(handle)
    exit(1)
}

print("\nTranscript lines: \(transcript.line_count)")

var totalWords = 0
var violations = 0

for i in 0..<Int(transcript.line_count) {
    let line = transcript.lines![i]
    let text = line.text.map { String(cString: $0) } ?? "<null>"
    print("\nLine \(i): \"\(text)\"")
    print("  start_time=\(String(format: "%.3f", line.start_time)), duration=\(String(format: "%.3f", line.duration))")
    print("  word_count=\(line.word_count)")

    if let words = line.words, line.word_count > 0 {
        var prevStart: Float = -1.0
        for j in 0..<Int(line.word_count) {
            let word = words[j]
            let wordText = word.text.map { String(cString: $0) } ?? "<null>"
            print("    [\(String(format: "%7.3f", word.start))s - \(String(format: "%7.3f", word.end))s] \(wordText.padding(toLength: 15, withPad: " ", startingAt: 0))  (conf: \(String(format: "%.2f", word.confidence)))")
            if word.start < prevStart {
                violations += 1
            }
            prevStart = word.start
            totalWords += 1
        }
    }
}

print("\n=== Swift API Test Results ===")
print("Total words: \(totalWords)")
print("Monotonicity violations: \(violations)")

let _ = moonshine_free_transcriber(handle)

if totalWords == 0 {
    print("FAIL: No words produced")
    exit(1)
}
if violations > 0 {
    print("FAIL: Monotonicity violations")
    exit(1)
}

print("PASS: \(totalWords) words with correct ordering via Swift API")
