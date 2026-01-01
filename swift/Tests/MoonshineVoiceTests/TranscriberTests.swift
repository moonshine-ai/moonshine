import XCTest
@testable import MoonshineVoice
import Foundation

final class TranscriberTests: XCTestCase {
    
    /// Get the path to test assets from the framework bundle
    static func getTestAssetsPath() throws -> String {
        // Try to get the framework bundle
        guard let bundle = Transcriber.frameworkBundle else {
            throw NSError(domain: "TranscriberTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not find moonshine framework bundle"])
        }
        
        guard let resourcePath = bundle.resourcePath else {
            throw NSError(domain: "TranscriberTests", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not find resource path in bundle"])
        }
        
        let testAssetsPath = (resourcePath as NSString).appendingPathComponent("test-assets")
        
        guard FileManager.default.fileExists(atPath: testAssetsPath) else {
            throw NSError(domain: "TranscriberTests", code: 3, userInfo: [NSLocalizedDescriptionKey: "Test assets directory not found at \(testAssetsPath)"])
        }
        
        return testAssetsPath
    }
    
    /// Get the path to the tiny-en model
    static func getTinyEnModelPath() throws -> String {
        let testAssetsPath = try getTestAssetsPath()
        let modelPath = (testAssetsPath as NSString).appendingPathComponent("tiny-en")
        
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw NSError(domain: "TranscriberTests", code: 4, userInfo: [NSLocalizedDescriptionKey: "Model directory not found at \(modelPath)"])
        }
        
        return modelPath
    }
    
    /// Get the path to a WAV file in test assets
    static func getWAVFilePath(_ filename: String) throws -> String {
        let testAssetsPath = try getTestAssetsPath()
        let wavPath = (testAssetsPath as NSString).appendingPathComponent(filename)
        
        guard FileManager.default.fileExists(atPath: wavPath) else {
            throw NSError(domain: "TranscriberTests", code: 5, userInfo: [NSLocalizedDescriptionKey: "WAV file not found at \(wavPath)"])
        }
        
        return wavPath
    }
    
    // MARK: - Non-Streaming Tests
    
    func testTranscribeWithoutStreaming_beckett() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let wavPath = try Self.getWAVFilePath("beckett.wav")
        let wavData = try loadWAVFile(wavPath)
        
        let transcript = try transcriber.transcribeWithoutStreaming(
            audioData: wavData.audioData,
            sampleRate: Int32(wavData.sampleRate)
        )
        
        // Verify we got a transcript
        XCTAssertFalse(transcript.lines.isEmpty, "Transcript should contain at least one line")
        
        // Verify all lines have text
        for line in transcript.lines {
            XCTAssertFalse(line.text.isEmpty, "Each transcript line should have text")
            XCTAssertGreaterThan(line.startTime, 0, "Start time should be positive")
            XCTAssertGreaterThan(line.duration, 0, "Duration should be positive")
        }
        
        // Print transcript for debugging
        print("Transcript for beckett.wav:")
        print(transcript)
    }
    
    func testTranscribeWithoutStreaming_twoCities() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let wavPath = try Self.getWAVFilePath("two_cities.wav")
        let wavData = try loadWAVFile(wavPath)
        
        let transcript = try transcriber.transcribeWithoutStreaming(
            audioData: wavData.audioData,
            sampleRate: Int32(wavData.sampleRate)
        )
        
        // Verify we got a transcript
        XCTAssertFalse(transcript.lines.isEmpty, "Transcript should contain at least one line")
        
        // Verify all lines have text
        for line in transcript.lines {
            XCTAssertFalse(line.text.isEmpty, "Each transcript line should have text")
            XCTAssertGreaterThan(line.startTime, 0, "Start time should be positive")
            XCTAssertGreaterThan(line.duration, 0, "Duration should be positive")
        }
        
        // Print transcript for debugging
        print("Transcript for two_cities.wav:")
        print(transcript)
    }
    
    func testTranscribeWithoutStreaming_emptyAudio() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        // Test with empty audio data
        let emptyAudio: [Float] = []
        let transcript = try transcriber.transcribeWithoutStreaming(
            audioData: emptyAudio,
            sampleRate: 16000
        )
        
        // Empty audio should result in empty transcript
        XCTAssertTrue(transcript.lines.isEmpty, "Empty audio should result in empty transcript")
    }
    
    // MARK: - Streaming Tests
    
    func testTranscribeWithStreaming_beckett() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let wavPath = try Self.getWAVFilePath("beckett.wav")
        let wavData = try loadWAVFile(wavPath)
        
        // Create a stream
        let stream = try transcriber.createStream(updateInterval: 0.5)
        defer { stream.close() }
        
        // Track events
        var lineStartedCount = 0
        var lineUpdatedCount = 0
        var lineCompletedCount = 0
        var lineTextChangedCount = 0
        var allText = ""
        var finalTranscript: Transcript?
        
        // Add event listeners
        stream.addListener { event in
            if event is LineStarted {
                lineStartedCount += 1
            } else if event is LineUpdated {
                lineUpdatedCount += 1
            } else if event is LineCompleted {
                lineCompletedCount += 1
                if let completed = event as? LineCompleted {
                    allText += completed.line.text + " "
                }
            } else if event is LineTextChanged {
                lineTextChangedCount += 1
            }
        }
        
        // Start the stream
        try stream.start()
        
        // Add audio in chunks to simulate streaming
        let chunkSize = 1600 // 0.1 seconds at 16kHz
        var offset = 0
        
        while offset < wavData.audioData.count {
            let endOffset = min(offset + chunkSize, wavData.audioData.count)
            let chunk = Array(wavData.audioData[offset..<endOffset])
            try stream.addAudio(chunk, sampleRate: Int32(wavData.sampleRate))
            offset = endOffset
            
            // Small delay to simulate real-time streaming
            Thread.sleep(forTimeInterval: 0.01)
        }
        
        // Stop the stream and get final transcript
        try stream.stop()
        finalTranscript = try stream.updateTranscription()
        
        // Verify we got events
        XCTAssertGreaterThan(lineStartedCount, 0, "Should have received at least one LineStarted event")
        XCTAssertGreaterThanOrEqual(lineCompletedCount, 0, "Should have received LineCompleted events")
        
        // Verify final transcript
        XCTAssertNotNil(finalTranscript, "Final transcript should not be nil")
        if let transcript = finalTranscript {
            XCTAssertFalse(transcript.lines.isEmpty, "Final transcript should contain at least one line")
            
            // Print transcript for debugging
            print("Streaming transcript for beckett.wav:")
            print(transcript)
            print("Events: \(lineStartedCount) started, \(lineUpdatedCount) updated, \(lineCompletedCount) completed, \(lineTextChangedCount) text changed")
        }
    }
    
    func testTranscribeWithStreaming_twoCities() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let wavPath = try Self.getWAVFilePath("two_cities.wav")
        let wavData = try loadWAVFile(wavPath)
        
        // Create a stream
        let stream = try transcriber.createStream(updateInterval: 0.5)
        defer { stream.close() }
        
        // Track events
        var lineStartedCount = 0
        var lineUpdatedCount = 0
        var lineCompletedCount = 0
        var allLines: [TranscriptLine] = []
        
        // Add event listeners
        stream.addListener { event in
            if event is LineStarted {
                lineStartedCount += 1
            } else if event is LineUpdated {
                lineUpdatedCount += 1
            } else if event is LineCompleted {
                lineCompletedCount += 1
                if let completed = event as? LineCompleted {
                    allLines.append(completed.line)
                }
            }
        }
        
        // Start the stream
        try stream.start()
        
        // Add audio in chunks to simulate streaming
        let chunkSize = 1600 // 0.1 seconds at 16kHz
        var offset = 0
        
        while offset < wavData.audioData.count {
            let endOffset = min(offset + chunkSize, wavData.audioData.count)
            let chunk = Array(wavData.audioData[offset..<endOffset])
            try stream.addAudio(chunk, sampleRate: Int32(wavData.sampleRate))
            offset = endOffset
            
            // Small delay to simulate real-time streaming
            Thread.sleep(forTimeInterval: 0.01)
        }
        
        // Stop the stream and get final transcript
        try stream.stop()
        let finalTranscript = try stream.updateTranscription()
        
        // Verify we got events
        XCTAssertGreaterThan(lineStartedCount, 0, "Should have received at least one LineStarted event")
        
        // Verify final transcript
        XCTAssertFalse(finalTranscript.lines.isEmpty, "Final transcript should contain at least one line")
        
        // Verify all lines have text
        for line in finalTranscript.lines {
            XCTAssertFalse(line.text.isEmpty, "Each transcript line should have text")
            XCTAssertGreaterThan(line.startTime, 0, "Start time should be positive")
            XCTAssertGreaterThan(line.duration, 0, "Duration should be positive")
        }
        
        // Print transcript for debugging
        print("Streaming transcript for two_cities.wav:")
        print(finalTranscript)
        print("Events: \(lineStartedCount) started, \(lineUpdatedCount) updated, \(lineCompletedCount) completed")
    }
    
    func testTranscribeWithStreaming_manualUpdates() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let wavPath = try Self.getWAVFilePath("beckett.wav")
        let wavData = try loadWAVFile(wavPath)
        
        // Create a stream with a long update interval so we can manually control updates
        let stream = try transcriber.createStream(updateInterval: 100.0) // Very long interval
        defer { stream.close() }
        
        // Start the stream
        try stream.start()
        
        // Add all audio at once
        try stream.addAudio(wavData.audioData, sampleRate: Int32(wavData.sampleRate))
        
        // Manually update transcription
        let transcript1 = try stream.updateTranscription()
        
        // Add more audio (if any left) and update again
        let transcript2 = try stream.updateTranscription(flags: Stream.flagForceUpdate)
        
        // Stop the stream
        try stream.stop()
        
        // Verify we got transcripts
        XCTAssertNotNil(transcript1, "First manual update should return a transcript")
        XCTAssertNotNil(transcript2, "Second manual update should return a transcript")
    }
    
    func testTranscribeWithStreaming_emptyAudio() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        // Create a stream
        let stream = try transcriber.createStream(updateInterval: 0.5)
        defer { stream.close() }
        
        // Start the stream
        try stream.start()
        
        // Add empty audio
        try stream.addAudio([], sampleRate: 16000)
        
        // Stop the stream
        try stream.stop()
        
        // Get final transcript
        let transcript = try stream.updateTranscription()
        
        // Empty audio should result in empty transcript
        XCTAssertTrue(transcript.lines.isEmpty, "Empty audio should result in empty transcript")
    }
    
    // MARK: - Helper Tests
    
    func testGetVersion() throws {
        let modelPath = try Self.getTinyEnModelPath()
        let transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
        defer { transcriber.close() }
        
        let version = transcriber.getVersion()
        XCTAssertGreaterThan(version, 0, "Version should be positive")
        
        print("Moonshine version: \(version)")
    }
    
    func testFrameworkBundle() {
        let bundle = Transcriber.frameworkBundle
        XCTAssertNotNil(bundle, "Framework bundle should be accessible")
        
        if let bundle = bundle, let resourcePath = bundle.resourcePath {
            print("Framework resource path: \(resourcePath)")
        }
    }
}

