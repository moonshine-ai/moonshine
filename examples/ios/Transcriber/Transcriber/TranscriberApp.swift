//
//  TranscriberApp.swift
//  Transcriber
//
//  Created by Pete Warden on 1/1/26.
//

import SwiftUI

import MoonshineVoice

@main
struct TranscriberApp: App {
    @State private var micTranscriber: MicTranscriber? = nil
    @State private var messages: [TranscriptLine] = []
    @State private var isRecording: Bool = false
    @State private var status: String = "Preparing…"
    @State private var isReady: Bool = false
    
    var body: some Scene {
        WindowGroup {
            ContentView(
                isRecording: $isRecording, messages: $messages,
                status: $status, isReady: isReady)
            .task {
                // Download the Medium Streaming English model on first run (into
                // the app's Application Support directory, reused thereafter),
                // then load it. Nothing is bundled in the app package.
                do {
                    let modelPath = try await downloadModel()
                    let transcriber = try MicTranscriber(
                        modelPath: modelPath, modelArch: ModelArch.mediumStreaming)

                    // Add event listeners
                    transcriber.addListener { event in
                        if event is LineStarted {
                            addNewMessage(event.line)
                        } else if event is LineTextChanged {
                            updateLatestMessage(event.line)
                        } else if event is LineCompleted {
                            if event.line.text.isEmpty {
                                messages.removeLast()
                            } else {
                                updateLatestMessage(event.line)
                            }
                        }
                    }
                    
                    // Store in @State after successful initialization
                    micTranscriber = transcriber
                    status = "Ready. Tap the mic to start."
                    isReady = true
                } catch {
                    status = "Error initializing transcriber: \(error.localizedDescription)"
                    print("Error initializing transcriber: \(error)")
                }
            }
        }
        .onChange(of: isRecording) { _, newIsRecording in
            handleRecordingChanged(newIsRecording)
        }
    }

    /// Downloads the Medium Streaming English model into Application Support on
    /// first run and returns the directory to load it from.
    func downloadModel() async throws -> String {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let root = support.appendingPathComponent(
            "moonshine-models/medium-streaming-en", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let downloader = AssetDownloader()
        let spec = ModelSpec.stt(language: "en", modelArch: .mediumStreaming)
        if !downloader.isModelPresent(root: root, spec: spec) {
            status = "Downloading Medium Streaming English model (first run only)…"
        }
        _ = try await downloader.ensureModelPresent(root: root, spec: spec)
        return root.path
    }
    
    func addNewMessage(_ message: TranscriptLine) {
        messages.append(message)
    }
    
    func updateLatestMessage(_ message: TranscriptLine) {
        messages[messages.count - 1] = message
    }

    func handleRecordingChanged(_ isRecording: Bool) {
        guard let micTranscriber = micTranscriber else { return }
        if isRecording {
            do {
                try micTranscriber.start()
            } catch {
                print("Error starting micTranscriber: \(error)")
            }
        } else {
            do {
                try micTranscriber.stop()
            } catch {
                print("Error stopping micTranscriber: \(error)")
            }
        }
    }
}
