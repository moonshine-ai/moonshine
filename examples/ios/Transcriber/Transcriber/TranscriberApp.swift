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
                // Download the Medium Streaming English model on first run (into a
                // managed cache directory, reused thereafter) and construct the
                // transcriber in one call. Nothing is bundled in the app package.
                do {
                    let transcriber = try await MicTranscriber.load(
                        language: "en",
                        modelArch: .mediumStreaming
                    ) { progress in
                        let pct =
                            progress.bytesTotal > 0
                            ? Int(progress.bytesDownloaded * 100 / progress.bytesTotal) : 0
                        Task { @MainActor in
                            status =
                                "Downloading model \(progress.fileIndex)/\(progress.totalFiles) (\(pct)%)…"
                        }
                    }

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
