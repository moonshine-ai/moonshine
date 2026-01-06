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
    
    var body: some Scene {
        WindowGroup {
            ContentView(isRecording: $isRecording, messages: $messages)
            .task {
                // Initialize micTranscriber when the app starts
                do {
                    guard let bundle = Transcriber.frameworkBundle else {
                        print("Error: Could not find moonshine framework bundle")
                        return
                    }
                    guard let resourcePath = bundle.resourcePath else {
                        print("Error: Could not find resource path in bundle")
                        return
                    }
                    let modelPath = resourcePath.appending("/models/base-en/")
                    if (!FileManager.default.fileExists(atPath: modelPath)) {
                        print("Error: Model path does not exist: \(modelPath)")
                        return
                    }
                    let transcriber = try MicTranscriber(modelPath: modelPath, modelArch: ModelArch.base)
                                            
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
                } catch {
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
