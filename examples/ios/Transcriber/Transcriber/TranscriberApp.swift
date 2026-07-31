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
    @StateObject private var session = TranscriptionSession()

    var body: some Scene {
        WindowGroup {
            ContentView(session: session)
                .task { await session.prepare() }
        }
    }
}

/// Owns the transcriber and republishes what it hears on the main thread.
@MainActor
final class TranscriptionSession: ObservableObject {
    @Published var lines: [String] = []
    /// The line currently being spoken, still changing as more audio arrives.
    @Published var liveText: String = ""
    @Published var status: String = "Preparing…"
    @Published var isReady: Bool = false
    @Published var isRecording: Bool = false

    private var mic: MicTranscriber?

    /// Downloads the streaming English model on first run. No language, model,
    /// or microphone permission handling needed here.
    func prepare() async {
        guard mic == nil else { return }

        let transcriber = MicTranscriber()
            .onText { [weak self] text in
                Task { @MainActor in self?.liveText = text }
            }
            .onLine { [weak self] line in
                Task { @MainActor in
                    self?.liveText = ""
                    if !line.text.isEmpty { self?.lines.append(line.text) }
                }
            }
            .onError { [weak self] error in
                Task { @MainActor in self?.status = "Error: \(error.localizedDescription)" }
            }
            .onProgress { [weak self] fraction, file in
                let name = (file as NSString).lastPathComponent
                Task { @MainActor in
                    self?.status = "Downloading \(name) \(Int(fraction * 100))%…"
                }
            }

        do {
            try await transcriber.load()
            mic = transcriber
            status = "Ready. Tap the mic to start."
            isReady = true
        } catch {
            status = "Couldn't load the model: \(error.localizedDescription)"
        }
    }

    func toggleRecording() {
        guard let mic else { return }
        do {
            if isRecording {
                try mic.stop()
                isRecording = false
                status = "Ready. Tap the mic to start."
            } else {
                try mic.start()
                isRecording = true
                status = "Listening…"
            }
        } catch {
            isRecording = false
            status = "Microphone error: \(error.localizedDescription)"
        }
    }
}
