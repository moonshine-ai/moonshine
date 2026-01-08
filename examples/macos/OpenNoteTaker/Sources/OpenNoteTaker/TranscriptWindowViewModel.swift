import Foundation
import Combine
import MoonshineVoice
import AVFoundation
import ScreenCaptureKit

@MainActor
class TranscriptWindowViewModel: ObservableObject {
    @Published var isRecording = false
    @Published var currentEntry: TranscriptEntry?
    
    private let document: TranscriptDocument
    private var transcriber: Transcriber?
    private var stream: MoonshineVoice.Stream?
    private var audioCapture: AudioCapture?
    private var cancellables = Set<AnyCancellable>()
    private var currentEntryId: UUID?
    private var recordingStartTime: Date?
    private var isPaused = false
    
    init(document: TranscriptDocument) {
        self.document = document
    }
    
    func setup() {
        AppState.shared.registerWindow(self)
    }
    
    func cleanup() {
        stopRecording()
        AppState.shared.unregisterWindow(self)
    }
    
    func toggleRecording() {
        if isRecording {
            pauseRecording()
        } else if isPaused {
            resumeRecording()
        } else {
            startRecording()
        }
    }
    
    func startRecording() {
        guard !isRecording else { return }
        
        Task { @MainActor in
            do {
            // Load model
            guard let bundle = Transcriber.frameworkBundle else {
                print("Error: Could not find moonshine framework bundle")
                return
            }
            
            guard let resourcePath = bundle.resourcePath else {
                print("Error: Could not find resource path in bundle")
                return
            }
            
            let testAssetsPath = resourcePath.appending("/test-assets")
            let modelPath = testAssetsPath.appending("/tiny-en")
            
            transcriber = try Transcriber(modelPath: modelPath, modelArch: .tiny)
            stream = try transcriber!.createStream(updateInterval: 0.5)
            
            // Set up event listener
            let listener = TranscriptListener { [weak self] event in
                self?.handleTranscriptEvent(event)
            }
            stream!.addListener(listener)
            
            // Start audio capture
            audioCapture = AudioCapture()
            audioCapture!.onAudioData = { [weak self] audioData, sampleRate in
                guard let self = self, self.isRecording else { return }
                do {
                    try self.stream?.addAudio(audioData, sampleRate: Int32(sampleRate))
                } catch {
                    print("Error adding audio: \(error)")
                }
            }
            
            try await audioCapture!.start()
            try stream!.start()
            
                recordingStartTime = Date()
                isRecording = true
                isPaused = false
                AppState.shared.startRecording(in: self)
                
            } catch {
                print("Error starting recording: \(error)")
            }
        }
    }
    
    func pauseRecording() {
        guard isRecording else { return }
        
        isPaused = true
        isRecording = false
        
        do {
            try stream?.stop()
            audioCapture?.stop()
        } catch {
            print("Error stopping recording: \(error)")
        }
        
        AppState.shared.stopRecording(in: self)
    }
    
    func resumeRecording() {
        guard isPaused else { return }
        
        // Add resume note
        let resumeEntry = TranscriptEntry(
            text: "Resumed at \(timeString(from: Date()))",
            timestamp: Date(),
            isNote: true,
            isComplete: true
        )
        document.addEntry(resumeEntry)
        
        Task { @MainActor in
            do {
                try stream?.start()
                try await audioCapture?.start()
                isRecording = true
                isPaused = false
                AppState.shared.startRecording(in: self)
            } catch {
                print("Error resuming recording: \(error)")
            }
        }
    }
    
    func stopRecording() {
        if isRecording {
            pauseRecording()
        }
        
        stream?.close()
        transcriber?.close()
        stream = nil
        transcriber = nil
        audioCapture = nil
        currentEntry = nil
        currentEntryId = nil
    }
    
    private func handleTranscriptEvent(_ event: TranscriptEvent) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            switch event {
            case let event as LineStarted:
                let entry = TranscriptEntry(
                    text: event.line.text,
                    timestamp: self.recordingStartTime?.addingTimeInterval(TimeInterval(event.line.startTime)) ?? Date(),
                    isNote: false,
                    isComplete: false,
                    lineId: event.line.lineId
                )
                self.currentEntryId = entry.id
                self.currentEntry = entry
                
            case let event as LineTextChanged:
                if event.line.lineId == self.currentEntry?.lineId {
                    var updated = self.currentEntry!
                    updated.text = event.line.text
                    self.currentEntry = updated
                }
                
            case let event as LineCompleted:
                if event.line.lineId == self.currentEntry?.lineId {
                    var completed = self.currentEntry!
                    completed.text = event.line.text
                    completed.isComplete = true
                    self.document.addEntry(completed)
                    self.currentEntry = nil
                    self.currentEntryId = nil
                }
                
            default:
                break
            }
        }
    }
    
    private func timeString(from date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }
}

class TranscriptListener: TranscriptEventListener {
    let handler: (TranscriptEvent) -> Void
    
    init(handler: @escaping (TranscriptEvent) -> Void) {
        self.handler = handler
    }
    
    func onLineStarted(_ event: LineStarted) {
        handler(event)
    }
    
    func onLineTextChanged(_ event: LineTextChanged) {
        handler(event)
    }
    
    func onLineCompleted(_ event: LineCompleted) {
        handler(event)
    }
    
    func onLineUpdated(_ event: LineUpdated) {
        handler(event)
    }
    
    func onError(_ event: TranscriptError) {
        print("Transcription error: \(event.error)")
    }
}

