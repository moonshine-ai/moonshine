import SwiftUI
import Combine
import AppKit

enum ExportFormat {
    case txt
    case docx
}

@MainActor
class AppState: ObservableObject {
    static let shared = AppState()
    
    @Published var activeRecordingWindow: TranscriptWindowViewModel?
    private var windows: [TranscriptWindowViewModel] = []
    
    private init() {}
    
    func createNewDocument() {
        // This will be handled by the DocumentGroup
        NSDocumentController.shared.newDocument(nil)
    }
    
    func registerWindow(_ viewModel: TranscriptWindowViewModel) {
        windows.append(viewModel)
    }
    
    func unregisterWindow(_ viewModel: TranscriptWindowViewModel) {
        windows.removeAll { $0 === viewModel }
    }
    
    func startRecording(in viewModel: TranscriptWindowViewModel) {
        // Pause any other active recording
        if let active = activeRecordingWindow, active !== viewModel {
            active.pauseRecording()
        }
        activeRecordingWindow = viewModel
    }
    
    func stopRecording(in viewModel: TranscriptWindowViewModel) {
        if activeRecordingWindow === viewModel {
            activeRecordingWindow = nil
        }
    }
    
    func insertNoteInActiveWindow() {
        // Find the key window and insert a note
        if let keyWindow = NSApplication.shared.keyWindow,
           let windowController = keyWindow.windowController,
           let document = windowController.document as? TranscriptDocument {
            document.insertNote()
        }
    }
    
    func exportActiveDocument(format: ExportFormat) {
        guard let keyWindow = NSApplication.shared.keyWindow,
              let windowController = keyWindow.windowController,
              let document = windowController.document as? TranscriptDocument else {
            return
        }
        
        let panel = NSSavePanel()
        panel.allowedContentTypes = format == .docx ? [.init(filenameExtension: "docx")!] : [.text]
        panel.nameFieldStringValue = document.data.title
        
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            
            do {
                if format == .docx {
                    try ExportManager.exportToDOCX(document: document, url: url)
                } else {
                    try ExportManager.exportToTXT(document: document, url: url)
                }
            } catch {
                let alert = NSAlert(error: error)
                alert.runModal()
            }
        }
    }
}

