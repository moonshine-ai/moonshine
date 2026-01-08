import SwiftUI

@main
struct OpenNoteTakerApp: App {
    @StateObject private var appState = AppState.shared
    
    var body: some Scene {
        DocumentGroup(newDocument: { TranscriptDocument() }) { file in
            TranscriptWindowView(document: file.document)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Transcript") {
                    appState.createNewDocument()
                }
                .keyboardShortcut("n", modifiers: .command)
            }
            
            CommandMenu("Transcript") {
                Button("New Note") {
                    appState.insertNoteInActiveWindow()
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])
                
                Divider()
                
                Button("Export to TXT...") {
                    appState.exportActiveDocument(format: .txt)
                }
                
                Button("Export to DOCX...") {
                    appState.exportActiveDocument(format: .docx)
                }
            }
        }
        
        Settings {
            SettingsView()
        }
    }
}

