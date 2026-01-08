import SwiftUI

struct TranscriptWindowView: View {
    @ObservedObject var document: TranscriptDocument
    @StateObject private var viewModel: TranscriptWindowViewModel
    @AppStorage("showTimestamps") private var showTimestamps = true
    @State private var editingTitle = false
    @State private var editedTitle: String = ""
    @FocusState private var isTitleFocused: Bool
    
    init(document: TranscriptDocument) {
        self.document = document
        _viewModel = StateObject(wrappedValue: TranscriptWindowViewModel(document: document))
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Title bar with editable title
            HStack {
                if editingTitle {
                    TextField("Title", text: $editedTitle)
                        .textFieldStyle(.plain)
                        .font(.headline)
                        .focused($isTitleFocused)
                        .onSubmit {
                            document.data.title = editedTitle
                            editingTitle = false
                            isTitleFocused = false
                        }
                        .onExitCommand {
                            editingTitle = false
                            editedTitle = document.data.title
                            isTitleFocused = false
                        }
                } else {
                    Text(document.data.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                        .onTapGesture(count: 2) {
                            editedTitle = document.data.title
                            editingTitle = true
                            isTitleFocused = true
                        }
                }
                Spacer()
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Record button
            HStack {
                Spacer()
                RecordButton(isRecording: viewModel.isRecording) {
                    viewModel.toggleRecording()
                }
                .keyboardShortcut("r", modifiers: .command)
                Spacer()
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Transcript content
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        if document.data.entries.isEmpty && !viewModel.isRecording {
                            Text("Click the record button to start transcribing")
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding()
                        }
                        
                        ForEach(document.data.entries) { entry in
                            TranscriptRowView(
                                entry: entry,
                                showTimestamp: showTimestamps,
                                onEdit: { newText in
                                    var updated = entry
                                    updated.text = newText
                                    document.updateEntry(updated)
                                },
                                onDelete: {
                                    document.deleteEntry(entry)
                                }
                            )
                            .id(entry.id)
                        }
                        
                        if viewModel.isRecording {
                            // Live updating entry
                            if let currentEntry = viewModel.currentEntry {
                                TranscriptRowView(
                                    entry: currentEntry,
                                    showTimestamp: showTimestamps,
                                    onEdit: nil,
                                    onDelete: nil
                                )
                                .id("current")
                            }
                        }
                    }
                    .padding()
                }
                .onChange(of: document.data.entries.count) { _ in
                    if let last = document.data.entries.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: viewModel.currentEntry?.id) { _ in
                    withAnimation {
                        proxy.scrollTo("current", anchor: .bottom)
                    }
                }
            }
        }
        .frame(minWidth: 600, minHeight: 400)
        .onAppear {
            viewModel.setup()
        }
        .onDisappear {
            viewModel.cleanup()
        }
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Circle()
                    .fill(isRecording ? Color.red : Color.gray)
                    .frame(width: 12, height: 12)
                Text(isRecording ? "Stop Recording" : "Start Recording")
                    .font(.headline)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(isRecording ? Color.red.opacity(0.1) : Color.blue.opacity(0.1))
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
}

struct TranscriptRowView: View {
    let entry: TranscriptEntry
    let showTimestamp: Bool
    let onEdit: ((String) -> Void)?
    let onDelete: (() -> Void)?
    
    @State private var isEditing = false
    @State private var editedText: String = ""
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if showTimestamp {
                Text(timeString(from: entry.timestamp))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(width: 80, alignment: .leading)
            }
            
            if isEditing && onEdit != nil {
                TextField("", text: $editedText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .onSubmit {
                        onEdit?(editedText)
                        isEditing = false
                    }
                    .onExitCommand {
                        isEditing = false
                        editedText = entry.text
                    }
            } else {
                Text(entry.text.isEmpty ? (entry.isNote ? "[Note]" : "") : entry.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        if entry.isComplete && onEdit != nil {
                            editedText = entry.text
                            isEditing = true
                        }
                    }
                    .contextMenu {
                        if onEdit != nil {
                            Button("Edit") {
                                editedText = entry.text
                                isEditing = true
                            }
                        }
                        if onDelete != nil {
                            Button("Delete", role: .destructive) {
                                onDelete?()
                            }
                        }
                    }
            }
        }
        .padding(.vertical, 4)
    }
    
    private func timeString(from date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }
}

