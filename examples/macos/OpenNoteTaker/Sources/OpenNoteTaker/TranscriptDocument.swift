import SwiftUI
import UniformTypeIdentifiers
import Combine
import AppKit

extension UTType {
    static let openNoteTaker = UTType(exportedAs: "com.moonshine.opennote", conformingTo: .json)
}

@MainActor
final class TranscriptDocument: @preconcurrency ReferenceFileDocument, ObservableObject, @unchecked Sendable {
    typealias Snapshot = TranscriptData
    
    nonisolated(unsafe) private var _data: TranscriptData
    var data: TranscriptData {
        get { _data }
        set {
            _data = newValue
            objectWillChange.send()
        }
    }
    
    nonisolated static var readableContentTypes: [UTType] { [.openNoteTaker, .json, .plainText] }
    nonisolated static var writableContentTypes: [UTType] { [.openNoteTaker, .json] }
    
    nonisolated init() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let defaultTitle = formatter.string(from: Date())
        
        let initialData = TranscriptData(
            title: defaultTitle,
            entries: [],
            createdAt: Date(),
            lastModified: Date()
        )
        self._data = initialData
    }
    
    nonisolated required init(configuration: ReadConfiguration) throws {
        guard let fileData = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        
        let initialData: TranscriptData
        if configuration.contentType == .openNoteTaker || configuration.contentType == .json {
            initialData = try JSONDecoder().decode(TranscriptData.self, from: fileData)
        } else if configuration.contentType == .plainText {
            // Try to parse as plain text
            let text = String(data: fileData, encoding: .utf8) ?? ""
            let lines = text.components(separatedBy: .newlines).filter { !$0.isEmpty }
            let entries = lines.enumerated().map { index, line in
                TranscriptEntry(
                    id: UUID(),
                    text: line,
                    timestamp: Date().addingTimeInterval(TimeInterval(index)),
                    isNote: false,
                    isComplete: true
                )
            }
            initialData = TranscriptData(
                title: configuration.file.filename ?? "Imported Transcript",
                entries: entries,
                createdAt: Date(),
                lastModified: Date()
            )
        } else {
            throw CocoaError(.fileReadUnsupportedScheme)
        }
        self._data = initialData
    }
    
    nonisolated func snapshot(contentType: UTType) throws -> TranscriptData {
        MainActor.assumeIsolated {
            data.lastModified = Date()
            return data
        }
    }
    
    nonisolated func fileWrapper(snapshot: TranscriptData, configuration: WriteConfiguration) throws -> FileWrapper {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(snapshot)
        return FileWrapper(regularFileWithContents: data)
    }
    
    func insertNote() {
        let entry = TranscriptEntry(
            id: UUID(),
            text: "",
            timestamp: Date(),
            isNote: true,
            isComplete: true
        )
        data.entries.append(entry)
        data.lastModified = Date()
    }
    
    func addEntry(_ entry: TranscriptEntry) {
        data.entries.append(entry)
        data.lastModified = Date()
    }
    
    func updateEntry(_ entry: TranscriptEntry) {
        if let index = data.entries.firstIndex(where: { $0.id == entry.id }) {
            var updated = entry
            // Preserve original timestamp and lineId if not provided
            if updated.timestamp == data.entries[index].timestamp {
                updated = TranscriptEntry(
                    id: entry.id,
                    text: entry.text,
                    timestamp: data.entries[index].timestamp,
                    isNote: entry.isNote,
                    isComplete: entry.isComplete,
                    lineId: data.entries[index].lineId
                )
            }
            data.entries[index] = updated
            data.lastModified = Date()
        }
    }
    
    func deleteEntry(_ entry: TranscriptEntry) {
        data.entries.removeAll { $0.id == entry.id }
        data.lastModified = Date()
    }
}

struct TranscriptData: Codable, Sendable {
    var title: String
    var entries: [TranscriptEntry]
    var createdAt: Date
    var lastModified: Date
}

struct TranscriptEntry: Identifiable, Codable, Equatable, Sendable {
    let id: UUID
    var text: String
    let timestamp: Date
    var isNote: Bool
    var isComplete: Bool
    var lineId: UInt64?
    
    init(id: UUID = UUID(), text: String, timestamp: Date, isNote: Bool = false, isComplete: Bool, lineId: UInt64? = nil) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.isNote = isNote
        self.isComplete = isComplete
        self.lineId = lineId
    }
    
    // Custom Equatable implementation to allow updating
    static func == (lhs: TranscriptEntry, rhs: TranscriptEntry) -> Bool {
        lhs.id == rhs.id
    }
}

