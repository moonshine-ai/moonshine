import Foundation
import AppKit

@MainActor
class ExportManager {
    static func exportToTXT(document: TranscriptDocument, url: URL) throws {
        var content = document.data.title + "\n\n"
        
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        
        for entry in document.data.entries {
            let timeString = formatter.string(from: entry.timestamp)
            if entry.isNote {
                content += "[\(timeString)] Note: \(entry.text)\n"
            } else {
                content += "[\(timeString)] \(entry.text)\n"
            }
        }
        
        try content.write(to: url, atomically: true, encoding: .utf8)
    }
    
    static func exportToDOCX(document: TranscriptDocument, url: URL) throws {
        // For DOCX, we'll create a simple XML-based document
        // This is a simplified version - for production, use a proper DOCX library
        
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        
        var paragraphs: [String] = []
        paragraphs.append("<w:p><w:r><w:t>\(escapeXML(document.data.title))</w:t></w:r></w:p>")
        paragraphs.append("<w:p></w:p>") // Empty line
        
        for entry in document.data.entries {
            let timeString = formatter.string(from: entry.timestamp)
            let prefix = entry.isNote ? "[\(timeString)] Note: " : "[\(timeString)] "
            paragraphs.append("<w:p><w:r><w:t>\(escapeXML(prefix + entry.text))</w:t></w:r></w:p>")
        }
        
        let content = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                \(paragraphs.joined(separator: "\n"))
            </w:body>
        </w:document>
        """
        
        // Create a proper DOCX file (ZIP archive with XML files)
        // For simplicity, we'll create a minimal DOCX structure
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        // Create [Content_Types].xml
        let contentTypes = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
            <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
            <Default Extension="xml" ContentType="application/xml"/>
            <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
        </Types>
        """
        try contentTypes.write(to: tempDir.appendingPathComponent("[Content_Types].xml"), atomically: true, encoding: .utf8)
        
        // Create word/document.xml
        let wordDir = tempDir.appendingPathComponent("word")
        try FileManager.default.createDirectory(at: wordDir, withIntermediateDirectories: true)
        try content.write(to: wordDir.appendingPathComponent("document.xml"), atomically: true, encoding: .utf8)
        
        // Create _rels/.rels
        let relsDir = tempDir.appendingPathComponent("_rels")
        try FileManager.default.createDirectory(at: relsDir, withIntermediateDirectories: true)
        let rels = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        </Relationships>
        """
        try rels.write(to: relsDir.appendingPathComponent(".rels"), atomically: true, encoding: .utf8)
        
        // Create ZIP archive
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.arguments = ["-r", "-q", url.path, "."]
        process.currentDirectoryPath = tempDir.path
        
        try process.run()
        process.waitUntilExit()
        
        // Clean up
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    private static func escapeXML(_ string: String) -> String {
        return string
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
    }
}


