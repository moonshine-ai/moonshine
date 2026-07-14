import Foundation
import XCTest

@testable import MoonshineVoice

/// Exercises ``AssetDownloader`` end-to-end against a mocked ``URLProtocol`` so no real network or
/// CDN is required. The native dependency catalog still resolves the real file lists, so these
/// tests also guard that the STT/intent/TTS manifests stay parseable and non-empty.
@available(iOS 15.0, macOS 12.0, *)
final class AssetDownloaderTests: XCTestCase {

    private var tempRoot: URL!

    override func setUp() {
        super.setUp()
        MockURLProtocol.reset()
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("moonshine-download-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempRoot)
        MockURLProtocol.reset()
        super.tearDown()
    }

    private func makeDownloader() -> AssetDownloader {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return AssetDownloader(session: URLSession(configuration: configuration))
    }

    /// Body served for a given URL: deterministic and derived from the filename so tests can assert
    /// exact content without knowing the full manifest up front.
    private static func body(for url: URL) -> Data {
        return Data("CONTENT:\(url.lastPathComponent)".utf8)
    }

    /// A range-aware 200/206 handler used by most tests.
    private func installStandardHandler() {
        MockURLProtocol.requestHandler = { request in
            let url = request.url!
            let full = Self.body(for: url)
            if let range = request.value(forHTTPHeaderField: "Range"),
                let start = Self.parseRangeStart(range), start < full.count {
                let remainder = full.subdata(in: start..<full.count)
                let response = HTTPURLResponse(
                    url: url, statusCode: 206, httpVersion: "HTTP/1.1",
                    headerFields: [
                        "Content-Length": String(remainder.count),
                        "Content-Range": "bytes \(start)-\(full.count - 1)/\(full.count)",
                    ])!
                return (response, remainder)
            }
            let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Length": String(full.count)])!
            return (response, full)
        }
    }

    private static func parseRangeStart(_ header: String) -> Int? {
        // Expect "bytes=<start>-"
        guard let eq = header.firstIndex(of: "="), let dash = header.firstIndex(of: "-") else {
            return nil
        }
        let startString = header[header.index(after: eq)..<dash]
        return Int(startString)
    }

    private func assertAllFilesMatchBody(under root: URL, file: StaticString = #filePath, line: UInt = #line) throws {
        let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)!
        var count = 0
        for case let fileURL as URL in enumerator {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDir)
            if isDir.boolValue { continue }
            count += 1
            let data = try Data(contentsOf: fileURL)
            XCTAssertEqual(
                data, Self.body(for: fileURL),
                "unexpected content for \(fileURL.lastPathComponent)", file: file, line: line)
            XCTAssertFalse(
                fileURL.lastPathComponent.hasSuffix(".part"),
                "stale .part left behind: \(fileURL.lastPathComponent)", file: file, line: line)
        }
        XCTAssertGreaterThan(count, 0, "expected at least one downloaded file", file: file, line: line)
    }

    // MARK: - STT

    func testDownloadsSttModelIntoEmptyDirectory() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let spec = ModelSpec.stt(language: "en", modelArch: .tiny)

        XCTAssertFalse(downloader.isModelPresent(root: tempRoot, spec: spec))
        let returned = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        XCTAssertEqual(returned, tempRoot)

        try assertAllFilesMatchBody(under: tempRoot)
        XCTAssertTrue(downloader.isModelPresent(root: tempRoot, spec: spec))
        XCTAssertGreaterThan(MockURLProtocol.requestedURLs.count, 0)
    }

    func testSkipsAlreadyPresentFiles() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let spec = ModelSpec.stt(language: "en", modelArch: .tiny)

        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        MockURLProtocol.reset()
        installStandardHandler()

        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        XCTAssertEqual(
            MockURLProtocol.requestedURLs.count, 0,
            "no files should be re-downloaded when already present")
    }

    func testIncludeSpellingAddsFilesForEnglish() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let withoutSpelling = ModelSpec.stt(language: "en", modelArch: .tiny)
        let withSpelling = ModelSpec.stt(
            language: "en", modelArch: .tiny, includeSpelling: true)

        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: withoutSpelling)
        let baseCount = MockURLProtocol.requestedURLs.count

        let spellingRoot = tempRoot.appendingPathComponent("spelling")
        try FileManager.default.createDirectory(at: spellingRoot, withIntermediateDirectories: true)
        MockURLProtocol.reset()
        installStandardHandler()
        _ = try await downloader.ensureModelPresent(root: spellingRoot, spec: withSpelling)
        XCTAssertGreaterThan(
            MockURLProtocol.requestedURLs.count, baseCount,
            "include_spelling should add the spelling model files")
    }

    // MARK: - Intent / TTS

    func testDownloadsIntentModel() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let spec = ModelSpec.intent(variant: "q4")
        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        try assertAllFilesMatchBody(under: tempRoot)
        XCTAssertTrue(downloader.isModelPresent(root: tempRoot, spec: spec))
    }

    func testDownloadsTtsAssetsIntoNestedPaths() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let spec = ModelSpec.tts(language: "en_us")
        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        try assertAllFilesMatchBody(under: tempRoot)
    }

    // MARK: - Progress, errors, resume

    func testReportsProgress() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let collector = ProgressCollector()
        _ = try await downloader.ensureModelPresent(
            root: tempRoot, spec: .stt(language: "en", modelArch: .tiny)
        ) { progress in
            collector.record(progress)
        }
        let events = collector.events
        XCTAssertFalse(events.isEmpty)
        XCTAssertTrue(events.allSatisfy { $0.fileIndex >= 1 && $0.fileIndex <= $0.totalFiles })
    }

    func testHttpErrorSurfacesAsAssetDownloadError() async throws {
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: nil)!
            return (response, Data())
        }
        let downloader = makeDownloader()
        do {
            _ = try await downloader.ensureModelPresent(
                root: tempRoot, spec: .stt(language: "en", modelArch: .tiny))
            XCTFail("expected an HTTP error")
        } catch let error as AssetDownloadError {
            guard case .httpStatus(let code, _) = error else {
                return XCTFail("expected .httpStatus, got \(error)")
            }
            XCTAssertEqual(code, 404)
        }
    }

    func testResumesFromPartialDownload() async throws {
        installStandardHandler()
        let downloader = makeDownloader()
        let spec = ModelSpec.stt(language: "en", modelArch: .tiny)

        // First, download everything so we can pick a concrete file to interrupt.
        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)
        let enumerator = FileManager.default.enumerator(at: tempRoot, includingPropertiesForKeys: nil)!
        guard let target = (enumerator.compactMap { $0 as? URL }.first {
            !$0.hasDirectoryPath
        }) else {
            return XCTFail("no file downloaded")
        }
        let fullData = try Data(contentsOf: target)
        XCTAssertGreaterThan(fullData.count, 2)

        // Simulate an interrupted download: remove the final file, leave a half-written .part.
        try FileManager.default.removeItem(at: target)
        let partURL = target.appendingPathExtension("part")
        let half = fullData.prefix(fullData.count / 2)
        try Data(half).write(to: partURL)

        MockURLProtocol.reset()
        installStandardHandler()
        _ = try await downloader.ensureModelPresent(root: tempRoot, spec: spec)

        let restored = try Data(contentsOf: target)
        XCTAssertEqual(restored, fullData, "resumed download must reconstruct the full file")
        XCTAssertTrue(
            MockURLProtocol.requestedURLs.contains { $0.rangeHeader != nil },
            "resume should send a Range request")
    }
}

// MARK: - Test helpers

/// Thread-safe accumulator for progress callbacks (they may arrive off the test's actor).
private final class ProgressCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [DownloadProgress] = []

    func record(_ progress: DownloadProgress) {
        lock.lock()
        storage.append(progress)
        lock.unlock()
    }

    var events: [DownloadProgress] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

struct RecordedRequest {
    let url: URL
    let rangeHeader: String?
}

/// Minimal in-process HTTP stub. Register `requestHandler` to map a request to a response + body.
/// The static state is guarded by `lock`, so the concurrency-safety opt-out is sound here.
final class MockURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _handler:
        (@Sendable (URLRequest) -> (HTTPURLResponse, Data))?
    nonisolated(unsafe) private static var _requested: [RecordedRequest] = []

    static var requestHandler: (@Sendable (URLRequest) -> (HTTPURLResponse, Data))? {
        get {
            lock.lock(); defer { lock.unlock() }
            return _handler
        }
        set {
            lock.lock(); defer { lock.unlock() }
            _handler = newValue
        }
    }

    static var requestedURLs: [RecordedRequest] {
        lock.lock(); defer { lock.unlock() }
        return _requested
    }

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        _handler = nil
        _requested = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.lock()
        Self._requested.append(
            RecordedRequest(
                url: request.url!, rangeHeader: request.value(forHTTPHeaderField: "Range")))
        let handler = Self._handler
        Self.lock.unlock()

        guard let handler = handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let (response, data) = handler(request)
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
