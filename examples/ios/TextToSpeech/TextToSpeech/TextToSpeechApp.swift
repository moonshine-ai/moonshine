import AVFoundation
import MoonshineVoice
import SwiftUI

/// Kokoro/Piper-supported languages with display names.
struct KokoroLanguage: Identifiable, Hashable {
    let id: String  // Moonshine language tag (e.g. "en_us")
    let displayName: String
    let voicePrefix: [String]  // e.g. ["af_", "am_"] for en_us

    func hash(into hasher: inout Hasher) { hasher.combine(id) }
    static func == (lhs: KokoroLanguage, rhs: KokoroLanguage) -> Bool {
        lhs.id == rhs.id
    }
}

/// Languages surfaced in the picker. Assets for anything beyond the bundled
/// en_us + `af_alloy` voice are fetched from the CDN on demand.
let kokoroLanguages: [KokoroLanguage] = [
    KokoroLanguage(id: "ar", displayName: "Arabic", voicePrefix: []),
    KokoroLanguage(id: "ca", displayName: "Catalan", voicePrefix: []),
    KokoroLanguage(id: "cs", displayName: "Czech", voicePrefix: []),
    KokoroLanguage(id: "da", displayName: "Danish", voicePrefix: []),
    KokoroLanguage(id: "de", displayName: "German", voicePrefix: []),
    KokoroLanguage(id: "el", displayName: "Greek", voicePrefix: []),
    KokoroLanguage(id: "en_gb", displayName: "English (UK)", voicePrefix: ["bf_", "bm_"]),
    KokoroLanguage(id: "en_us", displayName: "English (US)", voicePrefix: ["af_", "am_"]),
    KokoroLanguage(id: "es", displayName: "Spanish (ES)", voicePrefix: []),
    KokoroLanguage(id: "es_mx", displayName: "Spanish (MX)", voicePrefix: ["ef_", "em_"]),
    KokoroLanguage(id: "fa", displayName: "Persian", voicePrefix: []),
    KokoroLanguage(id: "fi", displayName: "Finnish", voicePrefix: []),
    KokoroLanguage(id: "fr", displayName: "French", voicePrefix: ["ff_"]),
    KokoroLanguage(id: "hi", displayName: "Hindi", voicePrefix: ["hf_", "hm_"]),
    KokoroLanguage(id: "hu", displayName: "Hungarian", voicePrefix: []),
    KokoroLanguage(id: "it", displayName: "Italian", voicePrefix: ["if_", "im_"]),
    KokoroLanguage(id: "ja", displayName: "Japanese", voicePrefix: ["jf_", "jm_"]),
    KokoroLanguage(id: "ka", displayName: "Georgian", voicePrefix: []),
    KokoroLanguage(id: "kk", displayName: "Kazakh", voicePrefix: []),
    KokoroLanguage(id: "lb", displayName: "Luxembourgish", voicePrefix: []),
    KokoroLanguage(id: "lv", displayName: "Latvian", voicePrefix: []),
    KokoroLanguage(id: "ml", displayName: "Malayalam", voicePrefix: []),
    KokoroLanguage(id: "nb", displayName: "Norwegian", voicePrefix: []),
    KokoroLanguage(id: "nl", displayName: "Dutch", voicePrefix: []),
    KokoroLanguage(id: "pl", displayName: "Polish", voicePrefix: []),
    KokoroLanguage(id: "pt", displayName: "Portuguese (PT)", voicePrefix: []),
    KokoroLanguage(id: "pt_br", displayName: "Portuguese (BR)", voicePrefix: ["pf_", "pm_"]),
    KokoroLanguage(id: "ro", displayName: "Romanian", voicePrefix: []),
    KokoroLanguage(id: "ru", displayName: "Russian", voicePrefix: []),
    KokoroLanguage(id: "sk", displayName: "Slovak", voicePrefix: []),
    KokoroLanguage(id: "sl", displayName: "Slovenian", voicePrefix: []),
    KokoroLanguage(id: "sr", displayName: "Serbian", voicePrefix: []),
    KokoroLanguage(id: "sv", displayName: "Swedish", voicePrefix: []),
    KokoroLanguage(id: "sw", displayName: "Swahili", voicePrefix: []),
    KokoroLanguage(id: "tr", displayName: "Turkish", voicePrefix: []),
    KokoroLanguage(id: "uk", displayName: "Ukrainian", voicePrefix: []),
    KokoroLanguage(id: "vi", displayName: "Vietnamese", voicePrefix: []),
    KokoroLanguage(id: "zh_hans", displayName: "Chinese (Mandarin)", voicePrefix: ["zf_", "zm_"]),
]

/// A single voice entry surfaced in the UI.
struct TtsVoice: Identifiable, Hashable {
    let id: String  // e.g. "kokoro_af_heart" or "piper_en_US-ryan-low"
    let displayName: String  // e.g. "Heart (Female) · Kokoro"
    let needsDownload: Bool  // true when the voice or its language assets aren't on disk yet

    func hash(into hasher: inout Hasher) { hasher.combine(id) }
    static func == (lhs: TtsVoice, rhs: TtsVoice) -> Bool { lhs.id == rhs.id }
}

/// Lightweight description of an ongoing asset download, surfaced to SwiftUI.
struct DownloadStatus: Equatable {
    let fileName: String
    let fileIndex: Int
    let totalFiles: Int
    let fraction: Double?  // nil when server didn't report Content-Length
}

/// Observable model that owns the TTS synthesizer, mirrors Android's on-demand
/// bootstrap flow, and surfaces download progress for SwiftUI.
@MainActor
class TTSModel: ObservableObject {
    @Published var selectedLanguage: KokoroLanguage =
        kokoroLanguages.first(where: { $0.id == "en_us" }) ?? kokoroLanguages[0]
    @Published var availableVoices: [TtsVoice] = []
    @Published var selectedVoice: TtsVoice? = nil
    @Published var isSpeaking: Bool = false
    @Published var isReady: Bool = false
    @Published var isDownloading: Bool = false
    @Published var isBootstrapping: Bool = true
    @Published var downloadStatus: DownloadStatus? = nil
    @Published var errorMessage: String? = nil

    private var tts: MoonshineVoice.TextToSpeech? = nil
    private var g2pRoot: URL?

    func initialize() {
        #if os(iOS)
            do {
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.playback, mode: .default)
                try session.setActive(true)
            } catch {
                print("Audio session setup warning: \(error)")
            }
        #endif

        Task { await bootstrap() }
    }

    // MARK: - Bootstrap / asset staging

    private func bootstrap() async {
        isBootstrapping = true
        errorMessage = nil
        do {
            let root = try prepareG2PRoot()
            g2pRoot = root
            // Fetch whatever the default en_us + af_alloy bundle needs beyond the
            // Kokoro model/voice we ship inside the app.
            try await downloadAssets(
                g2pRoot: root, language: selectedLanguage.id, voice: "kokoro_af_alloy")
            try createSynthesizer(voice: "kokoro_af_alloy")
            refreshVoices(preferVoice: "kokoro_af_alloy")
            isReady = true
        } catch {
            errorMessage = "Failed to initialize TTS: \(error.localizedDescription)"
        }
        isBootstrapping = false
    }

    /// Copies the read-only `tts-data/` bundle into a writable Application Support
    /// location on first launch, returning the writable URL.
    private func prepareG2PRoot() throws -> URL {
        let fm = FileManager.default
        let support = try fm.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let root = support.appendingPathComponent("tts-data", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        guard let bundled = Bundle.main.url(forResource: "tts-data", withExtension: nil) else {
            return root
        }
        try copyBundledTree(from: bundled, to: root, fm: fm)
        return root
    }

    /// Copy each file from the bundled read-only tree into the writable root,
    /// skipping files that already exist (cached from a previous launch).
    private func copyBundledTree(from src: URL, to dst: URL, fm: FileManager) throws {
        guard
            let enumerator = fm.enumerator(
                at: src, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return }
        for case let url as URL in enumerator {
            let rel = url.path.replacingOccurrences(of: src.path + "/", with: "")
            let target = dst.appendingPathComponent(rel)
            let isDir =
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isDir {
                try fm.createDirectory(at: target, withIntermediateDirectories: true)
            } else if !fm.fileExists(atPath: target.path) {
                try fm.createDirectory(
                    at: target.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try fm.copyItem(at: url, to: target)
            }
        }
    }

    // MARK: - Public language / voice switching

    func changeLanguage(_ lang: KokoroLanguage) {
        guard lang.id != selectedLanguage.id else { return }
        selectedLanguage = lang
        selectedVoice = nil
        refreshVoices(preferVoice: nil)
        // Attempt to land on a voice that is already downloaded for this language.
        let preferred =
            availableVoices.first(where: { !$0.needsDownload })
            ?? availableVoices.first
        if let preferred = preferred {
            changeVoice(preferred)
        } else {
            Task { await ensureAssetsThenRecreate(voice: nil) }
        }
    }

    func changeVoice(_ voice: TtsVoice?) {
        selectedVoice = voice
        Task { await ensureAssetsThenRecreate(voice: voice?.id) }
    }

    func speak(_ text: String) {
        guard let tts = tts, !text.isEmpty else { return }
        isSpeaking = true
        let currentTts = tts
        currentTts.say(text)
        Task.detached { [weak self] in
            currentTts.wait()
            await MainActor.run { [weak self] in
                self?.isSpeaking = false
            }
        }
    }

    // MARK: - Download + synthesizer refresh

    private func ensureAssetsThenRecreate(voice: String?) async {
        guard let root = g2pRoot else { return }
        errorMessage = nil

        let needsDownload = selectedVoice?.needsDownload == true
        if needsDownload {
            isDownloading = true
            do {
                try await downloadAssets(
                    g2pRoot: root,
                    language: selectedLanguage.id,
                    voice: voice)
                refreshVoices(preferVoice: voice)
            } catch {
                errorMessage = "Download failed: \(error.localizedDescription)"
                isDownloading = false
                return
            }
            isDownloading = false
        }

        do {
            try createSynthesizer(voice: voice)
            // Refresh voice states once everything is on disk.
            refreshVoices(preferVoice: voice)
        } catch {
            errorMessage = "Failed to create synthesizer: \(error.localizedDescription)"
        }
    }

    private func downloadAssets(
        g2pRoot: URL, language: String, voice: String?
    ) async throws {
        let statusCallback: @Sendable (TtsAssetDownloader.Progress) -> Void = { [weak self] p in
            let fraction: Double? = p.bytesTotal > 0
                ? min(1.0, Double(p.bytesDownloaded) / Double(p.bytesTotal)) : nil
            let fileName = (p.key as NSString).lastPathComponent
            let snapshot = DownloadStatus(
                fileName: fileName,
                fileIndex: p.fileIndex,
                totalFiles: p.totalFiles,
                fraction: fraction
            )
            Task { @MainActor in
                self?.downloadStatus = snapshot
            }
        }
        defer {
            Task { @MainActor in self.downloadStatus = nil }
        }
        try await TtsAssetDownloader.ensureAssetsPresent(
            g2pRoot: g2pRoot,
            language: language,
            voice: voice,
            onProgress: statusCallback
        )
    }

    private func createSynthesizer(voice: String?) throws {
        guard let root = g2pRoot else { return }
        tts?.close()
        tts = nil
        isReady = false
        tts = try MoonshineVoice.TextToSpeech(
            language: selectedLanguage.id,
            g2pRoot: root.path,
            voice: voice
        )
        isReady = true
    }

    // MARK: - Voice list

    private func refreshVoices(preferVoice: String?) {
        guard let root = g2pRoot else { return }
        do {
            let json = try MoonshineVoice.TextToSpeech.getVoices(
                languages: selectedLanguage.id,
                options: [TranscriberOption(name: "g2p_root", value: root.path)]
            )
            availableVoices = parseVoices(json: json, language: selectedLanguage)
        } catch {
            availableVoices = []
        }
        if let preferVoice = preferVoice,
            let match = availableVoices.first(where: { $0.id == preferVoice })
        {
            selectedVoice = match
        } else if selectedVoice == nil
            || !availableVoices.contains(where: { $0.id == selectedVoice?.id })
        {
            selectedVoice =
                availableVoices.first(where: { !$0.needsDownload })
                ?? availableVoices.first
        }
    }

    private func parseVoices(json: String, language: KokoroLanguage) -> [TtsVoice] {
        guard let data = json.data(using: .utf8),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let langVoices = dict[language.id] as? [[String: String]]
        else { return [] }

        let voices = langVoices.compactMap { entry -> TtsVoice? in
            guard let voiceId = entry["id"], let state = entry["state"]
            else { return nil }
            let isKokoro = voiceId.hasPrefix("kokoro_")
            let isPiper = voiceId.hasPrefix("piper_")
            guard isKokoro || isPiper else { return nil }
            let needsDownload = (state != "found")
            let engineLabel = isKokoro ? "Kokoro" : "Piper"
            let base =
                isKokoro
                ? formatKokoroName(String(voiceId.dropFirst("kokoro_".count)))
                : formatPiperName(String(voiceId.dropFirst("piper_".count)))
            var displayName = "\(base) · \(engineLabel)"
            if needsDownload { displayName += " — tap to download" }
            return TtsVoice(id: voiceId, displayName: displayName, needsDownload: needsDownload)
        }

        // Stable sort: found voices first within each engine.
        return voices.sorted { lhs, rhs in
            if lhs.needsDownload != rhs.needsDownload { return !lhs.needsDownload }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName)
                == .orderedAscending
        }
    }

    private func formatKokoroName(_ shortId: String) -> String {
        // shortId like "af_heart" -> "Heart (Female)"
        let parts = shortId.split(separator: "_", maxSplits: 1)
        guard parts.count == 2 else { return shortId }

        let prefix = String(parts[0])
        let name = String(parts[1]).capitalized

        let gender: String
        if prefix.hasSuffix("f") {
            gender = "Female"
        } else if prefix.hasSuffix("m") {
            gender = "Male"
        } else {
            gender = ""
        }
        return gender.isEmpty ? name : "\(name) (\(gender))"
    }

    private func formatPiperName(_ shortId: String) -> String {
        // shortId like "en_US-ryan-low" -> "Ryan Low (En US)"
        let parts = shortId.split(separator: "-", maxSplits: 1)
        guard parts.count == 2 else {
            return shortId.replacingOccurrences(of: "_", with: " ").capitalized
        }
        let langTag = parts[0].replacingOccurrences(of: "_", with: " ")
        let stem = parts[1]
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .capitalized
        return "\(stem) (\(langTag.capitalized))"
    }
}

@main
struct TextToSpeechApp: App {
    @StateObject private var model = TTSModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task {
                    model.initialize()
                }
        }
    }
}
