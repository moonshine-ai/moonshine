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
    let fraction: Double
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

    // MARK: - Bootstrap

    private func bootstrap() async {
        isBootstrapping = true
        errorMessage = nil
        await createSynthesizer(voice: "kokoro_af_alloy")
        isBootstrapping = false
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
            Task { await createSynthesizer(voice: nil) }
        }
    }

    func changeVoice(_ voice: TtsVoice?) {
        selectedVoice = voice
        Task { await createSynthesizer(voice: voice?.id) }
    }

    func speak(_ text: String) {
        guard let tts = tts, !text.isEmpty else { return }
        isSpeaking = true
        Task {
            // say returns once the audio has finished playing.
            do {
                try await tts.say(text)
            } catch {
                errorMessage = "Playback failed: \(error.localizedDescription)"
            }
            isSpeaking = false
        }
    }

    // MARK: - Synthesizer refresh

    /// Builds a synthesizer for `voice`, downloading whatever it needs. The
    /// engine reports its own progress as a `0..1` fraction, so the app no
    /// longer decides which files to fetch or where to put them.
    private func createSynthesizer(voice: String?) async {
        errorMessage = nil
        isReady = false
        isDownloading = selectedVoice?.needsDownload != false

        let instance = MoonshineVoice.TextToSpeech()
            .language(selectedLanguage.id)
            .onProgress { [weak self] fraction, file in
                let snapshot = DownloadStatus(
                    fileName: (file as NSString).lastPathComponent, fraction: fraction)
                Task { @MainActor in self?.downloadStatus = snapshot }
            }
        if let voice { instance.voice(voice) }

        do {
            try await instance.load()
            tts?.close()
            tts = instance
            isReady = true
        } catch {
            instance.close()
            errorMessage = "Failed to create synthesizer: \(error.localizedDescription)"
        }
        downloadStatus = nil
        isDownloading = false
        // Refresh voice states once everything is on disk.
        refreshVoices(preferVoice: voice)
    }

    // MARK: - Voice list

    /// Lists the voices for the current language, marking the ones already on
    /// disk. All voices of a language share one cache directory, so pointing
    /// `g2p_root` at it is enough to get accurate availability.
    private func refreshVoices(preferVoice: String?) {
        let cache = try? ModelCache.directory(
            for: .tts(language: selectedLanguage.id, voice: nil))
        do {
            let json = try MoonshineVoice.TextToSpeech.getVoices(
                languages: selectedLanguage.id,
                options: cache.map { [TranscriberOption(name: "g2p_root", value: $0.path)] }
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
            if needsDownload { displayName += " (tap to download)" }
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
