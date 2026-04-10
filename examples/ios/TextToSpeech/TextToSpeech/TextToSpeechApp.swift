import SwiftUI
import AVFoundation
import MoonshineVoice

/// Kokoro-supported languages with display names.
struct KokoroLanguage: Identifiable, Hashable {
    let id: String       // Moonshine language tag (e.g. "en_us")
    let displayName: String
    let voicePrefix: [String]  // e.g. ["af_", "am_"] for en_us

    func hash(into hasher: inout Hasher) { hasher.combine(id) }
    static func == (lhs: KokoroLanguage, rhs: KokoroLanguage) -> Bool { lhs.id == rhs.id }
}

let kokoroLanguages: [KokoroLanguage] = [
    KokoroLanguage(id: "en_us", displayName: "English (US)", voicePrefix: ["af_", "am_"]),
    KokoroLanguage(id: "en_gb", displayName: "English (UK)", voicePrefix: ["bf_", "bm_"]),
    KokoroLanguage(id: "es_mx", displayName: "Spanish", voicePrefix: ["ef_", "em_"]),
    KokoroLanguage(id: "fr", displayName: "French", voicePrefix: ["ff_"]),
    KokoroLanguage(id: "hi", displayName: "Hindi", voicePrefix: ["hf_", "hm_"]),
    KokoroLanguage(id: "it", displayName: "Italian", voicePrefix: ["if_", "im_"]),
    KokoroLanguage(id: "ja", displayName: "Japanese", voicePrefix: ["jf_", "jm_"]),
    KokoroLanguage(id: "pt_br", displayName: "Portuguese (BR)", voicePrefix: ["pf_", "pm_"]),
    KokoroLanguage(id: "zh_hans", displayName: "Chinese (Mandarin)", voicePrefix: ["zf_", "zm_"]),
]

/// A single Kokoro voice entry.
struct KokoroVoice: Identifiable, Hashable {
    let id: String       // e.g. "kokoro_af_heart"
    let displayName: String  // e.g. "Heart (Female)"
}

/// Observable model that owns the TTS synthesizer and manages state.
@MainActor
class TTSModel: ObservableObject {
    @Published var selectedLanguage: KokoroLanguage = kokoroLanguages[0]
    @Published var availableVoices: [KokoroVoice] = []
    @Published var selectedVoice: KokoroVoice? = nil
    @Published var isSpeaking: Bool = false
    @Published var isReady: Bool = false
    @Published var errorMessage: String? = nil

    private var tts: MoonshineVoice.TextToSpeech? = nil
    private var g2pRoot: String = ""

    func initialize() {
        guard let bundle = Bundle.main.url(forResource: "tts-data", withExtension: nil) else {
            errorMessage = "TTS assets not found in app bundle"
            return
        }
        g2pRoot = bundle.path

        #if os(iOS)
        // Configure audio session for playback
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            print("Audio session setup warning: \(error)")
        }
        #endif

        do {
            tts = try MoonshineVoice.TextToSpeech(
                language: selectedLanguage.id,
                g2pRoot: g2pRoot,
                voice: "kokoro_af_heart"
            )
            refreshVoices()
            isReady = true
        } catch {
            errorMessage = "Failed to initialize TTS: \(error)"
        }
    }

    func changeLanguage(_ lang: KokoroLanguage) {
        guard lang.id != selectedLanguage.id else { return }
        selectedLanguage = lang
        recreateSynthesizer(voice: nil)
        refreshVoices()
    }

    func changeVoice(_ voice: KokoroVoice?) {
        selectedVoice = voice
        recreateSynthesizer(voice: voice?.id)
    }

    func speak(_ text: String) {
        guard let tts = tts, !text.isEmpty else { return }
        isSpeaking = true
        // Run synthesis + playback off the main thread
        let currentTts = tts
        Task.detached { [weak self] in
            defer {
                Task { @MainActor [weak self] in
                    self?.isSpeaking = false
                }
            }
            do {
                try currentTts.say(text)
            } catch {
                Task { @MainActor [weak self] in
                    self?.errorMessage = "Playback failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func recreateSynthesizer(voice: String?) {
        tts?.close()
        tts = nil
        isReady = false
        errorMessage = nil

        do {
            tts = try MoonshineVoice.TextToSpeech(
                language: selectedLanguage.id,
                g2pRoot: g2pRoot,
                voice: voice
            )
            isReady = true
        } catch {
            errorMessage = "Failed to create synthesizer: \(error)"
        }
    }

    private func refreshVoices() {
        do {
            let json = try MoonshineVoice.TextToSpeech.getVoices(
                languages: selectedLanguage.id,
                options: [TranscriberOption(name: "g2p_root", value: g2pRoot)]
            )
            availableVoices = parseVoices(json: json, language: selectedLanguage)
            // Select the first voice if none selected or current doesn't match
            if let first = availableVoices.first,
               selectedVoice == nil || !availableVoices.contains(where: { $0.id == selectedVoice?.id })
            {
                selectedVoice = first
            }
        } catch {
            availableVoices = []
            selectedVoice = nil
        }
    }

    private func parseVoices(json: String, language: KokoroLanguage) -> [KokoroVoice] {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let langVoices = dict[language.id] as? [[String: String]]
        else { return [] }

        return langVoices.compactMap { entry -> KokoroVoice? in
            guard let voiceId = entry["id"],
                  let state = entry["state"],
                  state == "found",
                  voiceId.hasPrefix("kokoro_")
            else { return nil }

            let shortId = String(voiceId.dropFirst("kokoro_".count))
            let displayName = formatVoiceName(shortId)
            return KokoroVoice(id: voiceId, displayName: displayName)
        }
    }

    private func formatVoiceName(_ shortId: String) -> String {
        // shortId is like "af_heart" -> "Heart (Female)"
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
}

@main
struct TextToSpeechApp: App {
    @StateObject private var model = TTSModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task {
                    model.initialize()
                    // Speak welcome message once ready
                    if model.isReady {
                        model.speak("Welcome to Moonshine Text to Speech")
                    }
                }
        }
    }
}
