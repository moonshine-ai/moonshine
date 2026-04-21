import MoonshineVoice
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: TTSModel
    @State private var inputText: String = ""
    @FocusState private var textFieldFocused: Bool

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                languagePicker
                voicePicker

                Divider()

                TextField("Enter text to speak...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)
                    .padding(.horizontal)
                    .focused($textFieldFocused)
                    .submitLabel(.done)
                    .onSubmit { speakCurrentText() }

                speakButton

                if let error = model.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }

                progressSection

                Spacer()
            }
            .padding(.top)
            .navigationTitle("Moonshine TTS")
            .navigationBarTitleDisplayMode(.inline)
            .disabled(model.isDownloading || model.isBootstrapping)
        }
    }

    private var languagePicker: some View {
        HStack {
            Text("Language").font(.headline)
            Spacer()
            Picker(
                "Language",
                selection: Binding(
                    get: { model.selectedLanguage },
                    set: { model.changeLanguage($0) })
            ) {
                ForEach(kokoroLanguages) { lang in
                    Text(lang.displayName).tag(lang)
                }
            }
            .pickerStyle(.menu)
        }
        .padding(.horizontal)
    }

    private var voicePicker: some View {
        HStack {
            Text("Voice").font(.headline)
            Spacer()
            Picker(
                "Voice",
                selection: Binding(
                    get: {
                        model.selectedVoice
                            ?? TtsVoice(id: "", displayName: "", needsDownload: false)
                    },
                    set: { model.changeVoice($0.id.isEmpty ? nil : $0) })
            ) {
                ForEach(model.availableVoices) { voice in
                    Text(voice.displayName).tag(voice)
                }
            }
            .pickerStyle(.menu)
            .disabled(model.availableVoices.isEmpty)
        }
        .padding(.horizontal)
    }

    private var speakButton: some View {
        Button(action: { speakCurrentText() }) {
            HStack {
                Image(systemName: model.isSpeaking ? "speaker.wave.3.fill" : "play.fill")
                Text(model.isSpeaking ? "Speaking..." : "Speak")
            }
            .font(.title2)
            .frame(maxWidth: .infinity)
            .padding()
            .background(speakButtonEnabled ? Color.blue : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!speakButtonEnabled)
        .padding(.horizontal)
    }

    private var speakButtonEnabled: Bool {
        model.isReady
            && !model.isSpeaking
            && !model.isDownloading
            && !model.isBootstrapping
            && (model.selectedVoice?.needsDownload == false)
    }

    @ViewBuilder
    private var progressSection: some View {
        if model.isBootstrapping && model.downloadStatus == nil {
            ProgressView("Initializing TTS...")
        } else if let status = model.downloadStatus {
            VStack(spacing: 6) {
                Text(
                    "Downloading \(status.fileName) (\(status.fileIndex)/\(status.totalFiles))"
                )
                .font(.caption)
                .foregroundColor(.secondary)
                if let fraction = status.fraction {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
            }
            .padding(.horizontal)
        } else if !model.isReady && model.errorMessage == nil {
            ProgressView("Loading voice...")
        }
    }

    private func speakCurrentText() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        textFieldFocused = false
        model.speak(text.isEmpty ? "Hello world" : text)
    }
}

#Preview {
    ContentView(model: TTSModel())
}
