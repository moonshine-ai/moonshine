import MoonshineVoice
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: TTSModel
    @State private var inputText: String = ""
    @FocusState private var textFieldFocused: Bool

    /// Stands in for "no catalogue voice selected", which is the state while a
    /// clone is in use or before the first voice list arrives.
    private static let placeholderVoice = TtsVoice(
        id: "", displayName: "", needsDownload: false)

    var body: some View {
        NavigationView {
            ScrollView {
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

                    Divider()

                    cloningSection
                }
                .padding(.vertical)
            }
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
                    get: { model.selectedVoice ?? Self.placeholderVoice },
                    set: { model.changeVoice($0.id.isEmpty ? nil : $0) })
            ) {
                // A cloned voice is not in the catalogue, so it needs a row of
                // its own for the picker to have something to show.
                if model.selectedVoice == nil {
                    Text(model.isCloned ? "Your voice" : "")
                        .tag(Self.placeholderVoice)
                }
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
            && !model.isCloning
            // A cloned voice came from the recording rather than the catalogue,
            // so there is no selected voice to be waiting on a download.
            && (model.isCloned || model.selectedVoice?.needsDownload == false)
    }

    /// Record a few seconds and speak in that voice from then on. The engine it
    /// needs is a separate download, so this is deliberately a second step
    /// rather than something every reader pays for on launch.
    @ViewBuilder
    private var cloningSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Clone your voice")
                .font(.headline)
            Text(
                "Record about four seconds of speech and the synthesizer will "
                    + "answer in your voice. The recording never leaves the device."
            )
            .font(.caption)
            .foregroundColor(.secondary)

            Button(action: { model.cloneFromMicrophone() }) {
                HStack {
                    Image(systemName: model.isCloning ? "waveform" : "mic.fill")
                    Text(recordButtonTitle)
                }
                .font(.body.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding()
                .background(model.isCloning ? Color.gray : Color.red)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(model.isCloning || model.isBootstrapping)

            if let status = model.cloneStatus {
                Text(status)
                    .font(.caption)
                    .foregroundColor(model.isCloned ? .green : .secondary)
            }
            if model.isCloned {
                Text("Pick a voice above to go back to a preset one.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal)
    }

    private var recordButtonTitle: String {
        if model.isCloning { return "Listening…" }
        return model.isCloned ? "Record again" : "Record four seconds"
    }

    @ViewBuilder
    private var progressSection: some View {
        if model.isBootstrapping && model.downloadStatus == nil {
            ProgressView("Initializing TTS...")
        } else if let status = model.downloadStatus {
            VStack(spacing: 6) {
                Text("Downloading \(status.fileName)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                ProgressView(value: status.fraction)
                    .progressViewStyle(.linear)
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
