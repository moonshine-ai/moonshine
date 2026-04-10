import SwiftUI
import MoonshineVoice

struct ContentView: View {
    @ObservedObject var model: TTSModel
    @State private var inputText: String = ""
    @FocusState private var textFieldFocused: Bool

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Language picker
                HStack {
                    Text("Language")
                        .font(.headline)
                    Spacer()
                    Picker("Language", selection: Binding(
                        get: { model.selectedLanguage },
                        set: { model.changeLanguage($0) }
                    )) {
                        ForEach(kokoroLanguages) { lang in
                            Text(lang.displayName).tag(lang)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .padding(.horizontal)

                // Voice picker
                HStack {
                    Text("Voice")
                        .font(.headline)
                    Spacer()
                    Picker("Voice", selection: Binding(
                        get: { model.selectedVoice ?? KokoroVoice(id: "", displayName: "") },
                        set: { model.changeVoice($0.id.isEmpty ? nil : $0) }
                    )) {
                        ForEach(model.availableVoices) { voice in
                            Text(voice.displayName).tag(voice)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .padding(.horizontal)

                Divider()

                // Text input
                TextField("Enter text to speak...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)
                    .padding(.horizontal)
                    .focused($textFieldFocused)
                    .submitLabel(.done)
                    .onSubmit {
                        speakCurrentText()
                    }

                // Speak button
                Button(action: {
                    speakCurrentText()
                }) {
                    HStack {
                        Image(systemName: model.isSpeaking ? "speaker.wave.3.fill" : "play.fill")
                        Text(model.isSpeaking ? "Speaking..." : "Speak")
                    }
                    .font(.title2)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(model.isSpeaking ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(model.isSpeaking || !model.isReady || inputText.trimmingCharacters(in: .whitespaces).isEmpty)
                .padding(.horizontal)

                // Status / error
                if let error = model.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }

                if !model.isReady && model.errorMessage == nil {
                    ProgressView("Initializing TTS...")
                }

                Spacer()
            }
            .padding(.top)
            .navigationTitle("Moonshine TTS")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func speakCurrentText() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        textFieldFocused = false
        model.speak(text)
    }
}

#Preview {
    ContentView(model: TTSModel())
}
