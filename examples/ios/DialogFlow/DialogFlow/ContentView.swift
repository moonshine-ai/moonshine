import MoonshineVoice
import SwiftUI

struct ContentView: View {
    @ObservedObject var session: DialogSession
    @State private var typed: String = ""
    @FocusState private var textFieldFocused: Bool

    var body: some View {
        VStack(spacing: 12) {
            Text(session.status)
                .font(.footnote)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let fraction = session.downloadFraction {
                ProgressView(value: fraction)
                    .progressViewStyle(.linear)
            }

            transcript

            typeBar

            micButton
        }
        .padding()
        .onDisappear { session.close() }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(session.lines) { line in
                        HStack(alignment: .top, spacing: 8) {
                            Text(line.speaker == .you ? "you" : "assistant")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(line.speaker == .you ? .blue : .green)
                                .frame(width: 72, alignment: .leading)
                            Text(line.text)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .id(line.id)
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: session.lines.count) { _ in
                if let last = session.lines.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// Typing goes through the same path as speech, which is what makes the
    /// agent testable on a simulator with no microphone.
    private var typeBar: some View {
        HStack {
            TextField("Type an utterance instead…", text: $typed)
                .textFieldStyle(.roundedBorder)
                .focused($textFieldFocused)
                .submitLabel(.send)
                .onSubmit(sendTyped)
            Button("Send", action: sendTyped)
                .buttonStyle(.bordered)
                .disabled(!session.isReady || typed.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    private var micButton: some View {
        Button(action: {
            textFieldFocused = false
            if session.isListening {
                session.stopListening()
            } else {
                session.startListening()
            }
        }) {
            HStack {
                Image(systemName: session.isListening ? "stop.fill" : "mic.fill")
                Text(session.isListening ? "Stop" : "Start listening")
            }
            .font(.title3)
            .frame(maxWidth: .infinity)
            .padding()
            .background(session.isReady ? (session.isListening ? Color.red : Color.blue) : .gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!session.isReady)
    }

    private func sendTyped() {
        session.send(typed)
        typed = ""
    }
}

#Preview {
    ContentView(session: DialogSession())
}
