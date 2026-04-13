import SwiftUI
import UIKit

struct ContentView: View {
    @ObservedObject var session: IntentSessionModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    thresholdSection

                    HStack {
                        Button(session.isListening ? "Stop" : "Listen") {
                            session.toggleListening()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!session.isEngineReady)

                        if !session.isEngineReady && session.statusMessage.isEmpty {
                            ProgressView("Loading…")
                        }
                    }

                    if !session.statusMessage.isEmpty {
                        Text(session.statusMessage)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Text("Intent phrases")
                        .font(.headline)

                    ForEach($session.phrases) { $row in
                        phraseEditorRow(row: $row)
                    }

                    Button {
                        session.addPhrase()
                    } label: {
                        Label("Add phrase", systemImage: "plus.circle.fill")
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
            }
            .navigationTitle("Intent Recognizer")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                session.bootstrapIfNeeded()
            }
            .onChange(of: scenePhase) { phase in
                if phase == .background {
                    session.pauseMicIfNeededForBackground()
                }
            }
        }
    }

    private var thresholdSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Match threshold")
                .font(.headline)
            HStack {
                Slider(value: $session.threshold, in: 0...1, step: 0.01)
                Text(String(format: "%.2f", session.threshold))
                    .monospacedDigit()
                    .frame(minWidth: 44, alignment: .trailing)
            }
            Text("Higher values require closer semantic matches (same scale as the Python example).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func phraseEditorRow(row: Binding<IntentPhraseRow>) -> some View {
        let isLit = session.highlightedPhraseIDs.contains(row.wrappedValue.id)
        return HStack(alignment: .center, spacing: 8) {
            TextField("Intent phrase", text: row.text)
                .textFieldStyle(.roundedBorder)
                .submitLabel(.done)
                .onSubmit { session.phraseTextCommitted() }

            Button(role: .destructive) {
                session.removePhrase(id: row.wrappedValue.id)
            } label: {
                Image(systemName: "trash")
            }
            .accessibilityLabel("Delete phrase")
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isLit ? Color.accentColor.opacity(0.22) : Color(uiColor: .secondarySystemGroupedBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isLit ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .animation(.easeInOut(duration: 0.2), value: isLit)
        .onChange(of: row.wrappedValue.text) { _ in
            session.scheduleDebouncedIntentSync()
        }
    }
}

#Preview {
    ContentView(session: IntentSessionModel())
}
