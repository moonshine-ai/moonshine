import SwiftUI
import UIKit

struct ContentView: View {
    @ObservedObject var session: IntentSessionModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
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

                    transcriptOrStatusSection

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

                    thresholdSection
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

    private var transcriptOrStatusSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            if session.isListening {
                Text(session.liveTranscriptLine.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                     ? "Speak…"
                     : session.liveTranscriptLine)
                    .font(.body)
                    .foregroundStyle(
                        session.liveTranscriptLine.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? Color.secondary
                            : Color.primary
                    )
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if shouldShowSecondaryStatus {
                Text(session.statusMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var shouldShowSecondaryStatus: Bool {
        let s = session.statusMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return false }
        if session.isListening {
            return s.localizedCaseInsensitiveContains("error")
                || s.contains("Missing")
                || s.contains("Failed")
                || s.contains("Microphone")
                || s.contains("Could not")
        }
        return true
    }

    private var thresholdSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Minimum similarity")
                .font(.headline)
            HStack {
                Slider(
                    value: Binding(
                        get: { session.threshold },
                        set: { session.threshold = min(max($0, 0), 1) }
                    ),
                    in: 0...1,
                    step: 0.01
                )
                Text(String(format: "%.2f", session.threshold))
                    .monospacedDigit()
                    .frame(minWidth: 44, alignment: .trailing)
            }
            Text(
                "Same meaning as Python’s --threshold: an intent must score at least this value (cosine similarity, roughly 0–1). Lower accepts weaker matches; 0 still picks the closest phrase among your intents because almost all scores are ≥ 0. The default here is 0.8 (fairly strict); lower toward 0.5 for more permissive matching."
            )
            .font(.caption)
            .foregroundStyle(.secondary)

            if let minUsed = session.lastScoredMinimumSimilarity, let best = session.lastTopMatchSimilarity {
                Text(
                    String(
                        format: "Last completed line: best score %.2f (minimum used %.2f)",
                        Double(best),
                        minUsed
                    )
                )
                .font(.caption2)
                .foregroundStyle(.tertiary)
            }
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
