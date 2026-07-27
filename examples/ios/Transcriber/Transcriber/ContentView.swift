//
//  ContentView.swift
//  Transcriber
//
//  Created by Pete Warden on 1/1/26.
//

import SwiftUI

import MoonshineVoice

struct ContentView: View {
    @ObservedObject var session: TranscriptionSession

    var body: some View {
        VStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(session.lines.enumerated()), id: \.offset) { _, line in
                            transcriptText(line)
                        }
                        if !session.liveText.isEmpty {
                            transcriptText(session.liveText)
                                .foregroundColor(.secondary)
                        }
                        // Bottom anchor for scrolling
                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding(.vertical)
                }
                .onChange(of: session.lines.count) { _, _ in
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
                .onChange(of: session.liveText) { _, _ in
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
            }

            Spacer()

            if !session.status.isEmpty {
                Text(session.status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            HStack {
                Spacer()
                Button(action: { session.toggleRecording() }) {
                    Image(systemName: session.isRecording ? "mic.fill" : "mic")
                        .font(.system(size: 36))
                        .foregroundColor(session.isRecording ? .red : .blue)
                        .padding()
                        .background(
                            Circle()
                                .fill(
                                    session.isRecording
                                        ? Color.red.opacity(0.2) : Color.blue.opacity(0.2))
                        )
                }
                .disabled(!session.isReady)
                .opacity(session.isReady ? 1 : 0.4)
                Spacer()
            }
        }
        .padding()
    }

    private func transcriptText(_ text: String) -> some View {
        Text(text)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.vertical, 4)
    }
}

#Preview {
    ContentView(session: TranscriptionSession())
}
