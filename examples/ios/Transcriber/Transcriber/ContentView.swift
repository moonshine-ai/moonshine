//
//  ContentView.swift
//  Transcriber
//
//  Created by Pete Warden on 1/1/26.
//

import SwiftUI

import MoonshineVoice

struct ContentView: View {
    @Binding var isRecording: Bool
    @Binding var messages: [TranscriptLine]
    @Binding var status: String
    /// Disables the mic button until the model has finished downloading/loading.
    var isReady: Bool = false
    
    var body: some View {
        VStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(messages, id: \.lineId) { message in
                            Text(message.text)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal)
                                .padding(.vertical, 4)
                        }
                        // Bottom anchor for scrolling
                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding(.vertical)
                }
                .onChange(of: messages.count) { _, _ in
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
                .onChange(of: messages.last?.text) { _, _ in
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            
            Spacer()

            if !status.isEmpty {
                Text(status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            HStack {
                Spacer()
                Button(action: {
                    isRecording.toggle()
                }) {
                    Image(systemName: isRecording ? "mic.fill" : "mic")
                        .font(.system(size: 36))
                        .foregroundColor(isRecording ? .red : .blue)
                        .padding()
                        .background(
                            Circle()
                                .fill(isRecording ? Color.red.opacity(0.2) : Color.blue.opacity(0.2))
                        )
                }
                .disabled(!isReady)
                .opacity(isReady ? 1 : 0.4)
                Spacer()
            }
        }
        .padding()
    }
}

#Preview {
    ContentView(
        isRecording: .constant(false), messages: .constant([]),
        status: .constant("Ready"), isReady: true)
}
