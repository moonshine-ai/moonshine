import SwiftUI

struct SettingsView: View {
    @AppStorage("showTimestamps") private var showTimestamps = true
    
    var body: some View {
        Form {
            Section("Display") {
                Toggle("Show Timestamps", isOn: $showTimestamps)
            }
        }
        .padding()
        .frame(width: 400, height: 200)
    }
}

