import SwiftUI

@main
struct IntentRecognizerApp: App {
    @StateObject private var session = IntentSessionModel()

    var body: some Scene {
        WindowGroup {
            ContentView(session: session)
        }
    }
}
