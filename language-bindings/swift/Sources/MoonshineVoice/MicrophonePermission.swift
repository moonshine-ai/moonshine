@preconcurrency import AVFoundation
import Foundation

/// Makes sure the app is allowed to record before anything opens the microphone,
/// prompting the user if the system has not asked yet.
///
/// Blocking here is deliberate: both callers are already off the main thread and
/// starting capture is one step of a longer setup, so an app that has to
/// interleave a permission callback into that sequence ends up with more state
/// machine than the feature is worth.
func ensureMicrophonePermission() throws {
    #if os(iOS) || os(tvOS) || os(watchOS)
    let audioSession = AVAudioSession.sharedInstance()
    try audioSession.setCategory(.record, mode: .default)
    try audioSession.setActive(true)

    switch audioSession.recordPermission {
    case .denied:
        throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
    case .undetermined:
        var granted = false
        let semaphore = DispatchSemaphore(value: 0)
        audioSession.requestRecordPermission { allowed in
            granted = allowed
            semaphore.signal()
        }
        semaphore.wait()
        if !granted {
            throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
        }
    default:
        break
    }
    #elseif os(macOS)
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .denied, .restricted:
        throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
    case .notDetermined:
        var granted = false
        let semaphore = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .audio) { allowed in
            granted = allowed
            semaphore.signal()
        }
        semaphore.wait()
        if !granted {
            throw MoonshineError.custom(message: "Microphone permission denied", code: -1)
        }
    default:
        break
    }
    #endif
}

/// Releases the recording audio session once capture has finished.
func releaseMicrophoneSession() {
    #if os(iOS) || os(tvOS) || os(watchOS)
    try? AVAudioSession.sharedInstance().setActive(false)
    #endif
}
