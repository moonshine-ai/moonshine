import Foundation

/// Protocol for objects that can listen to transcript events.
/// 
/// Implement this protocol to receive notifications about transcription events.
/// All methods have default no-op implementations, so you only need to override
/// the ones you care about.
public protocol TranscriptEventListener: AnyObject {
    /// Called when a new transcription line starts.
    func onLineStarted(_ event: LineStarted)
    
    /// Called when an existing transcription line is updated.
    func onLineUpdated(_ event: LineUpdated)
    
    /// Called when the text of a transcription line changes.
    func onLineTextChanged(_ event: LineTextChanged)
    
    /// Called when a transcription line is completed.
    func onLineCompleted(_ event: LineCompleted)
    
    /// Called when an error occurs.
    func onError(_ event: TranscriptError)
}

/// Default implementations that do nothing.
public extension TranscriptEventListener {
    func onLineStarted(_ event: LineStarted) {}
    func onLineUpdated(_ event: LineUpdated) {}
    func onLineTextChanged(_ event: LineTextChanged) {}
    func onLineCompleted(_ event: LineCompleted) {}
    func onError(_ event: TranscriptError) {}
}

