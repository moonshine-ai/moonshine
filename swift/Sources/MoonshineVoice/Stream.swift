import Foundation

/// A stream for real-time transcription with event-based updates.
public class Stream {
    private let transcriber: Transcriber
    private let api: MoonshineAPI
    private let handle: Int32
    private let updateInterval: TimeInterval
    
    private var listeners: [ListenerWrapper] = []
    private var streamTime: TimeInterval = 0.0
    private var lastUpdateTime: TimeInterval = 0.0
    
    /// Flags for transcription operations.
    public static let flagForceUpdate: UInt32 = 1 << 0
    
    internal init(
        transcriber: Transcriber,
        handle: Int32,
        updateInterval: TimeInterval = 0.5,
        flags: UInt32 = 0
    ) {
        self.transcriber = transcriber
        self.api = MoonshineAPI.shared
        self.handle = handle
        self.updateInterval = updateInterval
    }
    
    deinit {
        close()
    }
    
    /// Start the stream.
    public func start() throws {
        try api.startStream(transcriberHandle: transcriber.handle, streamHandle: handle)
    }
    
    /// Stop the stream.
    /// This will process any remaining audio and emit final events.
    public func stop() throws {
        try api.stopStream(transcriberHandle: transcriber.handle, streamHandle: handle)
        // There may be some audio left in the stream, so we need to transcribe it
        // to get the final transcript and emit events.
        do {
            try updateTranscription(flags: 0)
        } catch {
            emitError(error)
        }
    }
    
    /// Add audio data to the stream.
    /// - Parameters:
    ///   - audioData: Array of PCM audio samples (float, -1.0 to 1.0)
    ///   - sampleRate: Sample rate in Hz (default: 16000)
    public func addAudio(_ audioData: [Float], sampleRate: Int32 = 16000) throws {
        try api.addAudioToStream(
            transcriberHandle: transcriber.handle,
            streamHandle: handle,
            audioData: audioData,
            sampleRate: sampleRate,
            flags: 0
        )
        
        streamTime += Double(audioData.count) / Double(sampleRate)
        
        // Auto-update if enough time has passed
        if streamTime - lastUpdateTime >= updateInterval {
            try updateTranscription(flags: 0)
            lastUpdateTime = streamTime
        }
    }
    
    /// Manually update the transcription from the stream.
    /// - Parameter flags: Flags for transcription (e.g., `Stream.flagForceUpdate`)
    /// - Returns: The current transcript
    @discardableResult
    public func updateTranscription(flags: UInt32 = 0) throws -> Transcript {
        let transcript = try api.transcribeStream(
            transcriberHandle: transcriber.handle,
            streamHandle: handle,
            flags: flags
        )
        notifyFromTranscript(transcript)
        return transcript
    }
    
    /// Add an event listener to the stream.
    /// - Parameter listener: Either a `TranscriptEventListener` instance or a closure
    /// - Note: Closures can throw errors, which will be caught and emitted as error events
    public func addListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        listeners.append(.closure(listener))
    }
    
    /// Add a `TranscriptEventListener` instance to the stream.
    /// - Parameter listener: The listener object
    public func addListener(_ listener: TranscriptEventListener) {
        listeners.append(.object(listener))
    }
    
    /// Remove an event listener from the stream.
    /// - Parameter listener: The listener to remove
    /// - Note: Closure-based listeners cannot be reliably removed due to Swift's closure identity limitations.
    ///   Consider using `TranscriptEventListener` objects instead if you need to remove listeners.
    public func removeListener(_ listener: @escaping (TranscriptEvent) throws -> Void) {
        // Note: We can't reliably compare closures in Swift, so this is a best-effort removal
        // Users should prefer TranscriptEventListener objects if they need to remove listeners
        listeners.removeAll { wrapper in
            if case .closure = wrapper {
                // We can't compare closures, so we'll just remove the first closure we find
                // This is not ideal, but Swift doesn't provide closure identity
                return true
            }
            return false
        }
    }
    
    /// Remove a `TranscriptEventListener` instance from the stream.
    /// - Parameter listener: The listener object to remove
    public func removeListener(_ listener: TranscriptEventListener) {
        listeners.removeAll { wrapper in
            if case .object(let obj) = wrapper {
                return obj === listener
            }
            return false
        }
    }
    
    /// Remove all event listeners from the stream.
    public func removeAllListeners() {
        listeners.removeAll()
    }
    
    /// Close the stream and free its resources.
    public func close() {
        do {
            try api.freeStream(transcriberHandle: transcriber.handle, streamHandle: handle)
        } catch {
            // Ignore errors during cleanup
        }
        removeAllListeners()
    }

    public func getHandle() -> Int32 {
        return handle;
    }
    
    // MARK: - Private Methods
    
    private func notifyFromTranscript(_ transcript: Transcript) {
        for line in transcript.lines {
            if line.isNew {
                emit(LineStarted(line: line, streamHandle: handle))
            }
            if line.isUpdated && !line.isNew && !line.isComplete {
                emit(LineUpdated(line: line, streamHandle: handle))
            }
            if line.hasTextChanged {
                emit(LineTextChanged(line: line, streamHandle: handle))
            }
            if line.isComplete && line.isUpdated {
                emit(LineCompleted(line: line, streamHandle: handle))
            }
        }
    }
    
    private func emit(_ event: TranscriptEvent) {
        for wrapper in listeners {
            switch wrapper {
            case .closure(let closure):
                // Closures might throw, so we catch errors
                do {
                    try closure(event)
                } catch {
                    // Don't let listener errors break the stream
                    // Emit an error event if possible, but don't recurse
                    let errorEvent = TranscriptError(
                        line: nil,
                        streamHandle: handle,
                        error: error
                    )
                    // Only emit to other listeners to avoid recursion
                    for otherWrapper in listeners {
                        // Compare wrappers to avoid emitting to the same listener that errored
                        let isSameWrapper: Bool
                        switch (wrapper, otherWrapper) {
                        case (.object(let lhsObj), .object(let rhsObj)):
                            isSameWrapper = lhsObj === rhsObj
                        case (.closure, .closure):
                            // Can't compare closures reliably, so we'll emit to all closures
                            // This is not perfect but avoids missing error notifications
                            isSameWrapper = false
                        default:
                            isSameWrapper = false
                        }
                        
                        if !isSameWrapper {
                            switch otherWrapper {
                            case .closure(let otherClosure):
                                do {
                                    try otherClosure(errorEvent)
                                } catch {
                                    // Ignore errors in error handlers
                                }
                            case .object(let listener):
                                listener.onError(errorEvent)
                            }
                        }
                    }
                }
            case .object(let listener):
                // Protocol methods don't throw, so no need for do-catch
                if let lineStarted = event as? LineStarted {
                    listener.onLineStarted(lineStarted)
                } else if let lineUpdated = event as? LineUpdated {
                    listener.onLineUpdated(lineUpdated)
                } else if let lineTextChanged = event as? LineTextChanged {
                    listener.onLineTextChanged(lineTextChanged)
                } else if let lineCompleted = event as? LineCompleted {
                    listener.onLineCompleted(lineCompleted)
                } else if let error = event as? TranscriptError {
                    listener.onError(error)
                }
            }
        }
    }
    
    private func emitError(_ error: Error) {
        let errorEvent = TranscriptError(
            line: nil,
            streamHandle: handle,
            error: error
        )
        emit(errorEvent)
    }
    
    // MARK: - Listener Wrapper
    
    private enum ListenerWrapper {
        case closure((TranscriptEvent) throws -> Void)
        case object(TranscriptEventListener)
        
        static func == (lhs: ListenerWrapper, rhs: ListenerWrapper) -> Bool {
            switch (lhs, rhs) {
            case (.closure, .closure):
                // Closures can't be compared for equality in Swift
                return false
            case (.object(let lhsObj), .object(let rhsObj)):
                return lhsObj === rhsObj
            default:
                return false
            }
        }
    }
}

