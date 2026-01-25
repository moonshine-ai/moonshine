import Foundation
import Moonshine

public struct TranscribeStreamFlags {
    public static let flagForceUpdate: UInt32 = 1 << 0
}

/// Internal wrapper for the Moonshine C API.
internal final class MoonshineAPI: @unchecked Sendable {
    static nonisolated let shared = MoonshineAPI()

    private init() {}

    /// Get the version of the loaded Moonshine library.
    func getVersion() -> Int32 {
        return moonshine_get_version()
    }

    /// Convert an error code to a human-readable string.
    func errorToString(_ errorCode: Int32) -> String {
        guard let errorString = moonshine_error_to_string(errorCode) else {
            return "Unknown error"
        }
        return String(cString: errorString)
    }

    /// Load a transcriber from files on disk.
    func loadTranscriberFromFiles(
        path: String,
        modelArch: ModelArch,
        options: [TranscriberOption]? = nil,
        moonshineVersion: Int32 = 20000
    ) throws -> Int32 {
        let pathCString = path.cString(using: .utf8)!

        var handle: Int32

        if let options = options, !options.isEmpty {
            // Store C string arrays to keep them alive
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }

            // Build option structs - the C API only reads pointers during the call
            // so we can safely use the array base addresses
            let optionStructs = (0..<options.count).map { i -> transcriber_option_t in
                // Get base address of the C string array
                // Note: These pointers are valid as long as the arrays exist
                return transcriber_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }

            // Keep string arrays alive and make the call
            // The pointers in optionStructs reference the arrays, so they remain valid
            handle = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_load_transcriber_from_files(
                        pathCString,
                        modelArch.rawValue,
                        buffer.baseAddress,
                        UInt64(options.count),
                        moonshineVersion
                    )
                }
            }
        } else {
            handle = moonshine_load_transcriber_from_files(
                pathCString,
                modelArch.rawValue,
                nil,
                0,
                moonshineVersion
            )
        }

        if handle < 0 {
            let errorString = errorToString(handle)
            throw MoonshineError.custom(
                message: "Failed to load transcriber: \(errorString)", code: handle)
        }

        return handle
    }

    /// Free a transcriber handle.
    func freeTranscriber(_ handle: Int32) {
        moonshine_free_transcriber(handle)
    }

    /// Transcribe audio without streaming.
    func transcribeWithoutStreaming(
        transcriberHandle: Int32,
        audioData: [Float],
        sampleRate: Int32,
        flags: UInt32
    ) throws -> Transcript {
        var outTranscriptPtr: UnsafeMutablePointer<transcript_t>? = nil

        let error = audioData.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                return Int32(-3)  // MOONSHINE_ERROR_INVALID_ARGUMENT
            }
            // C function takes float* but doesn't modify, so we can safely cast
            // Use withUnsafeMutablePointer to get the correct pointer type for **
            return withUnsafeMutablePointer(to: &outTranscriptPtr) { transcriptPtrPtr in
                moonshine_transcribe_without_streaming(
                    transcriberHandle,
                    UnsafeMutablePointer(mutating: baseAddress),
                    UInt64(audioData.count),
                    sampleRate,
                    flags,
                    transcriptPtrPtr
                )
            }
        }

        try checkError(error)

        guard let transcriptPtr = outTranscriptPtr else {
            return Transcript(lines: [])
        }

        return parseTranscript(transcriptPtr)
    }

    /// Create a stream for real-time transcription.
    func createStream(transcriberHandle: Int32, flags: UInt32) throws -> Int32 {
        let handle = moonshine_create_stream(transcriberHandle, flags)
        try checkError(handle)
        return handle
    }

    /// Free a stream handle.
    func freeStream(transcriberHandle: Int32, streamHandle: Int32) throws {
        let error = moonshine_free_stream(transcriberHandle, streamHandle)
        try checkError(error)
    }

    /// Start a stream.
    func startStream(transcriberHandle: Int32, streamHandle: Int32) throws {
        let error = moonshine_start_stream(transcriberHandle, streamHandle)
        try checkError(error)
    }

    /// Stop a stream.
    func stopStream(transcriberHandle: Int32, streamHandle: Int32) throws {
        let error = moonshine_stop_stream(transcriberHandle, streamHandle)
        try checkError(error)
    }

    /// Add audio data to a stream.
    func addAudioToStream(
        transcriberHandle: Int32,
        streamHandle: Int32,
        audioData: [Float],
        sampleRate: Int32,
        flags: UInt32
    ) throws {
        let error = audioData.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                return Int32(-3)  // MOONSHINE_ERROR_INVALID_ARGUMENT
            }
            // C function takes const float*, so we can pass UnsafePointer directly
            return moonshine_transcribe_add_audio_to_stream(
                transcriberHandle,
                streamHandle,
                baseAddress,
                UInt64(audioData.count),
                sampleRate,
                flags
            )
        }
        try checkError(error)
    }

    /// Transcribe a stream and get updated results.
    func transcribeStream(
        transcriberHandle: Int32,
        streamHandle: Int32,
        flags: UInt32
    ) throws -> Transcript {
        var outTranscriptPtr: UnsafeMutablePointer<transcript_t>? = nil

        // Use withUnsafeMutablePointer to get the correct pointer type for **
        let error = withUnsafeMutablePointer(to: &outTranscriptPtr) { transcriptPtrPtr in
            moonshine_transcribe_stream(
                transcriberHandle,
                streamHandle,
                flags,
                transcriptPtrPtr
            )
        }

        try checkError(error)

        guard let transcriptPtr = outTranscriptPtr else {
            return Transcript(lines: [])
        }

        return parseTranscript(transcriptPtr)
    }

    /// Parse a C transcript structure into a Swift Transcript.
    private func parseTranscript(_ transcriptPtr: UnsafeMutablePointer<transcript_t>) -> Transcript
    {
        let transcript = transcriptPtr.pointee
        var lines: [TranscriptLine] = []

        for i in 0..<transcript.line_count {
            let lineC = transcript.lines[Int(i)]

            // Extract text
            var text = ""
            if let textPtr = lineC.text {
                text = String(cString: textPtr)
            }

            // Extract audio data if available
            var audioData: [Float]? = nil
            if let audioPtr = lineC.audio_data, lineC.audio_data_count > 0 {
                // Validate audio_data_count is reasonable (max ~10 minutes at 16kHz = 9,600,000 samples)
                let maxReasonableCount: UInt64 = 10_000_000
                let audioCountUInt64 = UInt64(lineC.audio_data_count)
                let audioCount = min(audioCountUInt64, maxReasonableCount)
                
                // Check that the count can be safely converted to Int
                if audioCount <= UInt64(Int.max) {
                    let intCount = Int(audioCount)
                    if intCount > 0 {
                        // Safely create the buffer and array
                        audioData = Array(
                            UnsafeBufferPointer(
                                start: audioPtr,
                                count: intCount
                            ))
                    }
                }
                // If validation fails, audioData remains nil and we continue without audio data
            }

            let line = TranscriptLine(
                text: text,
                startTime: lineC.start_time,
                duration: lineC.duration,
                lineId: lineC.id,
                isComplete: lineC.is_complete != 0,
                isUpdated: lineC.is_updated != 0,
                isNew: lineC.is_new != 0,
                hasTextChanged: lineC.has_text_changed != 0,
                audioData: audioData
            )
            lines.append(line)
        }

        return Transcript(lines: lines)
    }
}

/// Transcriber option for advanced configuration.
public struct TranscriberOption {
    public let name: String
    public let value: String

    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }
}
