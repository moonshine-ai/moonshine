import Foundation
import Moonshine

public struct TranscribeStreamFlags {
    public static let flagForceUpdate: UInt32 = 1 << 0
    /// Run the alphanumeric spelling-fusion path on completed lines.
    /// Requires the transcriber to have been built with a
    /// ``spelling_model_path`` option (or ``Transcriber.spellingModelPath``);
    /// without one this flag is a no-op. See
    /// ``MOONSHINE_FLAG_SPELLING_MODE`` in the C header for details.
    public static let flagSpellingMode: UInt32 = 1 << 1
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
            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                // Get base address of the C string array
                // Note: These pointers are valid as long as the arrays exist
                return moonshine_option_t(
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

            // Extract word timestamps if available
            var words: [WordTiming] = []
            if let wordsPtr = lineC.words, lineC.word_count > 0 {
                for j in 0..<Int(lineC.word_count) {
                    let wordC = wordsPtr[j]
                    var wordText = ""
                    if let wordTextPtr = wordC.text {
                        wordText = String(cString: wordTextPtr)
                    }
                    words.append(WordTiming(
                        word: wordText,
                        start: wordC.start,
                        end: wordC.end,
                        confidence: wordC.confidence
                    ))
                }
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
                hasSpeakerId: lineC.has_speaker_id != 0,
                speakerId: lineC.speaker_id,
                speakerIndex: lineC.speaker_index,
                audioData: audioData,
                words: words
            )
            lines.append(line)
        }

        return Transcript(lines: lines)
    }

    // MARK: - Text to Speech

    /// Create a TTS synthesizer from files on disk.
    func createTtsSynthesizerFromFiles(
        language: String,
        options: [TranscriberOption]? = nil,
        moonshineVersion: Int32 = 20000
    ) throws -> Int32 {
        let langCString = language.cString(using: .utf8)!

        var handle: Int32

        if let options = options, !options.isEmpty {
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }

            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                return moonshine_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }

            handle = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_create_tts_synthesizer_from_files(
                        langCString,
                        nil,
                        0,
                        buffer.baseAddress,
                        UInt64(options.count),
                        moonshineVersion
                    )
                }
            }
        } else {
            handle = moonshine_create_tts_synthesizer_from_files(
                langCString,
                nil,
                0,
                nil,
                0,
                moonshineVersion
            )
        }

        if handle < 0 {
            let errorString = errorToString(handle)
            throw MoonshineError.custom(
                message: "Failed to create TTS synthesizer: \(errorString)", code: handle)
        }

        return handle
    }

    /// Synthesize text to speech, returning PCM float samples and sample rate.
    func textToSpeech(
        ttsHandle: Int32,
        text: String,
        options: [TranscriberOption]? = nil
    ) throws -> TtsSynthesisResult {
        let textCString = text.cString(using: .utf8)!
        var outAudioData: UnsafeMutablePointer<Float>? = nil
        var outAudioDataSize: UInt64 = 0
        var outSampleRate: Int32 = 0

        let error: Int32

        if let options = options, !options.isEmpty {
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }

            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                return moonshine_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }

            error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_text_to_speech(
                        ttsHandle,
                        textCString,
                        buffer.baseAddress,
                        UInt64(options.count),
                        &outAudioData,
                        &outAudioDataSize,
                        &outSampleRate
                    )
                }
            }
        } else {
            error = moonshine_text_to_speech(
                ttsHandle,
                textCString,
                nil,
                0,
                &outAudioData,
                &outAudioDataSize,
                &outSampleRate
            )
        }

        try checkError(error)

        var samples: [Float] = []
        if let audioPtr = outAudioData, outAudioDataSize > 0 {
            samples = Array(UnsafeBufferPointer(start: audioPtr, count: Int(outAudioDataSize)))
            free(outAudioData)
        }

        return TtsSynthesisResult(samples: samples, sampleRateHz: outSampleRate)
    }

    /// Free a TTS synthesizer handle.
    func freeTtsSynthesizer(_ handle: Int32) {
        moonshine_free_tts_synthesizer(handle)
    }

    /// Get TTS voices JSON for the given languages.
    func getTtsVoices(
        languages: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        let langCString = languages.cString(using: .utf8)!
        var outJson: UnsafeMutablePointer<CChar>? = nil

        let error: Int32

        if let options = options, !options.isEmpty {
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }

            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                return moonshine_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }

            error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_get_tts_voices(
                        langCString,
                        buffer.baseAddress,
                        UInt64(options.count),
                        &outJson
                    )
                }
            }
        } else {
            error = moonshine_get_tts_voices(
                langCString,
                nil,
                0,
                &outJson
            )
        }

        try checkError(error)

        guard let jsonPtr = outJson else {
            return "{}"
        }
        let result = String(cString: jsonPtr)
        free(outJson)
        return result
    }

    /// Get TTS dependencies JSON for the given languages.
    func getTtsDependencies(
        languages: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        let langCString = languages.cString(using: .utf8)!
        var outJson: UnsafeMutablePointer<CChar>? = nil

        let error: Int32

        if let options = options, !options.isEmpty {
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }

            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                return moonshine_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }

            error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_get_tts_dependencies(
                        langCString,
                        buffer.baseAddress,
                        UInt64(options.count),
                        &outJson
                    )
                }
            }
        } else {
            error = moonshine_get_tts_dependencies(
                langCString,
                nil,
                0,
                &outJson
            )
        }

        try checkError(error)

        guard let jsonPtr = outJson else {
            return "[]"
        }
        let result = String(cString: jsonPtr)
        free(outJson)
        return result
    }

    // MARK: - Intent recognition

    func createIntentRecognizer(
        modelPath: String,
        embeddingModelArch: UInt32,
        modelVariant: String = "q4"
    ) throws -> Int32 {
        let pathC = modelPath.cString(using: .utf8)!
        let variantC = modelVariant.cString(using: .utf8)!
        let handle = moonshine_create_intent_recognizer(
            pathC, embeddingModelArch, variantC)
        if handle < 0 {
            throw MoonshineError.custom(
                message: "Failed to create intent recognizer: \(errorToString(handle))",
                code: handle)
        }
        return handle
    }

    func freeIntentRecognizer(_ handle: Int32) {
        moonshine_free_intent_recognizer(handle)
    }

    func registerIntentRecognizerIntent(handle: Int32, canonicalPhrase: String,
                                        embedding: UnsafeMutablePointer<Float>? = nil,
                                        embeddingSize: UInt64 = 0,
                                        priority: Int32 = 0) throws {
        let phraseC = canonicalPhrase.cString(using: .utf8)!
        try checkError(moonshine_register_intent(handle, phraseC, embedding, embeddingSize, priority))
    }

    /// - Returns: `true` if an intent was removed, `false` if the phrase was not registered.
    func unregisterIntentRecognizerIntent(handle: Int32, canonicalPhrase: String) throws -> Bool {
        let phraseC = canonicalPhrase.cString(using: .utf8)!
        let err = moonshine_unregister_intent(handle, phraseC)
        if err == 0 {
            return true
        }
        if err == -3 {
            return false
        }
        try checkError(err)
        return false
    }

    func getClosestIntents(
        intentRecognizerHandle: Int32,
        utterance: String,
        toleranceThreshold: Float
    ) throws -> [(canonicalPhrase: String, similarity: Float)] {
        var matchesPtr: UnsafeMutablePointer<moonshine_intent_match_t>? = nil
        var count: UInt64 = 0
        let err: Int32 = utterance.withCString { utterC in
            withUnsafeMutablePointer(to: &matchesPtr) { matchesPP in
                withUnsafeMutablePointer(to: &count) { countP in
                    moonshine_get_closest_intents(
                        intentRecognizerHandle,
                        utterC,
                        toleranceThreshold,
                        matchesPP,
                        countP
                    )
                }
            }
        }
        try checkError(err)
        let n = Int(count)
        var out: [(canonicalPhrase: String, similarity: Float)] = []
        if let base = matchesPtr, n > 0 {
            for i in 0..<n {
                let row = base[i]
                let phrase: String
                if let p = row.canonical_phrase {
                    phrase = String(cString: p)
                } else {
                    phrase = ""
                }
                out.append((phrase, row.similarity))
            }
        }
        moonshine_free_intent_matches(matchesPtr, count)
        return out
    }

    func getIntentRecognizerIntentCount(handle: Int32) throws -> Int32 {
        let n = moonshine_get_intent_count(handle)
        if n < 0 {
            try checkError(n)
        }
        return n
    }

    func clearIntentRecognizerIntents(handle: Int32) throws {
        try checkError(moonshine_clear_intents(handle))
    }

    func calculateIntentEmbedding(handle: Int32, sentence: String) throws -> [Float] {
        var outPtr: UnsafeMutablePointer<Float>? = nil
        var outSize: UInt64 = 0
        let err: Int32 = sentence.withCString { sentenceC in
            withUnsafeMutablePointer(to: &outPtr) { embPP in
                withUnsafeMutablePointer(to: &outSize) { sizeP in
                    moonshine_calculate_intent_embedding(
                        handle, sentenceC, embPP, sizeP, nil)
                }
            }
        }
        try checkError(err)
        let n = Int(outSize)
        var result = [Float]()
        if let base = outPtr, n > 0 {
            result.reserveCapacity(n)
            for i in 0..<n {
                result.append(base[i])
            }
            moonshine_free_intent_embedding(base)
        }
        return result
    }
}

/// Transcriber option for advanced configuration.
public struct TranscriberOption: Sendable {
    public let name: String
    public let value: String

    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }
}
