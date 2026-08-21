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

/// What a pull from a streaming generation found, mirroring the positive
/// statuses of ``moonshine_tts_next_chunk``.
internal enum TtsChunkResult {
    case chunk(TtsChunk)
    /// No complete sentence is buffered yet. Push more text, or flush.
    case needText
    /// Input ended and everything queued has been synthesized.
    case endOfStream
    /// A cancel discarded the reply being generated. Reported once, and only
    /// when there was something to discard, so a reader can tell an
    /// interruption from having run out of text.
    case cancelled
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
        moonshineVersion: Int32 = 30000
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

    /// Find the best short window of speech in a recording, for voice cloning.
    ///
    /// Wraps ``moonshine_extract_speech_clip``. Returns `nil` when the recording
    /// does not yet contain enough speech, which is how the incremental capture
    /// path knows to keep listening.
    func extractSpeechClip(
        audioData: [Float],
        sampleRate: Int32,
        ttsSynthesizerHandle: Int32,
        clipDurationSeconds: Float,
        minimumSpeechSeconds: Float
    ) throws -> (
        audio: [Float]?, startTime: Float, speechDuration: Float, transcript: String?
    ) {
        let options = [
            TranscriberOption(
                name: "clip_duration_seconds", value: String(clipDurationSeconds)),
            TranscriberOption(
                name: "minimum_speech_seconds", value: String(minimumSpeechSeconds)),
        ]
        let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
        let valueCStrings = options.map { $0.value.cString(using: .utf8)! }
        let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
            moonshine_option_t(
                name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
            )
        }

        var clip = moonshine_speech_clip_t()
        let error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
            audioData.withUnsafeBufferPointer { audio in
                optionStructs.withUnsafeBufferPointer { opts in
                    moonshine_extract_speech_clip(
                        audio.baseAddress,
                        UInt64(audioData.count),
                        sampleRate,
                        ttsSynthesizerHandle,
                        opts.baseAddress,
                        UInt64(options.count),
                        &clip
                    )
                }
            }
        }
        try checkError(error)

        var transcript: String?
        if let transcriptPtr = clip.transcript {
            transcript = String(cString: transcriptPtr)
            moonshine_free_buffer(transcriptPtr)
        }

        guard clip.is_complete != 0, let samples = clip.audio_data, clip.audio_length > 0 else {
            return (nil, clip.start_time, clip.speech_duration, transcript)
        }
        let audio = Array(UnsafeBufferPointer(start: samples, count: Int(clip.audio_length)))
        moonshine_free_buffer(samples)
        return (audio, clip.start_time, clip.speech_duration, transcript)
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

    /// Replace the contextual-biasing key terms on a live transcriber.
    func setKeyterms(transcriberHandle: Int32, keyterms: String) throws {
        let error = keyterms.withCString { keytermsC in
            moonshine_transcriber_set_keyterms(transcriberHandle, keytermsC)
        }
        if error < 0 {
            let errorString = errorToString(error)
            throw MoonshineError.custom(
                message: "Failed to set key terms: \(errorString)", code: error)
        }
    }

    /// Pick the key terms out of a passage of text on a live transcriber.
    func setContext(transcriberHandle: Int32, context: String, maxTerms: Int32) throws {
        let error = context.withCString { contextC in
            moonshine_transcriber_set_context(transcriberHandle, contextC, maxTerms)
        }
        if error < 0 {
            let errorString = errorToString(error)
            throw MoonshineError.custom(
                message: "Failed to set context: \(errorString)", code: error)
        }
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

            // Extract speaker spans if available
            var speakerSpans: [SpeakerSpan] = []
            if let spansPtr = lineC.speaker_spans, lineC.speaker_span_count > 0 {
                for j in 0..<Int(lineC.speaker_span_count) {
                    let spanC = spansPtr[j]
                    speakerSpans.append(SpeakerSpan(
                        startTime: spanC.start_time,
                        duration: spanC.duration,
                        speakerId: spanC.speaker_id,
                        speakerIndex: spanC.speaker_index,
                        startChar: spanC.start_char,
                        endChar: spanC.end_char
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
                haveSpeakersChanged: lineC.have_speakers_changed != 0,
                speakerSpans: speakerSpans,
                audioData: audioData,
                words: words,
                lastTranscriptionLatencyMs: lineC.last_transcription_latency_ms
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
        moonshineVersion: Int32 = 30000
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

    /// Create a TTS synthesizer from in-memory asset buffers keyed by canonical filename.
    ///
    /// The buffers pointed to by ``memoryPtrs`` are **not** copied by the native layer; the caller
    /// must keep them valid until the synthesizer is freed.
    func createTtsSynthesizerFromMemory(
        language: String,
        filenames: [String],
        memoryPtrs: [UnsafePointer<UInt8>?],
        memorySizes: [UInt64],
        options: [TranscriberOption]? = nil,
        moonshineVersion: Int32 = 30000
    ) throws -> Int32 {
        precondition(filenames.count == memoryPtrs.count && filenames.count == memorySizes.count)
        let langCString = language.cString(using: .utf8)!
        let fileCStrings = filenames.map { $0.cString(using: .utf8)! }
        let opts = options ?? []
        let nameCStrings = opts.map { $0.name.cString(using: .utf8)! }
        let valueCStrings = opts.map { $0.value.cString(using: .utf8)! }
        let optionStructs = (0..<opts.count).map { i -> moonshine_option_t in
            moonshine_option_t(
                name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
            )
        }

        // The C signature is ``const char **`` / ``const uint8_t **`` / ``const uint64_t *`` — the
        // outer pointers are mutable, so use mutable buffers (contents are still treated as const).
        var filePtrs: [UnsafePointer<CChar>?] = fileCStrings.map {
            $0.withUnsafeBufferPointer { $0.baseAddress }
        }
        var mem = memoryPtrs
        var sizes = memorySizes
        let handle: Int32 = withExtendedLifetime(
            (fileCStrings, nameCStrings, valueCStrings, optionStructs)
        ) {
            filePtrs.withUnsafeMutableBufferPointer { fileBuf in
                mem.withUnsafeMutableBufferPointer { memBuf in
                    sizes.withUnsafeMutableBufferPointer { sizeBuf in
                        optionStructs.withUnsafeBufferPointer { optBuf in
                            moonshine_create_tts_synthesizer_from_memory(
                                langCString,
                                fileBuf.baseAddress,
                                UInt64(filenames.count),
                                memBuf.baseAddress,
                                sizeBuf.baseAddress,
                                opts.isEmpty ? nil : optBuf.baseAddress,
                                UInt64(opts.count),
                                moonshineVersion
                            )
                        }
                    }
                }
            }
        }

        if handle < 0 {
            let errorString = errorToString(handle)
            throw MoonshineError.custom(
                message: "Failed to create TTS synthesizer from memory: \(errorString)", code: handle)
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

    /// Synthesize speech from IPA phonemes, returning PCM float samples and sample rate.
    ///
    /// `phonemes` is an IPA string in the same format produced by the
    /// `moonshine_text_to_phonemes` C API, letting callers inspect or edit phonemes between G2P
    /// and vocoding. Skips grapheme-to-phoneme conversion.
    func phonemesToSpeech(
        ttsHandle: Int32,
        phonemes: String,
        options: [TranscriberOption]? = nil
    ) throws -> TtsSynthesisResult {
        let phonemesCString = phonemes.cString(using: .utf8)!
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
                    moonshine_phonemes_to_speech(
                        ttsHandle,
                        phonemesCString,
                        buffer.baseAddress,
                        UInt64(options.count),
                        &outAudioData,
                        &outAudioDataSize,
                        &outSampleRate
                    )
                }
            }
        } else {
            error = moonshine_phonemes_to_speech(
                ttsHandle,
                phonemesCString,
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

    // MARK: - Streaming TTS

    /// Split `text` into the utterances a streaming synthesizer speaks one at a time.
    func ttsSplitUtterances(
        language: String,
        text: String,
        options: [TranscriberOption]? = nil
    ) throws -> [String] {
        let langCString = language.isEmpty ? nil : language.cString(using: .utf8)
        let textCString = text.cString(using: .utf8)!
        var outJson: UnsafeMutablePointer<CChar>? = nil

        let error: Int32
        if let options, !options.isEmpty {
            let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
            let valueCStrings = options.map { $0.value.cString(using: .utf8)! }
            let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                moonshine_option_t(
                    name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                    value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                )
            }
            error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                optionStructs.withUnsafeBufferPointer { buffer in
                    moonshine_tts_split_utterances(
                        langCString, textCString, buffer.baseAddress,
                        UInt64(options.count), &outJson)
                }
            }
        } else {
            error = moonshine_tts_split_utterances(langCString, textCString, nil, 0, &outJson)
        }

        try checkError(error)

        guard let jsonPtr = outJson else { return [] }
        let json = String(cString: jsonPtr)
        free(outJson)
        guard let data = json.data(using: .utf8),
            let units = try? JSONSerialization.jsonObject(with: data) as? [String]
        else {
            return []
        }
        return units
    }

    func ttsPushText(ttsHandle: Int32, text: String) throws {
        try checkError(moonshine_tts_push_text(ttsHandle, text.cString(using: .utf8)!))
    }

    func ttsFlush(ttsHandle: Int32) throws {
        try checkError(moonshine_tts_flush(ttsHandle))
    }

    func ttsEndInput(ttsHandle: Int32) throws {
        try checkError(moonshine_tts_end_input(ttsHandle))
    }

    func ttsCancel(ttsHandle: Int32) throws {
        try checkError(moonshine_tts_cancel(ttsHandle))
    }

    /// Whether a streaming generation is in flight on this synthesizer.
    func ttsIsStreaming(ttsHandle: Int32) -> Bool {
        return moonshine_tts_is_streaming(ttsHandle) > 0
    }

    /// Synthesize the next chunk.
    func ttsNextChunk(ttsHandle: Int32) throws -> TtsChunkResult {
        var out: UnsafePointer<tts_chunk_t>? = nil
        let status = moonshine_tts_next_chunk(ttsHandle, 0, &out)
        if status == MOONSHINE_TTS_END_OF_STREAM {
            return .endOfStream
        }
        if status == MOONSHINE_TTS_CANCELLED {
            return .cancelled
        }
        if status == MOONSHINE_TTS_NEED_TEXT {
            return .needText
        }
        try checkError(status)
        guard let raw = out?.pointee else { return .needText }
        var samples: [Float] = []
        if let audio = raw.audio_data, raw.audio_data_count > 0 {
            samples = Array(UnsafeBufferPointer(start: audio, count: Int(raw.audio_data_count)))
        }
        let chunk = TtsChunk(
            samples: samples,
            sampleRateHz: raw.sample_rate,
            text: raw.text.map { String(cString: $0) } ?? "",
            utteranceId: raw.utterance_id,
            isFinal: raw.is_final != 0)
        return .chunk(chunk)
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

    /// Get G2P-only asset dependency keys as a comma-separated string.
    func getG2pDependencies(
        languages: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        return try stringOutDependencyCall(
            primaryArg: languages,
            emptyResult: ""
        ) { primaryC, optsPtr, optsCount, outPtr in
            moonshine_get_g2p_dependencies(primaryC, optsPtr, optsCount, outPtr)
        }(options)
    }

    /// Get the speech-to-text model download manifest as a JSON object string.
    ///
    /// - Parameters:
    ///   - language: Language code (e.g. `"en"`) or English name; must not be empty.
    ///   - options: Optional options; recognizes `model_arch` (decimal string) and
    ///     `include_spelling` (bool).
    func getSttDependencies(
        language: String,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        return try stringOutDependencyCall(
            primaryArg: language,
            emptyResult: "{}"
        ) { primaryC, optsPtr, optsCount, outPtr in
            moonshine_get_stt_dependencies(primaryC, optsPtr, optsCount, outPtr)
        }(options)
    }

    /// Get the embedding model download manifest as a JSON object string.
    ///
    /// - Parameters:
    ///   - modelName: Embedding model id, or `nil` to use the default model.
    ///   - options: Optional options; recognizes `variant`.
    func getEmbeddingDependencies(
        modelName: String?,
        options: [TranscriberOption]? = nil
    ) throws -> String {
        return try stringOutDependencyCall(
            primaryArg: modelName,
            emptyResult: "{}"
        ) { primaryC, optsPtr, optsCount, outPtr in
            moonshine_get_embedding_dependencies(primaryC, optsPtr, optsCount, outPtr)
        }(options)
    }

    /// Get the speaker diarization download manifest as a JSON object string. There is one set of
    /// models and it takes no options, so this does not go through `stringOutDependencyCall`.
    func getDiarizationDependencies() throws -> String {
        var outJson: UnsafeMutablePointer<CChar>? = nil
        try checkError(moonshine_get_diarization_dependencies(&outJson))
        guard let jsonPtr = outJson else {
            return "{}"
        }
        let result = String(cString: jsonPtr)
        free(outJson)
        return result
    }

    /// Shared plumbing for the `moonshine_get_*_dependencies` family: marshals the primary string
    /// argument and options, invokes `call`, converts errors, and returns the malloc'd string.
    private func stringOutDependencyCall(
        primaryArg: String?,
        emptyResult: String,
        _ call: @escaping (
            UnsafePointer<CChar>?,
            UnsafePointer<moonshine_option_t>?,
            UInt64,
            UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
        ) -> Int32
    ) -> ([TranscriberOption]?) throws -> String {
        return { options in
            let primaryC = primaryArg?.cString(using: .utf8)
            var outJson: UnsafeMutablePointer<CChar>? = nil

            let error: Int32
            if let options = options, !options.isEmpty {
                let nameCStrings = options.map { $0.name.cString(using: .utf8)! }
                let valueCStrings = options.map { $0.value.cString(using: .utf8)! }
                let optionStructs = (0..<options.count).map { i -> moonshine_option_t in
                    moonshine_option_t(
                        name: nameCStrings[i].withUnsafeBufferPointer { $0.baseAddress },
                        value: valueCStrings[i].withUnsafeBufferPointer { $0.baseAddress }
                    )
                }
                error = withExtendedLifetime((nameCStrings, valueCStrings, optionStructs)) {
                    optionStructs.withUnsafeBufferPointer { buffer in
                        call(primaryC, buffer.baseAddress, UInt64(options.count), &outJson)
                    }
                }
            } else {
                error = call(primaryC, nil, 0, &outJson)
            }

            try checkError(error)

            guard let jsonPtr = outJson else {
                return emptyResult
            }
            let result = String(cString: jsonPtr)
            free(outJson)
            return result
        }
    }

    // MARK: - Text embeddings

    func createEmbeddingModel(
        modelPath: String,
        embeddingModelArch: UInt32,
        modelVariant: String = "q4"
    ) throws -> Int32 {
        let pathC = modelPath.cString(using: .utf8)!
        let variantC = modelVariant.cString(using: .utf8)!
        let handle = moonshine_create_embedding_model(
            pathC, embeddingModelArch, variantC)
        if handle < 0 {
            throw MoonshineError.custom(
                message: "Failed to create embedding model: \(errorToString(handle))",
                code: handle)
        }
        return handle
    }

    func freeEmbeddingModel(_ handle: Int32) {
        moonshine_free_embedding_model(handle)
    }

    /// Embed `sentence` with the model behind `handle`.
    func calculateEmbedding(handle: Int32, sentence: String) throws -> [Float] {
        var embeddingPtr: UnsafeMutablePointer<Float>? = nil
        var count: UInt64 = 0
        let err: Int32 = sentence.withCString { sentenceC in
            withUnsafeMutablePointer(to: &embeddingPtr) { embeddingPP in
                withUnsafeMutablePointer(to: &count) { countP in
                    moonshine_calculate_embedding(
                        handle, sentenceC, embeddingPP, countP, nil)
                }
            }
        }
        try checkError(err)
        var out: [Float] = []
        if let base = embeddingPtr, count > 0 {
            out = Array(UnsafeBufferPointer(start: base, count: Int(count)))
        }
        moonshine_free_embedding(embeddingPtr)
        return out
    }

    /// Cosine similarity between two embeddings of equal length, in `-1...1`.
    func calculateEmbeddingDistance(
        handle: Int32,
        embeddingA: [Float],
        embeddingB: [Float]
    ) throws -> Float {
        guard embeddingA.count == embeddingB.count, !embeddingA.isEmpty else {
            throw MoonshineError.custom(
                message: "Embedding sizes differ: \(embeddingA.count) vs \(embeddingB.count)",
                code: Int32(-3))  // MOONSHINE_ERROR_INVALID_ARGUMENT
        }
        var similarity: Float = 0
        let err: Int32 = embeddingA.withUnsafeBufferPointer { bufA in
            embeddingB.withUnsafeBufferPointer { bufB in
                moonshine_calculate_embedding_distance(
                    handle,
                    bufA.baseAddress,
                    bufB.baseAddress,
                    UInt64(embeddingA.count),
                    &similarity)
            }
        }
        try checkError(err)
        return similarity
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
