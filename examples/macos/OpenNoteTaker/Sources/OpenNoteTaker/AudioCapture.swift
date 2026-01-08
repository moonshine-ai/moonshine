import AVFoundation
@preconcurrency import ScreenCaptureKit
import Combine

@MainActor
class AudioCapture {
    var onAudioData: (([Float], Double) -> Void)?
    
    private var audioEngine: AVAudioEngine?
    private var screenCapture: SCStream?
    private var isCapturing = false
    private let sampleRate: Double = 16000
    private let targetFormat: AVAudioFormat
    
    private var micAudioBuffer: [Float] = []
    private var systemAudioBuffer: [Float] = []
    private let bufferLock = NSLock()
    
    init() {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            fatalError("Failed to create audio format")
        }
        self.targetFormat = format
    }
    
    func start() async throws {
        guard !isCapturing else { return }
        
        // Start microphone capture
        try startMicrophoneCapture()
        
        // Start system audio capture
        try await startSystemAudioCapture()
        
        isCapturing = true
    }
    
    func stop() {
        guard isCapturing else { return }
        
        stopMicrophoneCapture()
        stopSystemAudioCapture()
        
        isCapturing = false
    }
    
    private func startMicrophoneCapture() throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        
        // Request microphone permission
        let permissionStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        if permissionStatus == .denied {
            throw NSError(domain: "AudioCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
        }
        
        if permissionStatus == .notDetermined {
            var permissionGranted = false
            let semaphore = DispatchSemaphore(value: 0)
            
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                permissionGranted = granted
                semaphore.signal()
            }
            
            semaphore.wait()
            
            if !permissionGranted {
                throw NSError(domain: "AudioCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
            }
        }
        
        let needsConversion = inputFormat.sampleRate != targetFormat.sampleRate ||
                             inputFormat.channelCount != targetFormat.channelCount ||
                             inputFormat.commonFormat != targetFormat.commonFormat
        
        let converter = needsConversion ? AVAudioConverter(from: inputFormat, to: targetFormat) : nil
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, time in
            guard let self = self, self.isCapturing else { return }
            
            var audioData: [Float] = []
            
            if let converter = converter {
                let capacity = AVAudioFrameCount(
                    Double(buffer.frameLength) * self.targetFormat.sampleRate / inputFormat.sampleRate
                )
                guard let convertedBuffer = AVAudioPCMBuffer(
                    pcmFormat: self.targetFormat,
                    frameCapacity: capacity
                ) else { return }
                
                var error: NSError?
                let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                    outStatus.pointee = .haveData
                    return buffer
                }
                
                converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)
                
                if let error = error {
                    print("Audio conversion error: \(error.localizedDescription)")
                    return
                }
                
                guard let channelData = convertedBuffer.floatChannelData else { return }
                let frameLength = Int(convertedBuffer.frameLength)
                audioData.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
            } else {
                guard let channelData = buffer.floatChannelData else { return }
                let frameLength = Int(buffer.frameLength)
                audioData.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
            }
            
            self.bufferLock.lock()
            self.micAudioBuffer.append(contentsOf: audioData)
            self.bufferLock.unlock()
            
            self.processBuffers()
        }
        
        try engine.start()
        self.audioEngine = engine
    }
    
    private func stopMicrophoneCapture() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        bufferLock.lock()
        micAudioBuffer.removeAll()
        bufferLock.unlock()
    }
    
    private func startSystemAudioCapture() async throws {
        // Request screen recording permission for system audio
        // Note: ScreenCaptureKit requires screen recording permission
        // Get available displays
        let availableContent = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = availableContent.displays.first else {
            throw NSError(domain: "AudioCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not get display"])
        }
        
        let contentFilter = SCContentFilter(
            display: display,
            excludingWindows: []
        )
        
        let streamConfig = SCStreamConfiguration()
        streamConfig.capturesAudio = true
        streamConfig.excludesCurrentProcessAudio = false
        streamConfig.sampleRate = Int(sampleRate)
        streamConfig.channelCount = 1
        
        let audioStreamOutput = AudioStreamOutput { [weak self] audioData in
            guard let self = self, self.isCapturing else { return }
            self.bufferLock.lock()
            self.systemAudioBuffer.append(contentsOf: audioData)
            self.bufferLock.unlock()
            self.processBuffers()
        }
        
        let stream = SCStream(filter: contentFilter, configuration: streamConfig, delegate: nil)
        
        try stream.addStreamOutput(audioStreamOutput, type: SCStreamOutputType.audio, sampleHandlerQueue: DispatchQueue(label: "audio.sample"))
        try await stream.startCapture()
        
        self.screenCapture = stream
    }
    
    private func stopSystemAudioCapture() {
        screenCapture?.stopCapture()
        screenCapture = nil
        bufferLock.lock()
        systemAudioBuffer.removeAll()
        bufferLock.unlock()
    }
    
    private func processBuffers() {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        
        // Mix microphone and system audio
        let minLength = min(micAudioBuffer.count, systemAudioBuffer.count)
        if minLength > 0 {
            var mixedAudio: [Float] = []
            mixedAudio.reserveCapacity(minLength)
            
            for i in 0..<minLength {
                // Simple mixing: average the two sources
                let mixed = (micAudioBuffer[i] + systemAudioBuffer[i]) / 2.0
                mixedAudio.append(mixed)
            }
            
            // Remove processed samples
            micAudioBuffer.removeFirst(minLength)
            systemAudioBuffer.removeFirst(minLength)
            
            // Send to callback
            onAudioData?(mixedAudio, sampleRate)
        } else if !micAudioBuffer.isEmpty {
            // Only microphone audio available
            let audio = Array(micAudioBuffer)
            micAudioBuffer.removeAll()
            onAudioData?(audio, sampleRate)
        } else if !systemAudioBuffer.isEmpty {
            // Only system audio available
            let audio = Array(systemAudioBuffer)
            systemAudioBuffer.removeAll()
            onAudioData?(audio, sampleRate)
        }
    }
}

class AudioStreamOutput: NSObject, SCStreamOutput {
    let onAudio: ([Float]) -> Void
    
    init(onAudio: @escaping ([Float]) -> Void) {
        self.onAudio = onAudio
    }
    
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
        let audioFormatList = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
        guard let audioFormat = audioFormatList?.pointee else { return }
        
        var audioBufferList = AudioBufferList()
        var blockBuffer: CMBlockBuffer?
        
        var bufferListSize = MemoryLayout<AudioBufferList>.size
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &bufferListSize,
            bufferListOut: &audioBufferList,
            bufferListSize: bufferListSize,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        
        guard status == noErr else { return }
        defer { blockBuffer = nil }
        
        let bufferCount = Int(audioBufferList.mNumberBuffers)
        guard bufferCount > 0 else { return }
        
        // Access the first buffer directly
        let firstBuffer = audioBufferList.mBuffers
        guard let data = firstBuffer.mData else { return }
        
        let frameLength = Int(firstBuffer.mDataByteSize) / MemoryLayout<Float>.size
        let audioData = UnsafeBufferPointer<Float>(
            start: data.assumingMemoryBound(to: Float.self),
            count: frameLength
        )
        
        // Convert to mono if needed
        var monoAudio: [Float] = []
        if audioFormat.mChannelsPerFrame > 1 {
            // Average channels for mono
            let samplesPerChannel = frameLength / Int(audioFormat.mChannelsPerFrame)
            monoAudio.reserveCapacity(samplesPerChannel)
            for i in 0..<samplesPerChannel {
                var sum: Float = 0
                for channel in 0..<Int(audioFormat.mChannelsPerFrame) {
                    let index = i * Int(audioFormat.mChannelsPerFrame) + channel
                    if index < frameLength {
                        sum += audioData[index]
                    }
                }
                monoAudio.append(sum / Float(audioFormat.mChannelsPerFrame))
            }
        } else {
            monoAudio = Array(audioData)
        }
        
        // Resample if needed
        if audioFormat.mSampleRate != 16000 {
            // Simple linear resampling (for production, use a proper resampler)
            let ratio = 16000.0 / audioFormat.mSampleRate
            let targetLength = Int(Double(monoAudio.count) * ratio)
            var resampled: [Float] = []
            resampled.reserveCapacity(targetLength)
            
            for i in 0..<targetLength {
                let sourceIndex = Double(i) / ratio
                let index = Int(sourceIndex)
                let fraction = sourceIndex - Double(index)
                
                if index + 1 < monoAudio.count {
                    let interpolated = monoAudio[index] * Float(1.0 - fraction) + monoAudio[index + 1] * Float(fraction)
                    resampled.append(interpolated)
                } else if index < monoAudio.count {
                    resampled.append(monoAudio[index])
                }
            }
            onAudio(resampled)
        } else {
            onAudio(monoAudio)
        }
    }
}

