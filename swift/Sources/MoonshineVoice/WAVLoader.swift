import Foundation

/// Error types for WAV file loading
public enum WAVError: Error {
    case fileNotFound(String)
    case invalidRIFF
    case invalidWAVE
    case noFmtChunk
    case noDataChunk
    case unsupportedFormat(Int)
    case unsupportedBitDepth(Int)
}

/// Result of loading a WAV file
public struct WAVData {
    public let audioData: [Float]
    public let sampleRate: Int
    
    public init(audioData: [Float], sampleRate: Int) {
        self.audioData = audioData
        self.sampleRate = sampleRate
    }
}

/// Load a WAV file and return audio data as float array and sample rate.
/// Supports 16-bit, 24-bit, and 32-bit PCM WAV files. Audio samples are normalized to [-1.0, 1.0].
public func loadWAVFile(_ filePath: String) throws -> WAVData {
    let url = URL(fileURLWithPath: filePath)
    
    guard FileManager.default.fileExists(atPath: filePath) else {
        throw WAVError.fileNotFound(filePath)
    }
    
    let data = try Data(contentsOf: url)
    var offset = 0
    
    // Read RIFF header
    guard data.count >= 4 else { throw WAVError.invalidRIFF }
    let riffHeader = data.subdata(in: offset..<offset+4)
    guard String(data: riffHeader, encoding: .ascii) == "RIFF" else {
        throw WAVError.invalidRIFF
    }
    offset += 4
    
    // Read chunk size (skip it)
    guard data.count >= offset + 4 else { throw WAVError.invalidRIFF }
    offset += 4
    
    // Read WAVE header
    guard data.count >= offset + 4 else { throw WAVError.invalidWAVE }
    let waveHeader = data.subdata(in: offset..<offset+4)
    guard String(data: waveHeader, encoding: .ascii) == "WAVE" else {
        throw WAVError.invalidWAVE
    }
    offset += 4
    
    // Find fmt and data chunks
    var foundFmt = false
    var foundData = false
    var audioFormat: UInt16 = 0
    var numChannels: UInt16 = 0
    var sampleRate: UInt32 = 0
    var bitsPerSample: UInt16 = 0
    var dataOffset = 0
    var dataSize = 0
    
    while offset < data.count - 8 {
        // Read chunk ID
        guard data.count >= offset + 4 else { break }
        let chunkID = data.subdata(in: offset..<offset+4)
        offset += 4
        
        // Read chunk size
        guard data.count >= offset + 4 else { break }
        let chunkSizeData = data.subdata(in: offset..<offset+4)
        let chunkSize = chunkSizeData.withUnsafeBytes { bytes in
            UInt32(littleEndian: bytes.load(as: UInt32.self))
        }
        offset += 4
        
        let chunkIDString = String(data: chunkID, encoding: .ascii) ?? ""
        
        if chunkIDString == "fmt " {
            foundFmt = true
            guard chunkSize >= 16 else { throw WAVError.noFmtChunk }
            guard data.count >= offset + Int(chunkSize) else { throw WAVError.noFmtChunk }
            
            // Read audio format (1 = PCM)
            let fmtData = data.subdata(in: offset..<offset+Int(chunkSize))
            audioFormat = fmtData.withUnsafeBytes { bytes in
                UInt16(littleEndian: bytes.load(fromByteOffset: 0, as: UInt16.self))
            }
            numChannels = fmtData.withUnsafeBytes { bytes in
                UInt16(littleEndian: bytes.load(fromByteOffset: 2, as: UInt16.self))
            }
            sampleRate = fmtData.withUnsafeBytes { bytes in
                UInt32(littleEndian: bytes.load(fromByteOffset: 4, as: UInt32.self))
            }
            bitsPerSample = fmtData.withUnsafeBytes { bytes in
                UInt16(littleEndian: bytes.load(fromByteOffset: 14, as: UInt16.self))
            }
            
            offset += Int(chunkSize)
            
            if audioFormat != 1 {
                throw WAVError.unsupportedFormat(Int(audioFormat))
            }
            
            if bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32 {
                throw WAVError.unsupportedBitDepth(Int(bitsPerSample))
            }
        } else if chunkIDString == "data" {
            foundData = true
            dataOffset = offset
            dataSize = Int(chunkSize)
            break
        } else {
            // Skip unknown chunks
            offset += Int(chunkSize)
        }
    }
    
    guard foundFmt else { throw WAVError.noFmtChunk }
    guard foundData else { throw WAVError.noDataChunk }
    
    // Read audio data
    let bytesPerSample = Int(bitsPerSample) / 8
    let bytesPerFrame = bytesPerSample * Int(numChannels)
    let numFrames = dataSize / bytesPerFrame
    
    guard data.count >= dataOffset + dataSize else {
        throw WAVError.noDataChunk
    }
    
    let audioDataBytes = data.subdata(in: dataOffset..<dataOffset + dataSize)
    var audioData: [Float] = []
    audioData.reserveCapacity(numFrames)
    
    if numChannels > 1 {
        // Multi-channel: mix down to mono by averaging
        var byteOffset = 0
        for _ in 0..<numFrames {
            var channelSum: Float = 0.0
            for _ in 0..<numChannels {
                let sample: Float
                if bitsPerSample == 16 {
                    let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+2)
                    let sampleInt16 = sampleData.withUnsafeBytes { bytes in
                        Int16(littleEndian: bytes.load(as: Int16.self))
                    }
                    sample = Float(sampleInt16) / 32768.0
                    byteOffset += 2
                } else if bitsPerSample == 24 {
                    let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+3)
                    let bytes = [UInt8](sampleData)
                    var sampleInt32 = Int32(bytes[0]) | (Int32(bytes[1]) << 8) | (Int32(bytes[2]) << 16)
                    // Sign extend
                    if sampleInt32 & 0x800000 != 0 {
                        sampleInt32 |= Int32(bitPattern: 0xFF000000)
                    }
                    sample = Float(sampleInt32) / 8388608.0
                    byteOffset += 3
                } else { // 32-bit
                    let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+4)
                    let sampleInt32 = sampleData.withUnsafeBytes { bytes in
                        Int32(littleEndian: bytes.load(as: Int32.self))
                    }
                    sample = Float(sampleInt32) / 2147483648.0
                    byteOffset += 4
                }
                channelSum += sample
            }
            audioData.append(channelSum / Float(numChannels))
        }
    } else {
        // Mono audio
        var byteOffset = 0
        for _ in 0..<numFrames {
            let sample: Float
            if bitsPerSample == 16 {
                let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+2)
                let sampleInt16 = sampleData.withUnsafeBytes { bytes in
                    Int16(littleEndian: bytes.load(as: Int16.self))
                }
                sample = Float(sampleInt16) / 32768.0
                byteOffset += 2
            } else if bitsPerSample == 24 {
                let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+3)
                let bytes = [UInt8](sampleData)
                var sampleInt32 = Int32(bytes[0]) | (Int32(bytes[1]) << 8) | (Int32(bytes[2]) << 16)
                // Sign extend
                if sampleInt32 & 0x800000 != 0 {
                    sampleInt32 |= Int32(bitPattern: 0xFF000000)
                }
                sample = Float(sampleInt32) / 8388608.0
                byteOffset += 3
            } else { // 32-bit
                let sampleData = audioDataBytes.subdata(in: byteOffset..<byteOffset+4)
                let sampleInt32 = sampleData.withUnsafeBytes { bytes in
                    Int32(littleEndian: bytes.load(as: Int32.self))
                }
                sample = Float(sampleInt32) / 2147483648.0
                byteOffset += 4
            }
            audioData.append(sample)
        }
    }
    
    return WAVData(audioData: audioData, sampleRate: Int(sampleRate))
}

