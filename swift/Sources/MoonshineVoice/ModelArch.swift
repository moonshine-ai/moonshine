import Foundation

/// Model architecture types supported by Moonshine Voice.
public enum ModelArch: UInt32 {
    case tiny = 0
    case base = 1
    case tinyStreaming = 2
    case baseStreaming = 3
    case smallStreaming = 4
    case mediumStreaming = 5
}

