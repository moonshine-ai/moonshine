import Foundation

/// Base error class for all Moonshine Voice errors.
public enum MoonshineError: Error {
    case unknown(message: String = "Unknown error")
    case invalidHandle(message: String = "Invalid handle")
    case invalidArgument(message: String = "Invalid argument")
    case custom(message: String, code: Int32)
    
    /// Error code associated with the error
    public var errorCode: Int32 {
        switch self {
        case .unknown:
            return -1
        case .invalidHandle:
            return -2
        case .invalidArgument:
            return -3
        case .custom(_, let code):
            return code
        }
    }
    
    /// Human-readable error message
    public var message: String {
        switch self {
        case .unknown(let message):
            return message
        case .invalidHandle(let message):
            return message
        case .invalidArgument(let message):
            return message
        case .custom(let message, _):
            return message
        }
    }
}

/// Helper function to check error codes and throw appropriate Swift errors
internal func checkError(_ errorCode: Int32) throws {
    guard errorCode >= 0 else {
        switch errorCode {
        case -1:
            throw MoonshineError.unknown()
        case -2:
            throw MoonshineError.invalidHandle()
        case -3:
            throw MoonshineError.invalidArgument()
        default:
            throw MoonshineError.custom(message: "Unknown error code: \(errorCode)", code: errorCode)
        }
    }
}

