import Foundation

/// Embedding model architectures for intent recognition.
///
/// Internal to the binding; intent matching is reached through ``DialogFlow``.
enum EmbeddingModelArch: UInt32 {
    case gemma300m = 0
}
