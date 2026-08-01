import Foundation

/// Embedding model architectures.
///
/// Internal to the binding; phrase matching is reached through ``AgentFlow``.
enum EmbeddingModelArch: UInt32 {
    case gemma300m = 0
}
