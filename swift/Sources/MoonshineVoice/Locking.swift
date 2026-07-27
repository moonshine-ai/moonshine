import Foundation

/// A mutex with a scoped `withLock`, usable from `async` code.
///
/// `NSLock.lock()` is unavailable in asynchronous contexts (holding it across a
/// suspension point would block a cooperative thread), and its own `withLock`
/// needs a newer deployment target than this package has, so the engines share
/// this small wrapper. Nothing here ever awaits while holding the lock.
final class Mutex: @unchecked Sendable {
    private let underlying = NSLock()

    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        underlying.lock()
        defer { underlying.unlock() }
        return try body()
    }
}
