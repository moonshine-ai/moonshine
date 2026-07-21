/**
 * Intentionally very simple event emitter.
 *
 * @privateRemarks
 * This is essentially a stripped down copy of EventHooks in hooks.ts.
 */
export declare class EventDispatcher<T extends Record<keyof T, unknown[]>> {
    private _listeners;
    /**
     * Starts listening to an event.
     * @param event the event to listen to.
     * @param listener function to be called when an this event is emitted.
     * @param priority optional priority to insert this hook with.
     */
    on<K extends keyof T>(event: K, listener: (this: undefined, ...args: T[K]) => void, priority?: number): void;
    /**
     * Stops listening to an event.
     * @param event the event to stop listening to.
     * @param listener the function to remove from the listener array.
     */
    off<K extends keyof T>(event: K, listener: (this: undefined, ...args: T[K]) => void): void;
    /**
     * Emits an event to all currently subscribed listeners.
     * @param event the event to emit.
     * @param args any arguments required for the event.
     */
    trigger<K extends keyof T>(event: K, ...args: T[K]): void;
}
