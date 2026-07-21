type EventHooksMomento<T extends Record<keyof T, unknown[]>, _R> = {
    __eventHooksMomentoBrand: never;
};
/**
 * Event emitter which allows listeners to return a value.
 *
 * This is beneficial for the themes since it allows plugins to modify the HTML output
 * without doing unsafe text replacement.
 *
 * Very simple event emitter class which collects the return values of its listeners.
 *
 * @example
 * ```ts
 * const x = new EventHooks<{ a: [string] }, string>()
 * x.on('a', a => a.repeat(123)) // ok, returns a string
 * x.on('b', console.log) // error, 'b' is not assignable to 'a'
 * x.on('a' a => 1) // error, returns a number but expected a string
 * ```
 */
export declare class EventHooks<T extends Record<keyof T, unknown[]>, R> {
    private _listeners;
    /**
     * Starts listening to an event.
     * @param event the event to listen to.
     * @param listener function to be called when an this event is emitted.
     * @param order optional order to insert this hook with.
     */
    on<K extends keyof T>(event: K, listener: (...args: T[K]) => R, order?: number): void;
    /**
     * Listens to a single occurrence of an event.
     * @param event the event to listen to.
     * @param listener function to be called when an this event is emitted.
     * @param order optional order to insert this hook with.
     */
    once<K extends keyof T>(event: K, listener: (...args: T[K]) => R, order?: number): void;
    /**
     * Stops listening to an event.
     * @param event the event to stop listening to.
     * @param listener the function to remove from the listener array.
     */
    off<K extends keyof T>(event: K, listener: (...args: T[K]) => R): void;
    /**
     * Emits an event to all currently subscribed listeners.
     * @param event the event to emit.
     * @param args any arguments required for the event.
     */
    emit<K extends keyof T>(event: K, ...args: T[K]): R[];
    saveMomento(): EventHooksMomento<T, R>;
    restoreMomento(momento: EventHooksMomento<T, R>): void;
}
export {};
