export declare class DefaultMap<K, V> extends Map<K, V> {
    private creator;
    constructor(creator: (key: K) => V);
    get(key: K): V;
    getNoInsert(key: K): V | undefined;
}
export declare class StableKeyMap<K extends {
    getStableKey(): string;
}, V> {
    [Symbol.toStringTag]: string;
    private impl;
    get size(): number;
    set(key: K, value: V): this;
    get(key: K): V | undefined;
    has(key: K): boolean;
    clear(): void;
    delete(key: K): boolean;
    forEach(callbackfn: (value: V, key: K, map: Map<K, V>) => void, thisArg?: any): void;
    entries(): IterableIterator<[K, V]>;
    keys(): IterableIterator<K>;
    values(): IterableIterator<V>;
    [Symbol.iterator](): IterableIterator<[K, V]>;
}
