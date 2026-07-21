export declare const emptyArray: readonly [];
/**
 * Inserts an item into an array sorted by priority. If two items have the same priority,
 * the item will be inserted later will be placed later in the array.
 * @param arr modified by inserting item.
 * @param item
 */
export declare function insertPrioritySorted<T extends {
    priority: number;
}>(arr: T[], item: T): T[];
/**
 * Inserts an item into an array sorted by order. If two items have the same order,
 * the item inserted later will be placed later in the array.
 * The array will be sorted with lower order being placed sooner.
 * @param arr modified by inserting item.
 * @param item
 */
export declare function insertOrderSorted<T extends {
    order: number;
}>(arr: T[], item: T): T[];
/**
 * Performs a binary search of a given array, returning the index of the first item
 * for which `partition` returns true. Returns the -1 if there are no items in `arr`
 * such that `partition(item)` is true.
 * @param arr
 * @param partition should return true while less than the partition point.
 */
export declare function binaryFindPartition<T>(arr: readonly T[], partition: (item: T) => boolean): number;
/**
 * Removes an item from the array if the array exists and the item is included
 * within it.
 * @param arr
 * @param item
 */
export declare function removeIfPresent<T>(arr: T[] | undefined, item: T): void;
/**
 * Remove items in an array which match a predicate.
 * @param arr
 * @param predicate
 */
export declare function removeIf<T>(arr: T[], predicate: (item: T) => boolean): void;
/**
 * Filters out duplicate values from the given iterable.
 * @param arr
 */
export declare function unique<T>(arr: Iterable<T> | undefined): T[];
export declare function partition<T>(iter: Iterable<T>, predicate: (item: T) => boolean): [T[], T[]];
export declare function zip<T extends Iterable<any>[]>(...args: T): Iterable<{
    [K in keyof T]: T[K] extends Iterable<infer U> ? U : T[K];
}>;
export declare function filterMap<T, U>(iter: Iterable<T> | undefined, fn: (item: T) => U | undefined): U[];
export declare function firstDefined<T, U>(array: readonly T[] | undefined, callback: (element: T, index: number) => U | undefined): U | undefined;
export declare function filter<T>(array: readonly T[] | undefined, predicate: (value: T, index: number, array: readonly T[]) => boolean): readonly T[];
export declare function aggregate<T>(arr: T[], fn: (item: T) => number): number;
export declare function aggregateWithJoiner<T>(arr: T[], fn: (item: T) => number, joiner: string): number;
export declare function joinArray<T>(arr: readonly T[] | undefined, joiner: string, mapper: (item: T) => string): string;
export declare function maxElementByScore<T>(arr: readonly T[], score: (a: T) => number): T | undefined;
