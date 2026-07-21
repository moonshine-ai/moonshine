export const emptyArray = [];
/**
 * Inserts an item into an array sorted by priority. If two items have the same priority,
 * the item will be inserted later will be placed later in the array.
 * @param arr modified by inserting item.
 * @param item
 */
export function insertPrioritySorted(arr, item) {
    const index = binaryFindPartition(arr, (v) => v.priority < item.priority);
    arr.splice(index === -1 ? arr.length : index, 0, item);
    return arr;
}
/**
 * Inserts an item into an array sorted by order. If two items have the same order,
 * the item inserted later will be placed later in the array.
 * The array will be sorted with lower order being placed sooner.
 * @param arr modified by inserting item.
 * @param item
 */
export function insertOrderSorted(arr, item) {
    const index = binaryFindPartition(arr, (v) => v.order > item.order);
    arr.splice(index === -1 ? arr.length : index, 0, item);
    return arr;
}
/**
 * Performs a binary search of a given array, returning the index of the first item
 * for which `partition` returns true. Returns the -1 if there are no items in `arr`
 * such that `partition(item)` is true.
 * @param arr
 * @param partition should return true while less than the partition point.
 */
export function binaryFindPartition(arr, partition) {
    if (arr.length === 0) {
        return -1;
    }
    let low = 0, high = arr.length - 1;
    while (high > low) {
        const mid = low + Math.floor((high - low) / 2);
        if (partition(arr[mid])) {
            high = mid;
        }
        else {
            low = mid + 1;
        }
    }
    return partition(arr[low]) ? low : -1;
}
/**
 * Removes an item from the array if the array exists and the item is included
 * within it.
 * @param arr
 * @param item
 */
export function removeIfPresent(arr, item) {
    if (!arr) {
        return;
    }
    const index = arr.indexOf(item);
    if (index !== -1) {
        arr.splice(index, 1);
    }
}
/**
 * Remove items in an array which match a predicate.
 * @param arr
 * @param predicate
 */
export function removeIf(arr, predicate) {
    for (let i = 0; i < arr.length; i++) {
        if (predicate(arr[i])) {
            arr.splice(i, 1);
            i--;
        }
    }
}
/**
 * Filters out duplicate values from the given iterable.
 * @param arr
 */
export function unique(arr) {
    return Array.from(new Set(arr));
}
export function partition(iter, predicate) {
    const left = [];
    const right = [];
    for (const item of iter) {
        if (predicate(item)) {
            left.push(item);
        }
        else {
            right.push(item);
        }
    }
    return [left, right];
}
export function* zip(...args) {
    const iterators = args.map((x) => x[Symbol.iterator]());
    for (;;) {
        const next = iterators.map((i) => i.next());
        if (next.some((v) => v.done)) {
            break;
        }
        yield next.map((v) => v.value);
    }
}
export function filterMap(iter, fn) {
    const result = [];
    for (const item of iter || []) {
        const newItem = fn(item);
        if (newItem !== void 0) {
            result.push(newItem);
        }
    }
    return result;
}
export function firstDefined(array, callback) {
    if (array === undefined) {
        return undefined;
    }
    for (let i = 0; i < array.length; i++) {
        const result = callback(array[i], i);
        if (result !== undefined) {
            return result;
        }
    }
    return undefined;
}
export function filter(array, predicate) {
    return array ? array.filter(predicate) : emptyArray;
}
export function aggregate(arr, fn) {
    return arr.reduce((sum, it) => sum + fn(it), 0);
}
export function aggregateWithJoiner(arr, fn, joiner) {
    return (arr.reduce((sum, it) => sum + fn(it), 0) +
        (arr.length - 1) * joiner.length);
}
export function joinArray(arr, joiner, mapper) {
    if (arr?.length) {
        return arr.map(mapper).join(joiner);
    }
    return "";
}
export function maxElementByScore(arr, score) {
    if (arr.length === 0) {
        return undefined;
    }
    let largest = arr[0];
    let largestScore = score(arr[0]);
    for (let i = 1; i < arr.length; ++i) {
        const itemScore = score(arr[i]);
        if (itemScore > largestScore) {
            largest = arr[i];
            largestScore = itemScore;
        }
    }
    return largest;
}
