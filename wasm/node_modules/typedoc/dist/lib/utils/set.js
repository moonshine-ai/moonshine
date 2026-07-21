export function setIntersection(a, b) {
    const result = new Set();
    for (const elem of a) {
        if (b.has(elem)) {
            result.add(elem);
        }
    }
    return result;
}
export function setDifference(a, b) {
    const result = new Set(a);
    for (const elem of b) {
        result.delete(elem);
    }
    return result;
}
