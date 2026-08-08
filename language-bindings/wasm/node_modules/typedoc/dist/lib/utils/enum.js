export function getEnumFlags(flags) {
    const result = [];
    for (let i = 1; i <= flags; i <<= 1) {
        if (flags & i) {
            result.push(i);
        }
    }
    return result;
}
// T & {} reduces inference priority
export function removeFlag(flag, remove) {
    return (flag & ~remove);
}
export function hasAllFlags(flags, check) {
    return (flags & check) === check;
}
export function hasAnyFlag(flags, check) {
    return (flags & check) !== 0;
}
export function debugFlags(Enum, flags) {
    return getEnumKeys(Enum).filter((key) => (Enum[key] & flags) === Enum[key]);
}
// Note: String enums are not handled.
export function getEnumKeys(Enum) {
    const E = Enum;
    return Object.keys(E).filter((k) => E[E[k]] === k);
}
