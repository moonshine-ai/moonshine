const opt = Symbol();
/**
 * Symbol that may be placed on a schema object to define how additional properties are handled.
 * By default, additional properties are not checked.
 */
export const additionalProperties = Symbol();
export function validate(schema, obj) {
    let type = schema;
    if (opt in schema) {
        if (obj == null) {
            return true;
        }
        type = schema[opt];
    }
    if (type === String) {
        return typeof obj === "string";
    }
    if (type === Number) {
        return typeof obj === "number";
    }
    if (type === Boolean) {
        return typeof obj === "boolean";
    }
    if (typeof type === "function") {
        return type(obj);
    }
    if (Array.isArray(type)) {
        if (type[0] === Array) {
            return (Array.isArray(obj) &&
                obj.every((item) => validate(type[1], item)));
        }
        return type.includes(obj);
    }
    if (additionalProperties in schema &&
        !schema[additionalProperties]) {
        if (Object.keys(obj).some((key) => !(key in schema))) {
            return false;
        }
    }
    return (!!obj &&
        typeof obj === "object" &&
        !Array.isArray(obj) &&
        Object.entries(type).every(([key, prop]) => validate(prop, obj[key])));
}
export function optional(x) {
    return { [opt]: x };
}
export function isTagString(x) {
    return typeof x === "string" && /^@[a-zA-Z][a-zA-Z0-9]*$/.test(x);
}
