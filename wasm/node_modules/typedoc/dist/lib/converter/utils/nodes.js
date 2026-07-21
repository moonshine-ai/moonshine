import ts from "typescript";
export function isNamedNode(node) {
    const name = node.name;
    return !!name && (ts.isMemberName(name) || ts.isComputedPropertyName(name));
}
export function getHeritageTypes(declarations, kind) {
    const exprs = declarations.flatMap((d) => (d.heritageClauses ?? [])
        .filter((hc) => hc.token === kind)
        .flatMap((hc) => hc.types));
    const seenTexts = new Set();
    return exprs.filter((expr) => {
        const text = expr.getText();
        if (seenTexts.has(text)) {
            return false;
        }
        seenTexts.add(text);
        return true;
    });
}
export function isObjectType(type) {
    return typeof type.objectFlags === "number";
}
export function isTypeReference(type) {
    return (isObjectType(type) &&
        (type.objectFlags & ts.ObjectFlags.Reference) !== 0);
}
