import ts from "typescript";
/**
 * Return the default value of the given node.
 *
 * @param node  The TypeScript node whose default value should be extracted.
 * @returns The default value as a string.
 */
export function convertDefaultValue(node) {
    const anyNode = node;
    if (anyNode?.initializer) {
        return convertExpression(anyNode.initializer);
    }
    else {
        return undefined;
    }
}
export function convertExpression(expression) {
    switch (expression.kind) {
        case ts.SyntaxKind.StringLiteral:
        case ts.SyntaxKind.TrueKeyword:
        case ts.SyntaxKind.FalseKeyword:
        case ts.SyntaxKind.NullKeyword:
        case ts.SyntaxKind.NumericLiteral:
        case ts.SyntaxKind.BigIntLiteral:
        case ts.SyntaxKind.Identifier:
            return expression.getText();
    }
    if (ts.isPrefixUnaryExpression(expression)) {
        const inner = convertExpression(expression.operand);
        if (inner != "...") {
            return expression.getText();
        }
    }
    if (ts.isArrayLiteralExpression(expression) &&
        expression.elements.length === 0) {
        return "[]";
    }
    if (ts.isObjectLiteralExpression(expression) &&
        expression.properties.length === 0) {
        return "{}";
    }
    // a.b.c.d
    if (ts.isPropertyAccessExpression(expression)) {
        const parts = [expression.name.getText()];
        let iter = expression.expression;
        while (ts.isPropertyAccessExpression(iter)) {
            parts.unshift(iter.name.getText());
            iter = iter.expression;
        }
        if (ts.isIdentifier(iter)) {
            parts.unshift(iter.text);
            return parts.join(".");
        }
    }
    // More complex expressions are generally not useful in the documentation.
    // Show that there was a value, but not specifics.
    return "...";
}
