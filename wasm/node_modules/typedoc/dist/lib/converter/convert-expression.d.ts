import ts from "typescript";
/**
 * Return the default value of the given node.
 *
 * @param node  The TypeScript node whose default value should be extracted.
 * @returns The default value as a string.
 */
export declare function convertDefaultValue(node: ts.Declaration | undefined): string | undefined;
export declare function convertExpression(expression: ts.Expression): string;
