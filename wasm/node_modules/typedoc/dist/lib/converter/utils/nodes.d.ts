import ts from "typescript";
export declare function isNamedNode(node: unknown): node is {
    readonly name: ts.Identifier | ts.PrivateIdentifier | ts.ComputedPropertyName;
};
export declare function getHeritageTypes(declarations: readonly (ts.ClassDeclaration | ts.InterfaceDeclaration)[], kind: ts.SyntaxKind.ImplementsKeyword | ts.SyntaxKind.ExtendsKeyword): ts.ExpressionWithTypeArguments[];
export declare function isObjectType(type: ts.Type): type is ts.ObjectType;
export declare function isTypeReference(type: ts.Type): type is ts.TypeReference;
