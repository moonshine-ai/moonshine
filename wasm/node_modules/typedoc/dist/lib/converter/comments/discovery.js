import ts from "typescript";
import { ReflectionKind } from "../../models/index.js";
import { assertNever } from "../../utils/index.js";
import { CommentStyle } from "../../utils/options/declaration.js";
import { nicePath } from "../../utils/paths.js";
import { ok } from "assert";
import { filter, firstDefined } from "../../utils/array.js";
const variablePropertyKinds = [
    ts.SyntaxKind.PropertyDeclaration,
    ts.SyntaxKind.PropertySignature,
    ts.SyntaxKind.BinaryExpression,
    ts.SyntaxKind.PropertyAssignment,
    ts.SyntaxKind.ShorthandPropertyAssignment,
    // class X { constructor(/** Comment */ readonly z: string) }
    ts.SyntaxKind.Parameter,
    // Variable values
    ts.SyntaxKind.VariableDeclaration,
    ts.SyntaxKind.BindingElement,
    ts.SyntaxKind.ExportAssignment,
    ts.SyntaxKind.PropertyAccessExpression,
];
// Note: This does NOT include JSDoc syntax kinds. This is important!
// Comments from @typedef and @callback tags are handled specially by
// the JSDoc converter because we only want part of the comment when
// getting them.
const wantedKinds = {
    [ReflectionKind.Project]: [
        ts.SyntaxKind.SourceFile,
        ts.SyntaxKind.ModuleDeclaration,
    ],
    [ReflectionKind.Module]: [
        ts.SyntaxKind.SourceFile,
        ts.SyntaxKind.ModuleDeclaration,
    ],
    [ReflectionKind.Namespace]: [
        ts.SyntaxKind.ModuleDeclaration,
        ts.SyntaxKind.SourceFile,
        ts.SyntaxKind.BindingElement,
        ts.SyntaxKind.ExportSpecifier,
        ts.SyntaxKind.NamespaceExport,
        // @namespace support
        ts.SyntaxKind.VariableDeclaration,
        ts.SyntaxKind.BindingElement,
        ts.SyntaxKind.ExportAssignment,
        ts.SyntaxKind.PropertyAccessExpression,
        ts.SyntaxKind.PropertyDeclaration,
        ts.SyntaxKind.PropertyAssignment,
        ts.SyntaxKind.ShorthandPropertyAssignment,
    ],
    [ReflectionKind.Enum]: [
        ts.SyntaxKind.EnumDeclaration,
        ts.SyntaxKind.VariableDeclaration,
    ],
    [ReflectionKind.EnumMember]: [
        ts.SyntaxKind.EnumMember,
        // These here so that @enum gets comments
        ts.SyntaxKind.PropertyAssignment,
        ts.SyntaxKind.PropertySignature,
    ],
    [ReflectionKind.Variable]: variablePropertyKinds,
    [ReflectionKind.Function]: [
        ts.SyntaxKind.FunctionDeclaration,
        ts.SyntaxKind.BindingElement,
        ts.SyntaxKind.VariableDeclaration,
        ts.SyntaxKind.ExportAssignment,
        ts.SyntaxKind.PropertyAccessExpression,
        ts.SyntaxKind.PropertyDeclaration,
        ts.SyntaxKind.PropertyAssignment,
        ts.SyntaxKind.ShorthandPropertyAssignment,
    ],
    [ReflectionKind.Class]: [
        ts.SyntaxKind.ClassDeclaration,
        ts.SyntaxKind.BindingElement,
        // If marked with @class
        ts.SyntaxKind.VariableDeclaration,
        ts.SyntaxKind.ExportAssignment,
        ts.SyntaxKind.FunctionDeclaration,
    ],
    [ReflectionKind.Interface]: [
        ts.SyntaxKind.InterfaceDeclaration,
        ts.SyntaxKind.TypeAliasDeclaration,
    ],
    [ReflectionKind.Constructor]: [ts.SyntaxKind.Constructor],
    [ReflectionKind.Property]: variablePropertyKinds,
    [ReflectionKind.Method]: [
        ts.SyntaxKind.FunctionDeclaration,
        ts.SyntaxKind.MethodDeclaration,
    ],
    [ReflectionKind.CallSignature]: [
        ts.SyntaxKind.FunctionDeclaration,
        ts.SyntaxKind.VariableDeclaration,
        ts.SyntaxKind.MethodDeclaration,
        ts.SyntaxKind.MethodDeclaration,
        ts.SyntaxKind.PropertyDeclaration,
        ts.SyntaxKind.PropertySignature,
        ts.SyntaxKind.CallSignature,
    ],
    [ReflectionKind.IndexSignature]: [ts.SyntaxKind.IndexSignature],
    [ReflectionKind.ConstructorSignature]: [ts.SyntaxKind.ConstructSignature],
    [ReflectionKind.Parameter]: [ts.SyntaxKind.Parameter],
    [ReflectionKind.TypeLiteral]: [ts.SyntaxKind.TypeLiteral],
    [ReflectionKind.TypeParameter]: [ts.SyntaxKind.TypeParameter],
    [ReflectionKind.Accessor]: [ts.SyntaxKind.PropertyDeclaration],
    [ReflectionKind.GetSignature]: [ts.SyntaxKind.GetAccessor],
    [ReflectionKind.SetSignature]: [ts.SyntaxKind.SetAccessor],
    [ReflectionKind.TypeAlias]: [ts.SyntaxKind.TypeAliasDeclaration],
    [ReflectionKind.Reference]: [
        ts.SyntaxKind.NamespaceExport,
        ts.SyntaxKind.ExportSpecifier,
    ],
    // Non-TS kind, will never have comments.
    [ReflectionKind.Document]: [],
};
export function discoverFileComments(node, commentStyle) {
    const text = node.text;
    const comments = collectCommentRanges(ts.getLeadingCommentRanges(text, node.pos));
    const selectedDocComments = comments.filter((ranges) => permittedRange(text, ranges, commentStyle));
    return selectedDocComments.map((ranges) => {
        return {
            file: node,
            ranges,
            jsDoc: findJsDocForComment(node, ranges),
            inheritedFromParentDeclaration: false,
        };
    });
}
export function discoverNodeComment(node, commentStyle) {
    const text = node.getSourceFile().text;
    const comments = collectCommentRanges(ts.getLeadingCommentRanges(text, node.pos));
    comments.reverse();
    const selectedDocComment = comments.find((ranges) => permittedRange(text, ranges, commentStyle));
    if (selectedDocComment) {
        return {
            file: node.getSourceFile(),
            ranges: selectedDocComment,
            jsDoc: findJsDocForComment(node, selectedDocComment),
            inheritedFromParentDeclaration: false,
        };
    }
}
function checkCommentDeclarations(commentNodes, reverse, commentStyle) {
    const discovered = [];
    for (const { node, inheritedFromParentDeclaration } of commentNodes) {
        const text = node.getSourceFile().text;
        const comments = collectCommentRanges(ts.getLeadingCommentRanges(text, node.pos));
        if (reverse) {
            comments.reverse();
        }
        const selectedDocComment = comments.find((ranges) => permittedRange(text, ranges, commentStyle));
        if (selectedDocComment) {
            discovered.push({
                file: node.getSourceFile(),
                ranges: selectedDocComment,
                jsDoc: findJsDocForComment(node, selectedDocComment),
                inheritedFromParentDeclaration,
            });
        }
    }
    return discovered;
}
export function discoverComment(symbol, kind, logger, commentStyle, checker, declarationWarnings) {
    // For a module comment, we want the first one defined in the file,
    // not the last one, since that will apply to the import or declaration.
    const reverse = !symbol.declarations?.some(ts.isSourceFile);
    const wantedDeclarations = filter(symbol.declarations, (decl) => wantedKinds[kind].includes(decl.kind));
    const commentNodes = wantedDeclarations.flatMap((decl) => declarationToCommentNodes(decl, checker));
    // Special behavior here!
    // Signatures and symbols have two distinct discovery methods as of TypeDoc 0.26.
    // This method discovers comments for symbols, and function-likes will only have
    // a symbol comment if there is more than one signature (== more than one declaration)
    // and there is a comment on the implementation signature.
    if (kind & ReflectionKind.ContainsCallSignatures) {
        const canHaveOverloads = wantedDeclarations.some((node) => [
            ts.SyntaxKind.FunctionDeclaration,
            ts.SyntaxKind.MethodDeclaration,
            ts.SyntaxKind.Constructor,
        ].includes(node.kind));
        const isOverloaded = canHaveOverloads && wantedDeclarations.length > 1;
        if (isOverloaded) {
            commentNodes.length = 0;
            const implementationNode = wantedDeclarations.find((node) => node.body);
            if (implementationNode) {
                commentNodes.push({
                    node: implementationNode,
                    inheritedFromParentDeclaration: false,
                });
            }
        }
        else if (canHaveOverloads) {
            // Single signature function, function reflection doesn't get a comment,
            // the signatures do.
            commentNodes.length = 0;
        }
        else {
            // Variable declaration which happens to include signatures.
        }
    }
    const discovered = checkCommentDeclarations(commentNodes, reverse, commentStyle);
    switch (discovered.length) {
        case 0:
            return undefined;
        case 1:
            return discovered[0];
        default: {
            if (discovered.filter((n) => !n.inheritedFromParentDeclaration)
                .length > 1 &&
                (declarationWarnings ||
                    discovered.some((dc) => !dc.file.isDeclarationFile))) {
                logger.warn(logger.i18n.symbol_0_has_multiple_declarations_with_comment(symbol.name));
                const locations = discovered.map(({ file, ranges: [{ pos }] }) => {
                    const path = nicePath(file.fileName);
                    const line = ts.getLineAndCharacterOfPosition(file, pos).line +
                        1;
                    return `${path}:${line}`;
                });
                logger.info(logger.i18n.comments_for_0_are_declared_at_1(symbol.name, locations.join("\n\t")));
            }
            return discovered[0];
        }
    }
}
export function discoverSignatureComment(declaration, checker, commentStyle) {
    for (const { node, inheritedFromParentDeclaration, } of declarationToCommentNodes(declaration, checker)) {
        if (ts.isJSDocSignature(node)) {
            const comment = node.parent.parent;
            ok(ts.isJSDoc(comment));
            return {
                file: node.getSourceFile(),
                ranges: [
                    {
                        kind: ts.SyntaxKind.MultiLineCommentTrivia,
                        pos: comment.pos,
                        end: comment.end,
                    },
                ],
                jsDoc: comment,
                inheritedFromParentDeclaration,
            };
        }
        const text = node.getSourceFile().text;
        const comments = collectCommentRanges(ts.getLeadingCommentRanges(text, node.pos));
        comments.reverse();
        const comment = comments.find((ranges) => permittedRange(text, ranges, commentStyle));
        if (comment) {
            return {
                file: node.getSourceFile(),
                ranges: comment,
                jsDoc: findJsDocForComment(node, comment),
                inheritedFromParentDeclaration,
            };
        }
    }
}
function findJsDocForComment(node, ranges) {
    if (ranges[0].kind === ts.SyntaxKind.MultiLineCommentTrivia) {
        const jsDocs = ts
            .getJSDocCommentsAndTags(node)
            .map((doc) => ts.findAncestor(doc, ts.isJSDoc));
        if (ts.isSourceFile(node)) {
            if (node.statements.length) {
                jsDocs.push(...ts
                    .getJSDocCommentsAndTags(node.statements[0])
                    .map((doc) => ts.findAncestor(doc, ts.isJSDoc)));
            }
        }
        return jsDocs.find((doc) => doc.pos === ranges[0].pos);
    }
}
/**
 * Check whether the given module declaration is the topmost.
 *
 * This function returns TRUE if there is no trailing module defined, in
 * the following example this would be the case only for module `C`.
 *
 * ```
 * module A.B.C { }
 * ```
 *
 * @param node  The module definition that should be tested.
 * @return TRUE if the given node is the topmost module declaration, FALSE otherwise.
 */
function isTopmostModuleDeclaration(node) {
    return node.getChildren().some(ts.isModuleBlock);
}
/**
 * Return the root module declaration of the given module declaration.
 *
 * In the following example this function would always return module
 * `A` no matter which of the modules was passed in.
 *
 * ```
 * module A.B.C { }
 * ```
 */
function getRootModuleDeclaration(node) {
    while (node.parent.kind === ts.SyntaxKind.ModuleDeclaration) {
        const parent = node.parent;
        if (node.name.pos === parent.name.end + 1) {
            node = parent;
        }
        else {
            break;
        }
    }
    return node;
}
function declarationToCommentNodeIgnoringParents(node) {
    // ts.SourceFile is a counterexample
    if (!node.parent)
        return node;
    // function foo(x: number)
    //              ^^^^^^^^^
    if (node.kind === ts.SyntaxKind.Parameter) {
        return node;
    }
    // const abc = 123
    //       ^^^
    if (node.parent.kind === ts.SyntaxKind.VariableDeclarationList) {
        return node.parent.parent;
    }
    // const a = () => {}
    //           ^^^^^^^^
    if (node.parent.kind === ts.SyntaxKind.VariableDeclaration) {
        return node.parent.parent.parent;
    }
    // class X { y = () => {} }
    //               ^^^^^^^^
    // function Z() {}
    // Z.method = () => {}
    //            ^^^^^^^^
    // export default () => {}
    //                ^^^^^^^^
    if ([
        ts.SyntaxKind.PropertyDeclaration,
        ts.SyntaxKind.BinaryExpression,
        ts.SyntaxKind.ExportAssignment,
    ].includes(node.parent.kind)) {
        return node.parent;
    }
    if (ts.isModuleDeclaration(node)) {
        if (!isTopmostModuleDeclaration(node)) {
            return;
        }
        else {
            return getRootModuleDeclaration(node);
        }
    }
    if (node.kind === ts.SyntaxKind.ExportSpecifier) {
        return node.parent.parent;
    }
    if (ts.SyntaxKind.NamespaceExport === node.kind) {
        return node.parent;
    }
}
function declarationToCommentNodes(node, checker) {
    const commentNode = declarationToCommentNodeIgnoringParents(node);
    if (commentNode) {
        return [
            {
                node: commentNode,
                inheritedFromParentDeclaration: false,
            },
        ];
    }
    const result = [
        {
            node,
            inheritedFromParentDeclaration: false,
        },
    ];
    let overloadIndex = undefined;
    if (ts.isMethodDeclaration(node)) {
        const symbol = checker.getSymbolAtLocation(node.name || node);
        if (symbol) {
            overloadIndex = symbol.declarations
                ?.filter((d) => d.kind === node.kind)
                .indexOf(node);
            ok(overloadIndex !== -1, "Should always find declaration");
        }
    }
    const seenSymbols = new Set();
    const bases = findBaseOfDeclaration(checker, node, (symbol) => {
        if (!seenSymbols.has(symbol)) {
            seenSymbols.add(symbol);
            if (overloadIndex === undefined) {
                return symbol.declarations?.map((node) => declarationToCommentNodeIgnoringParents(node) || node);
            }
            else if (symbol.declarations?.[overloadIndex]) {
                const parentSigNode = symbol.declarations[overloadIndex];
                return [
                    declarationToCommentNodeIgnoringParents(parentSigNode) ||
                        parentSigNode,
                ];
            }
        }
    });
    for (const parentCommentNode of bases || []) {
        result.push({
            node: parentCommentNode,
            inheritedFromParentDeclaration: true,
        });
    }
    // With overloaded functions/methods, TypeScript will use the comment on the first signature
    // declaration
    if ((ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node)) &&
        node.name) {
        const symbol = checker.getSymbolAtLocation(node.name);
        if (symbol && symbol.declarations[0] !== node) {
            result.push({
                node: symbol.declarations[0],
                inheritedFromParentDeclaration: true,
            });
        }
    }
    return result;
}
// Lifted from the TS source, with a couple minor modifications
function findBaseOfDeclaration(checker, declaration, cb) {
    const classOrInterfaceDeclaration = declaration.parent?.kind === ts.SyntaxKind.Constructor
        ? declaration.parent.parent
        : declaration.parent;
    if (!classOrInterfaceDeclaration)
        return;
    const isStaticMember = ts.getCombinedModifierFlags(declaration) & ts.ModifierFlags.Static;
    return firstDefined(ts.getAllSuperTypeNodes(classOrInterfaceDeclaration), (superTypeNode) => {
        const baseType = checker.getTypeAtLocation(superTypeNode);
        const type = isStaticMember && baseType.symbol
            ? checker.getTypeOfSymbol(baseType.symbol)
            : baseType;
        const symbol = checker.getPropertyOfType(type, declaration.symbol.name);
        return symbol ? cb(symbol) : undefined;
    });
}
/**
 * Separate comment ranges into arrays so that multiple line comments are kept together
 * and each block comment is left on its own.
 */
function collectCommentRanges(ranges) {
    const result = [];
    let collect = [];
    for (const range of ranges || []) {
        collect.push(range);
        switch (range.kind) {
            case ts.SyntaxKind.MultiLineCommentTrivia:
                if (collect.length) {
                    result.push(collect);
                    collect = [];
                }
                result.push([range]);
                break;
            case ts.SyntaxKind.SingleLineCommentTrivia:
                collect.push(range);
                break;
            /* istanbul ignore next */
            default:
                assertNever(range.kind);
        }
    }
    if (collect.length) {
        result.push(collect);
    }
    return result;
}
function permittedRange(text, ranges, commentStyle) {
    switch (commentStyle) {
        case CommentStyle.All:
            return true;
        case CommentStyle.Block:
            return ranges[0].kind === ts.SyntaxKind.MultiLineCommentTrivia;
        case CommentStyle.Line:
            return ranges[0].kind === ts.SyntaxKind.SingleLineCommentTrivia;
        case CommentStyle.JSDoc:
            return (ranges[0].kind === ts.SyntaxKind.MultiLineCommentTrivia &&
                text[ranges[0].pos] === "/" &&
                text[ranges[0].pos + 1] === "*" &&
                text[ranges[0].pos + 2] === "*");
    }
}
