import ts from "typescript";
import { Comment, ReflectionKind } from "../../models/index.js";
import { assertNever } from "../../utils/index.js";
import { lexBlockComment } from "./blockLexer.js";
import { discoverComment, discoverFileComments, discoverNodeComment, discoverSignatureComment, } from "./discovery.js";
import { lexLineComments } from "./lineLexer.js";
import { parseComment } from "./parser.js";
const jsDocCommentKinds = [
    ts.SyntaxKind.JSDocPropertyTag,
    ts.SyntaxKind.JSDocCallbackTag,
    ts.SyntaxKind.JSDocTypedefTag,
    ts.SyntaxKind.JSDocTemplateTag,
    ts.SyntaxKind.JSDocEnumTag,
];
let commentDiscoveryId = 0;
let commentCache = new WeakMap();
// We need to do this for tests so that changing the tsLinkResolution option
// actually works. Without it, we'd get the old parsed comment which doesn't
// have the TS symbols attached.
export function clearCommentCache() {
    commentCache = new WeakMap();
    commentDiscoveryId = 0;
}
function getCommentWithCache(discovered, config, logger, checker, files) {
    if (!discovered)
        return;
    const { file, ranges, jsDoc } = discovered;
    const cache = commentCache.get(file) || new Map();
    if (cache.has(ranges[0].pos)) {
        const clone = cache.get(ranges[0].pos).clone();
        clone.inheritedFromParentDeclaration =
            discovered.inheritedFromParentDeclaration;
        return clone;
    }
    let comment;
    switch (ranges[0].kind) {
        case ts.SyntaxKind.MultiLineCommentTrivia:
            comment = parseComment(lexBlockComment(file.text, ranges[0].pos, ranges[0].end, jsDoc, checker), config, file, logger, files);
            break;
        case ts.SyntaxKind.SingleLineCommentTrivia:
            comment = parseComment(lexLineComments(file.text, ranges), config, file, logger, files);
            break;
        default:
            assertNever(ranges[0].kind);
    }
    comment.discoveryId = ++commentDiscoveryId;
    comment.inheritedFromParentDeclaration =
        discovered.inheritedFromParentDeclaration;
    cache.set(ranges[0].pos, comment);
    commentCache.set(file, cache);
    return comment.clone();
}
function getCommentImpl(commentSource, config, logger, moduleComment, checker, files) {
    const comment = getCommentWithCache(commentSource, config, logger, config.useTsLinkResolution ? checker : undefined, files);
    if (comment?.getTag("@import") || comment?.getTag("@license")) {
        return;
    }
    if (moduleComment && comment) {
        // Module comment, make sure it is tagged with @packageDocumentation or @module.
        // If it isn't then the comment applies to the first statement in the file, so throw it away.
        if (!comment.hasModifier("@packageDocumentation") &&
            !comment.getTag("@module")) {
            return;
        }
    }
    if (!moduleComment && comment) {
        // Ensure module comments are not attached to non-module reflections.
        if (comment.hasModifier("@packageDocumentation") ||
            comment.getTag("@module")) {
            return;
        }
    }
    return comment;
}
export function getComment(symbol, kind, config, logger, checker, files) {
    const declarations = symbol.declarations || [];
    if (declarations.length &&
        declarations.every((d) => jsDocCommentKinds.includes(d.kind))) {
        return getJsDocComment(declarations[0], config, logger, checker, files);
    }
    const sf = declarations.find(ts.isSourceFile);
    if (sf) {
        return getFileComment(sf, config, logger, checker, files);
    }
    const isModule = declarations.some((decl) => {
        if (ts.isModuleDeclaration(decl) && ts.isStringLiteral(decl.name)) {
            return true;
        }
        return false;
    });
    const comment = getCommentImpl(discoverComment(symbol, kind, logger, config.commentStyle, checker, !config.suppressCommentWarningsInDeclarationFiles), config, logger, isModule, checker, files);
    if (!comment && kind === ReflectionKind.Property) {
        return getConstructorParamPropertyComment(symbol, config, logger, checker, files);
    }
    return comment;
}
export function getNodeComment(node, moduleComment, config, logger, checker, files) {
    return getCommentImpl(discoverNodeComment(node, config.commentStyle), config, logger, moduleComment, checker, files);
}
export function getFileComment(file, config, logger, checker, files) {
    for (const commentSource of discoverFileComments(file, config.commentStyle)) {
        const comment = getCommentWithCache(commentSource, config, logger, config.useTsLinkResolution ? checker : undefined, files);
        if (comment?.getTag("@license") || comment?.getTag("@import")) {
            continue;
        }
        if (comment?.getTag("@module") ||
            comment?.hasModifier("@packageDocumentation")) {
            return comment;
        }
        return;
    }
}
function getConstructorParamPropertyComment(symbol, config, logger, checker, files) {
    const decl = symbol.declarations?.find(ts.isParameter);
    if (!decl)
        return;
    const ctor = decl.parent;
    const comment = getSignatureComment(ctor, config, logger, checker, files);
    const paramTag = comment?.getIdentifiedTag(symbol.name, "@param");
    if (paramTag) {
        const result = new Comment(paramTag.content);
        result.sourcePath = comment.sourcePath;
        return result;
    }
}
export function getSignatureComment(declaration, config, logger, checker, files) {
    return getCommentImpl(discoverSignatureComment(declaration, checker, config.commentStyle), config, logger, false, checker, files);
}
export function getJsDocComment(declaration, config, logger, checker, files) {
    const file = declaration.getSourceFile();
    // First, get the whole comment. We know we'll need all of it.
    let parent = declaration.parent;
    while (!ts.isJSDoc(parent)) {
        parent = parent.parent;
    }
    // Then parse it.
    const comment = getCommentWithCache({
        file,
        ranges: [
            {
                kind: ts.SyntaxKind.MultiLineCommentTrivia,
                pos: parent.pos,
                end: parent.end,
            },
        ],
        jsDoc: parent,
        inheritedFromParentDeclaration: false,
    }, config, logger, config.useTsLinkResolution ? checker : undefined, files);
    // And pull out the tag we actually care about.
    if (ts.isJSDocEnumTag(declaration)) {
        const result = new Comment(comment.getTag("@enum")?.content);
        result.sourcePath = comment.sourcePath;
        return result;
    }
    if (ts.isJSDocTemplateTag(declaration) &&
        declaration.comment &&
        declaration.typeParameters.length > 1) {
        // We could just put the same comment on everything, but due to how comment parsing works,
        // we'd have to search for any @template with a name starting with the first type parameter's name
        // which feels horribly hacky.
        logger.warn(logger.i18n.multiple_type_parameters_on_template_tag_unsupported(), declaration);
        return;
    }
    let name;
    if (ts.isJSDocTemplateTag(declaration)) {
        // This isn't really ideal.
        name = declaration.typeParameters[0].name.text;
    }
    else {
        name = declaration.name?.getText();
    }
    if (!name) {
        return;
    }
    const tag = comment.getIdentifiedTag(name, `@${declaration.tagName.text}`);
    if (!tag) {
        logger.error(logger.i18n.failed_to_find_jsdoc_tag_for_name_0(name), declaration);
    }
    else {
        const result = new Comment(Comment.cloneDisplayParts(tag.content));
        result.sourcePath = comment.sourcePath;
        return result;
    }
}
