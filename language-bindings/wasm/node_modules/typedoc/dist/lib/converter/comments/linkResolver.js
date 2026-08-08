import ts from "typescript";
import { DeclarationReflection, Reflection, ReflectionKind, ReflectionSymbolId, } from "../../models/index.js";
import { parseDeclarationReference, } from "./declarationReference.js";
import { resolveDeclarationReference } from "./declarationReferenceResolver.js";
import { maxElementByScore } from "../../utils/array.js";
const urlPrefix = /^(http|ftp)s?:\/\//;
export function resolveLinks(comment, reflection, externalResolver, options) {
    comment.summary = resolvePartLinks(reflection, comment.summary, externalResolver, options);
    for (const tag of comment.blockTags) {
        tag.content = resolvePartLinks(reflection, tag.content, externalResolver, options);
    }
    if (reflection instanceof DeclarationReflection && reflection.readme) {
        reflection.readme = resolvePartLinks(reflection, reflection.readme, externalResolver, options);
    }
    if (reflection.isDocument()) {
        reflection.content = resolvePartLinks(reflection, reflection.content, externalResolver, options);
    }
}
export function resolvePartLinks(reflection, parts, externalResolver, options) {
    return parts.flatMap((part) => processPart(reflection, part, externalResolver, options));
}
function processPart(reflection, part, externalResolver, options) {
    if (part.kind === "inline-tag") {
        if (part.tag === "@link" ||
            part.tag === "@linkcode" ||
            part.tag === "@linkplain") {
            return resolveLinkTag(reflection, part, externalResolver, options);
        }
    }
    return part;
}
function resolveLinkTag(reflection, part, externalResolver, options) {
    // This tag may have already been resolved to if we are running in packages mode
    // or when reading in a JSON file. #2680.
    if (typeof part.target === "string" || part.target instanceof Reflection) {
        return part;
    }
    let defaultDisplayText = "";
    let pos = 0;
    const end = part.text.length;
    while (pos < end && ts.isWhiteSpaceLike(part.text.charCodeAt(pos))) {
        pos++;
    }
    let target;
    // Try to parse a declaration reference if we didn't use the TS symbol for resolution
    const declRef = parseDeclarationReference(part.text, pos, end);
    // Might already know where it should go if useTsLinkResolution is turned on
    if (part.target instanceof ReflectionSymbolId) {
        const tsTargets = reflection.project.getReflectionsFromSymbolId(part.target);
        if (tsTargets.length) {
            // Find the target most likely to have a real url in the generated documentation
            // 1. A direct export (class, interface, variable)
            // 2. A property of a direct export (class/interface property)
            // 3. A property of a type of an export (property on type alias)
            // 4. Whatever the first symbol found was
            target = maxElementByScore(tsTargets, (r) => {
                if (r.kindOf(ReflectionKind.SomeExport)) {
                    return 4;
                }
                if (r.kindOf(ReflectionKind.SomeMember) &&
                    r.parent?.kindOf(ReflectionKind.SomeExport)) {
                    return 3;
                }
                if (r.kindOf(ReflectionKind.SomeMember) &&
                    r.parent?.kindOf(ReflectionKind.TypeLiteral) &&
                    r.parent.parent?.kindOf(ReflectionKind.TypeAlias | ReflectionKind.Variable)) {
                    return 2;
                }
                return 1;
            });
            pos = end;
            defaultDisplayText =
                part.tsLinkText ||
                    (options.preserveLinkText ? part.text : target.name);
        }
        else {
            // If we didn't find a target, we might be pointing to a symbol in another project that will be merged in
            // or some external symbol, so ask external resolvers to try resolution. Don't use regular declaration ref
            // resolution in case it matches something that would have been merged in later.
            if (declRef) {
                pos = declRef[1];
            }
            const externalResolveResult = externalResolver(declRef?.[0] ?? part.target.toDeclarationReference(), reflection, part, part.target);
            defaultDisplayText =
                part.tsLinkText ||
                    (options.preserveLinkText
                        ? part.text
                        : part.text.substring(0, pos));
            switch (typeof externalResolveResult) {
                case "string":
                    target = externalResolveResult;
                    break;
                case "object":
                    target = externalResolveResult.target;
                    defaultDisplayText =
                        externalResolveResult.caption || defaultDisplayText;
            }
        }
    }
    if (!target && declRef) {
        // Got one, great! Try to resolve the link
        target = resolveDeclarationReference(reflection, declRef[0]);
        pos = declRef[1];
        if (target) {
            defaultDisplayText = options.preserveLinkText
                ? part.text
                : target.name;
        }
        else {
            // If we didn't find a link, it might be a @link tag to an external symbol, check that next.
            const externalResolveResult = externalResolver(declRef[0], reflection, part, part.target instanceof ReflectionSymbolId
                ? part.target
                : undefined);
            defaultDisplayText = options.preserveLinkText
                ? part.text
                : part.text.substring(0, pos);
            switch (typeof externalResolveResult) {
                case "string":
                    target = externalResolveResult;
                    break;
                case "object":
                    target = externalResolveResult.target;
                    defaultDisplayText =
                        externalResolveResult.caption || defaultDisplayText;
            }
        }
    }
    if (!target && urlPrefix.test(part.text)) {
        const wsIndex = part.text.search(/\s/);
        target = wsIndex === -1 ? part.text : part.text.substring(0, wsIndex);
        pos = target.length;
        defaultDisplayText = target;
    }
    // Remaining text after an optional pipe is the link text, so advance
    // until that's consumed.
    while (pos < end && ts.isWhiteSpaceLike(part.text.charCodeAt(pos))) {
        pos++;
    }
    if (pos < end && part.text[pos] === "|") {
        pos++;
    }
    if (!target) {
        return part;
    }
    part.target = target;
    part.text =
        part.text.substring(pos).trim() || defaultDisplayText || part.text;
    return part;
}
