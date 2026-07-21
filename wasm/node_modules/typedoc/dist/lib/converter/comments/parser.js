import assert, { ok } from "assert";
import { parseDocument as parseYamlDoc } from "yaml";
import { Comment, CommentTag, } from "../../models/index.js";
import { assertNever, removeIf } from "../../utils/index.js";
import { nicePath } from "../../utils/paths.js";
import { TokenSyntaxKind } from "./lexer.js";
import { extractTagName } from "./tagName.js";
import { FileRegistry } from "../../models/FileRegistry.js";
import { textContent, TextParserReentryState } from "./textParser.js";
import { hasDeclarationFileExtension } from "../../utils/fs.js";
function makeLookaheadGenerator(gen) {
    let trackHistory = false;
    const history = [];
    const next = [gen.next()];
    return {
        done() {
            return !!next[0].done;
        },
        peek() {
            ok(!next[0].done);
            return next[0].value;
        },
        take() {
            const thisItem = next.shift();
            if (trackHistory) {
                history.push(thisItem);
            }
            ok(!thisItem.done);
            next.push(gen.next());
            return thisItem.value;
        },
        mark() {
            ok(!trackHistory, "Can only mark one location for backtracking at a time");
            trackHistory = true;
        },
        release() {
            trackHistory = false;
            next.unshift(...history);
            history.length = 0;
        },
    };
}
export function parseComment(tokens, config, file, logger, files) {
    const lexer = makeLookaheadGenerator(tokens);
    const tok = lexer.done() || lexer.peek();
    const comment = new Comment();
    comment.sourcePath = file.fileName;
    comment.summary = blockContent(comment, lexer, config, logger.i18n, warningImpl, files);
    while (!lexer.done()) {
        comment.blockTags.push(blockTag(comment, lexer, config, logger.i18n, warningImpl, files));
    }
    const tok2 = tok;
    postProcessComment(comment, logger.i18n, () => `${nicePath(file.fileName)}:${file.getLineAndCharacterOfPosition(tok2.pos).line + 1}`, (message) => logger.warn(message));
    return comment;
    function warningImpl(message, token) {
        if (config.suppressCommentWarningsInDeclarationFiles &&
            hasDeclarationFileExtension(file.fileName)) {
            return;
        }
        logger.warn(message, token.pos, file);
    }
}
/**
 * Intended for parsing markdown documents. This only parses code blocks and
 * inline tags outside of code blocks, everything else is text.
 *
 * If you change this, also look at blockContent, as it likely needs similar
 * modifications to ensure parsing is consistent.
 */
export function parseCommentString(tokens, config, file, logger, files) {
    const suppressWarningsConfig = {
        ...config,
        jsDocCompatibility: {
            defaultTag: true,
            exampleTag: true,
            ignoreUnescapedBraces: true,
            inheritDocTag: true,
        },
        suppressCommentWarningsInDeclarationFiles: true,
    };
    const reentry = new TextParserReentryState();
    const content = [];
    const lexer = makeLookaheadGenerator(tokens);
    let atNewLine = false;
    while (!lexer.done()) {
        let consume = true;
        const next = lexer.peek();
        reentry.checkState(next);
        switch (next.kind) {
            case TokenSyntaxKind.TypeAnnotation:
                // Shouldn't have been produced by our lexer
                assert(false, "Should be unreachable");
                break;
            case TokenSyntaxKind.NewLine:
            case TokenSyntaxKind.Text:
            case TokenSyntaxKind.Tag:
            case TokenSyntaxKind.CloseBrace:
                textContent(file.fileName, next, logger.i18n, (msg, token) => logger.warn(msg, token.pos, file), content, files, atNewLine, reentry);
                break;
            case TokenSyntaxKind.Code:
                content.push({ kind: "code", text: next.text });
                break;
            case TokenSyntaxKind.OpenBrace:
                inlineTag(lexer, content, suppressWarningsConfig, logger.i18n, (message, token) => logger.warn(message, token.pos, file));
                consume = false;
                break;
            default:
                assertNever(next.kind);
        }
        atNewLine = next.kind === TokenSyntaxKind.NewLine;
        if (consume) {
            lexer.take();
        }
    }
    // Check for frontmatter
    let frontmatterData = {};
    const firstBlock = content.at(0);
    if (firstBlock?.text.startsWith("---\n")) {
        const end = firstBlock.text.indexOf("\n---\n");
        if (end !== -1) {
            const yamlText = firstBlock.text.slice("---\n".length, end);
            firstBlock.text = firstBlock.text
                .slice(end + "\n---\n".length)
                .trimStart();
            const frontmatter = parseYamlDoc(yamlText, { prettyErrors: false });
            for (const warning of frontmatter.warnings) {
                // Can't translate issues coming from external library...
                logger.warn(warning.message, warning.pos[0] + "---\n".length, file);
            }
            for (const error of frontmatter.errors) {
                // Can't translate issues coming from external library...
                logger.error(error.message, error.pos[0] + "---\n".length, file);
            }
            if (frontmatter.errors.length === 0) {
                const data = frontmatter.toJS();
                if (typeof data === "object") {
                    frontmatterData = data;
                }
                else {
                    logger.error(logger.i18n.yaml_frontmatter_not_an_object(), 5, file);
                }
            }
        }
    }
    return { content, frontmatter: frontmatterData };
}
const HAS_USER_IDENTIFIER = [
    "@callback",
    "@param",
    "@prop",
    "@property",
    "@template",
    "@typedef",
    "@typeParam",
    "@inheritDoc",
];
function makeCodeBlock(text) {
    return "```ts\n" + text + "\n```";
}
/**
 * Loop over comment, produce lint warnings, and set tag names for tags
 * which have them.
 */
function postProcessComment(comment, i18n, getPosition, warning) {
    for (const tag of comment.blockTags) {
        if (HAS_USER_IDENTIFIER.includes(tag.tag) && tag.content.length) {
            const first = tag.content[0];
            if (first.kind === "text") {
                const { name, newText } = extractTagName(first.text);
                tag.name = name;
                if (newText) {
                    first.text = newText;
                }
                else {
                    // Remove this token, no real text in it.
                    tag.content.shift();
                }
            }
        }
        if (tag.content.some((part) => part.kind === "inline-tag" && part.tag === "@inheritDoc")) {
            warning(i18n.inline_inheritdoc_should_not_appear_in_block_tag_in_comment_at_0(getPosition()));
        }
    }
    const remarks = comment.blockTags.filter((tag) => tag.tag === "@remarks");
    if (remarks.length > 1) {
        warning(i18n.at_most_one_remarks_tag_expected_in_comment_at_0(getPosition()));
        removeIf(comment.blockTags, (tag) => remarks.indexOf(tag) > 0);
    }
    const returns = comment.blockTags.filter((tag) => tag.tag === "@returns");
    if (remarks.length > 1) {
        warning(i18n.at_most_one_returns_tag_expected_in_comment_at_0(getPosition()));
        removeIf(comment.blockTags, (tag) => returns.indexOf(tag) > 0);
    }
    const inheritDoc = comment.blockTags.filter((tag) => tag.tag === "@inheritDoc");
    const inlineInheritDoc = comment.summary.filter((part) => part.kind === "inline-tag" && part.tag === "@inheritDoc");
    if (inlineInheritDoc.length + inheritDoc.length > 1) {
        warning(i18n.at_most_one_inheritdoc_tag_expected_in_comment_at_0(getPosition()));
        const allInheritTags = [...inlineInheritDoc, ...inheritDoc];
        removeIf(comment.summary, (part) => allInheritTags.indexOf(part) > 0);
        removeIf(comment.blockTags, (tag) => allInheritTags.indexOf(tag) > 0);
    }
    if ((inlineInheritDoc.length || inheritDoc.length) &&
        comment.summary.some((part) => part.kind !== "inline-tag" && /\S/.test(part.text))) {
        warning(i18n.content_in_summary_overwritten_by_inheritdoc_in_comment_at_0(getPosition()));
    }
    if ((inlineInheritDoc.length || inheritDoc.length) && remarks.length) {
        warning(i18n.content_in_remarks_block_overwritten_by_inheritdoc_in_comment_at_0(getPosition()));
    }
}
const aliasedTags = new Map([["@return", "@returns"]]);
function blockTag(comment, lexer, config, i18n, warning, files) {
    const blockTag = lexer.take();
    ok(blockTag.kind === TokenSyntaxKind.Tag, "blockTag called not at the start of a block tag."); // blockContent is broken if this fails.
    if (!config.blockTags.has(blockTag.text)) {
        warning(i18n.unknown_block_tag_0(blockTag.text), blockTag);
    }
    const tagName = aliasedTags.get(blockTag.text) || blockTag.text;
    let content;
    if (tagName === "@example") {
        return exampleBlock(comment, lexer, config, i18n, warning, files);
    }
    else if (["@default", "@defaultValue"].includes(tagName) &&
        config.jsDocCompatibility.defaultTag) {
        content = defaultBlockContent(comment, lexer, config, i18n, warning, files);
    }
    else {
        content = blockContent(comment, lexer, config, i18n, warning, files);
    }
    return new CommentTag(tagName, content);
}
/**
 * The `@default` tag gets a special case because otherwise we will produce many warnings
 * about unescaped/mismatched/missing braces in legacy JSDoc comments
 */
function defaultBlockContent(comment, lexer, config, i18n, warning, files) {
    lexer.mark();
    const tempRegistry = new FileRegistry();
    const content = blockContent(comment, lexer, config, i18n, () => { }, tempRegistry);
    const end = lexer.done() || lexer.peek();
    lexer.release();
    if (content.some((part) => part.kind === "code" || part.kind === "inline-tag")) {
        return blockContent(comment, lexer, config, i18n, warning, files);
    }
    const tokens = [];
    while ((lexer.done() || lexer.peek()) !== end) {
        tokens.push(lexer.take());
    }
    const blockText = tokens
        .map((tok) => tok.text)
        .join("")
        .trim();
    return [
        {
            kind: "code",
            text: makeCodeBlock(blockText),
        },
    ];
}
/**
 * The `@example` tag gets a special case because otherwise we will produce many warnings
 * about unescaped/mismatched/missing braces in legacy JSDoc comments.
 *
 * In TSDoc, we also want to treat the first line of the block as the example name.
 */
function exampleBlock(comment, lexer, config, i18n, warning, files) {
    lexer.mark();
    const tempRegistry = new FileRegistry();
    const content = blockContent(comment, lexer, config, i18n, () => { }, tempRegistry);
    const end = lexer.done() || lexer.peek();
    lexer.release();
    if (!config.jsDocCompatibility.exampleTag ||
        content.some((part) => part.kind === "code" && part.text.startsWith("```"))) {
        let exampleName = "";
        // First line of @example block is the example name.
        let warnedAboutRichNameContent = false;
        outer: while ((lexer.done() || lexer.peek()) !== end) {
            const next = lexer.peek();
            switch (next.kind) {
                case TokenSyntaxKind.NewLine:
                    lexer.take();
                    break outer;
                case TokenSyntaxKind.Text: {
                    const newline = next.text.indexOf("\n");
                    if (newline !== -1) {
                        exampleName += next.text.substring(0, newline);
                        next.pos += newline + 1;
                        break outer;
                    }
                    else {
                        exampleName += lexer.take().text;
                    }
                    break;
                }
                case TokenSyntaxKind.Code:
                case TokenSyntaxKind.Tag:
                case TokenSyntaxKind.TypeAnnotation:
                case TokenSyntaxKind.CloseBrace:
                case TokenSyntaxKind.OpenBrace:
                    if (!warnedAboutRichNameContent) {
                        warning(i18n.example_tag_literal_name(), lexer.peek());
                        warnedAboutRichNameContent = true;
                    }
                    exampleName += lexer.take().text;
                    break;
                default:
                    assertNever(next.kind);
            }
        }
        const content = blockContent(comment, lexer, config, i18n, warning, files);
        const tag = new CommentTag("@example", content);
        if (exampleName.trim()) {
            tag.name = exampleName.trim();
        }
        return tag;
    }
    const tokens = [];
    while ((lexer.done() || lexer.peek()) !== end) {
        tokens.push(lexer.take());
    }
    const blockText = tokens
        .map((tok) => tok.text)
        .join("")
        .trim();
    const caption = blockText.match(/^\s*<caption>(.*?)<\/caption>\s*(\n|$)/);
    if (caption) {
        const tag = new CommentTag("@example", [
            {
                kind: "code",
                text: makeCodeBlock(blockText.slice(caption[0].length)),
            },
        ]);
        tag.name = caption[1];
        return tag;
    }
    else {
        return new CommentTag("@example", [
            {
                kind: "code",
                text: makeCodeBlock(blockText),
            },
        ]);
    }
}
/**
 * If you change this, also look at parseCommentString as it
 * likely needs similar modifications to ensure parsing is consistent.
 */
function blockContent(comment, lexer, config, i18n, warning, files) {
    const content = [];
    let atNewLine = true;
    const reentry = new TextParserReentryState();
    loop: while (!lexer.done()) {
        const next = lexer.peek();
        reentry.checkState(next);
        let consume = true;
        switch (next.kind) {
            case TokenSyntaxKind.NewLine:
                content.push({ kind: "text", text: next.text });
                break;
            case TokenSyntaxKind.Text:
                textContent(comment.sourcePath, next, i18n, warning, 
                /*out*/ content, files, atNewLine, reentry);
                break;
            case TokenSyntaxKind.Code:
                content.push({ kind: "code", text: next.text });
                break;
            case TokenSyntaxKind.Tag:
                if (next.text === "@inheritdoc") {
                    if (!config.jsDocCompatibility.inheritDocTag) {
                        warning(i18n.inheritdoc_tag_properly_capitalized(), next);
                    }
                    next.text = "@inheritDoc";
                }
                if (config.modifierTags.has(next.text)) {
                    comment.modifierTags.add(next.text);
                    break;
                }
                else if (!atNewLine && !config.blockTags.has(next.text)) {
                    // Treat unknown tag as a modifier, but warn about it.
                    comment.modifierTags.add(next.text);
                    warning(i18n.treating_unrecognized_tag_0_as_modifier(next.text), next);
                    break;
                }
                else {
                    // Block tag or unknown tag, handled by our caller.
                    break loop;
                }
            case TokenSyntaxKind.TypeAnnotation:
                // We always ignore these. In TS files they are redundant, in JS files
                // they are required.
                break;
            case TokenSyntaxKind.CloseBrace:
                // Unmatched closing brace, generate a warning, and treat it as text.
                if (!config.jsDocCompatibility.ignoreUnescapedBraces) {
                    warning(i18n.unmatched_closing_brace(), next);
                }
                content.push({ kind: "text", text: next.text });
                break;
            case TokenSyntaxKind.OpenBrace:
                inlineTag(lexer, content, config, i18n, warning);
                consume = false;
                break;
            default:
                assertNever(next.kind);
        }
        if (consume && lexer.take().kind === TokenSyntaxKind.NewLine) {
            atNewLine = true;
        }
    }
    // Collapse adjacent text parts
    for (let i = 0; i < content.length - 1 /* inside loop */;) {
        if (content[i].kind === "text" && content[i + 1].kind === "text") {
            content[i].text += content[i + 1].text;
            content.splice(i + 1, 1);
        }
        else {
            i++;
        }
    }
    // Now get rid of extra whitespace, and any empty parts
    for (let i = 0; i < content.length /* inside loop */;) {
        if (i === 0 || content[i].kind === "inline-tag") {
            content[i].text = content[i].text.trimStart();
        }
        if (i === content.length - 1 || content[i].kind === "inline-tag") {
            content[i].text = content[i].text.trimEnd();
        }
        if (!content[i].text && content[i].kind === "text") {
            content.splice(i, 1);
        }
        else {
            i++;
        }
    }
    return content;
}
function inlineTag(lexer, block, config, i18n, warning) {
    const openBrace = lexer.take();
    // Now skip whitespace to grab the tag name.
    // If the first non-whitespace text after the brace isn't a tag,
    // then produce a warning and treat what we've consumed as plain text.
    if (lexer.done() ||
        ![TokenSyntaxKind.Text, TokenSyntaxKind.Tag].includes(lexer.peek().kind)) {
        if (!config.jsDocCompatibility.ignoreUnescapedBraces) {
            warning(i18n.unescaped_open_brace_without_inline_tag(), openBrace);
        }
        block.push({ kind: "text", text: openBrace.text });
        return;
    }
    let tagName = lexer.take();
    if (lexer.done() ||
        (tagName.kind === TokenSyntaxKind.Text &&
            (!/^\s+$/.test(tagName.text) ||
                lexer.peek().kind != TokenSyntaxKind.Tag))) {
        if (!config.jsDocCompatibility.ignoreUnescapedBraces) {
            warning(i18n.unescaped_open_brace_without_inline_tag(), openBrace);
        }
        block.push({ kind: "text", text: openBrace.text + tagName.text });
        return;
    }
    if (tagName.kind !== TokenSyntaxKind.Tag) {
        tagName = lexer.take();
    }
    if (!config.inlineTags.has(tagName.text)) {
        warning(i18n.unknown_inline_tag_0(tagName.text), tagName);
    }
    const content = [];
    // At this point, we know we have an inline tag. Treat everything following as plain text,
    // until we get to the closing brace.
    while (!lexer.done() && lexer.peek().kind !== TokenSyntaxKind.CloseBrace) {
        const token = lexer.take();
        if (token.kind === TokenSyntaxKind.OpenBrace) {
            warning(i18n.open_brace_within_inline_tag(), token);
        }
        content.push(token.kind === TokenSyntaxKind.NewLine ? " " : token.text);
    }
    if (lexer.done()) {
        warning(i18n.inline_tag_not_closed(), openBrace);
    }
    else {
        lexer.take(); // Close brace
    }
    const inlineTag = {
        kind: "inline-tag",
        tag: tagName.text,
        text: content.join(""),
    };
    if (tagName.tsLinkTarget) {
        inlineTag.target = tagName.tsLinkTarget;
    }
    // Separated from tsLinkTarget to avoid storing a useless empty string
    // if TS doesn't have an opinion on what the link text should be.
    if (tagName.tsLinkText) {
        inlineTag.tsLinkText = tagName.tsLinkText;
    }
    block.push(inlineTag);
}
