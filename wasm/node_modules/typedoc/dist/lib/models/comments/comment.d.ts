import type { Reflection } from "../reflections/index.js";
import { ReflectionSymbolId } from "../reflections/ReflectionSymbolId.js";
import type { Serializer, Deserializer, JSONOutput } from "../../serialization/index.js";
/**
 * Represents a parsed piece of a comment.
 * @category Comments
 * @see {@link JSONOutput.CommentDisplayPart}
 */
export type CommentDisplayPart = 
/**
 * Represents a plain text portion of the comment, may contain markdown
 */
{
    kind: "text";
    text: string;
}
/**
 * Represents a code block separated out form the plain text entry so
 * that TypeDoc knows to skip it when parsing relative links and inline tags.
 **/
 | {
    kind: "code";
    text: string;
} | InlineTagDisplayPart | RelativeLinkDisplayPart;
/**
 * Represents an inline tag like `{@link Foo}`
 *
 * The `@link`, `@linkcode`, and `@linkplain` tags may have a `target`
 * property set indicating which reflection/url they link to. They may also
 * have a `tsLinkText` property which includes the part of the `text` which
 * TypeScript thinks should be displayed as the link text.
 * @category Comments
 * @expand
 */
export interface InlineTagDisplayPart {
    kind: "inline-tag";
    tag: `@${string}`;
    text: string;
    target?: Reflection | string | ReflectionSymbolId;
    tsLinkText?: string;
}
/**
 * Represents a reference to a path relative to where the comment resides.
 * This is used to detect and copy relative image links.
 *
 * Use {@link FileRegistry} to determine what path on disc this refers to.
 *
 * This is used for relative links within comments/documents.
 * It is used to mark pieces of text which need to be replaced
 * to make links work properly.
 * @category Comments
 * @expand
 */
export interface RelativeLinkDisplayPart {
    kind: "relative-link";
    /**
     * The original relative text from the parsed comment.
     */
    text: string;
    /**
     * A link to either some document outside of the project or a reflection.
     * This may be `undefined` if the relative path does not exist.
     */
    target: number | undefined;
    /**
     * Anchor within the target page, validated after rendering if possible
     */
    targetAnchor: string | undefined;
}
/**
 * A model that represents a single TypeDoc comment tag.
 *
 * Tags are stored in the {@link Comment.blockTags} property.
 * @category Comments
 */
export declare class CommentTag {
    /**
     * The name of this tag, e.g. `@returns`, `@example`
     */
    tag: `@${string}`;
    /**
     * Some tags, (`@typedef`, `@param`, `@property`, etc.) may have a user defined identifier associated with them.
     * If this tag is one of those, it will be parsed out and included here.
     */
    name?: string;
    /**
     * The actual body text of this tag.
     */
    content: CommentDisplayPart[];
    /**
     * A flag which may be set by plugins to prevent TypeDoc from rendering this tag, if the plugin provides
     * custom rendering. Note: This flag is **not** serialized, it is expected to be set just before the comment
     * is rendered.
     */
    skipRendering: boolean;
    /**
     * Create a new CommentTag instance.
     */
    constructor(tag: `@${string}`, text: CommentDisplayPart[]);
    /**
     * Checks if this block tag is roughly equal to the other tag.
     * This isn't exactly equal, but just "roughly equal" by the tag
     * text.
     */
    similarTo(other: CommentTag): boolean;
    clone(): CommentTag;
    toObject(serializer: Serializer): JSONOutput.CommentTag;
    fromObject(de: Deserializer, obj: JSONOutput.CommentTag): void;
}
/**
 * A model that represents a comment.
 *
 * Instances of this model are created by the CommentPlugin. You can retrieve comments
 * through the {@link DeclarationReflection.comment} property.
 * @category Comments
 */
export declare class Comment {
    /**
     * Debugging utility for combining parts into a simple string. Not suitable for
     * rendering, but can be useful in tests.
     */
    static combineDisplayParts(parts: readonly CommentDisplayPart[] | undefined): string;
    /**
     * Helper utility to clone {@link Comment.summary} or {@link CommentTag.content}
     */
    static cloneDisplayParts(parts: readonly CommentDisplayPart[]): CommentDisplayPart[];
    static serializeDisplayParts(serializer: Serializer, parts: CommentDisplayPart[]): JSONOutput.CommentDisplayPart[];
    /** @hidden no point in showing this signature in api docs */
    static serializeDisplayParts(serializer: Serializer, parts: CommentDisplayPart[] | undefined): JSONOutput.CommentDisplayPart[] | undefined;
    static deserializeDisplayParts(de: Deserializer, parts: JSONOutput.CommentDisplayPart[]): CommentDisplayPart[];
    /**
     * Splits the provided parts into a header (first line, as a string)
     * and body (remaining lines). If the header line contains inline tags
     * they will be serialized to a string.
     */
    static splitPartsToHeaderAndBody(parts: readonly CommentDisplayPart[]): {
        header: string;
        body: CommentDisplayPart[];
    };
    /**
     * The content of the comment which is not associated with a block tag.
     */
    summary: CommentDisplayPart[];
    /**
     * All associated block level tags.
     */
    blockTags: CommentTag[];
    /**
     * All modifier tags present on the comment, e.g. `@alpha`, `@beta`.
     */
    modifierTags: Set<`@${string}`>;
    /**
     * Label associated with this reflection, if any (https://tsdoc.org/pages/tags/label/)
     */
    label?: string;
    /**
     * Full path to the file where this comment originated from, if any.
     * This field will not be serialized, so will not be present when handling JSON-revived reflections.
     *
     * Note: This field is non-enumerable to make testing comment contents with `deepEqual` easier.
     */
    sourcePath?: string;
    /**
     * Internal discovery ID used to prevent symbol comments from
     * being duplicated on signatures. Only set when the comment was created
     * from a `ts.CommentRange`.
     * @internal
     */
    discoveryId?: number;
    /**
     * If the comment was inherited from a different "parent" declaration
     * (see #2545), then it is desirable to know this as any `@param` tags
     * which do not apply should not cause warnings. This is not serialized,
     * and only set when the comment was created from a `ts.CommentRange`.
     */
    inheritedFromParentDeclaration?: boolean;
    /**
     * Creates a new Comment instance.
     */
    constructor(summary?: CommentDisplayPart[], blockTags?: CommentTag[], modifierTags?: Set<`@${string}`>);
    /**
     * Gets either the `@summary` tag, or a short version of the comment summary
     * section for rendering in module/namespace pages.
     */
    getShortSummary(useFirstParagraph: boolean): readonly CommentDisplayPart[];
    /**
     * Checks if this comment is roughly equal to the other comment.
     * This isn't exactly equal, but just "roughly equal" by the comment
     * text.
     */
    similarTo(other: Comment): boolean;
    /**
     * Create a deep clone of this comment.
     */
    clone(): Comment;
    /**
     * Returns true if this comment is completely empty.
     * @internal
     */
    isEmpty(): boolean;
    /**
     * Has this comment a visible component?
     *
     * @returns TRUE when this comment has a visible component.
     */
    hasVisibleComponent(): boolean;
    /**
     * Test whether this comment contains a tag with the given name.
     *
     * @param tagName  The name of the tag to look for.
     * @returns TRUE when this comment contains a tag with the given name, otherwise FALSE.
     */
    hasModifier(tagName: `@${string}`): boolean;
    removeModifier(tagName: `@${string}`): void;
    /**
     * Return the first tag with the given name.
     *
     * @param tagName  The name of the tag to look for.
     * @returns The found tag or undefined.
     */
    getTag(tagName: `@${string}`): CommentTag | undefined;
    /**
     * Get all tags with the given tag name.
     */
    getTags(tagName: `@${string}`): CommentTag[];
    getIdentifiedTag(identifier: string, tagName: `@${string}`): CommentTag | undefined;
    /**
     * Removes all block tags with the given tag name from the comment.
     * @param tagName
     */
    removeTags(tagName: `@${string}`): void;
    toObject(serializer: Serializer): JSONOutput.Comment;
    fromObject(de: Deserializer, obj: JSONOutput.Comment): void;
}
