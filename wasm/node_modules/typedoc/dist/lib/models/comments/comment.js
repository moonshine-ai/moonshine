var __esDecorate = (this && this.__esDecorate) || function (ctor, descriptorIn, decorators, contextIn, initializers, extraInitializers) {
    function accept(f) { if (f !== void 0 && typeof f !== "function") throw new TypeError("Function expected"); return f; }
    var kind = contextIn.kind, key = kind === "getter" ? "get" : kind === "setter" ? "set" : "value";
    var target = !descriptorIn && ctor ? contextIn["static"] ? ctor : ctor.prototype : null;
    var descriptor = descriptorIn || (target ? Object.getOwnPropertyDescriptor(target, contextIn.name) : {});
    var _, done = false;
    for (var i = decorators.length - 1; i >= 0; i--) {
        var context = {};
        for (var p in contextIn) context[p] = p === "access" ? {} : contextIn[p];
        for (var p in contextIn.access) context.access[p] = contextIn.access[p];
        context.addInitializer = function (f) { if (done) throw new TypeError("Cannot add initializers after decoration has completed"); extraInitializers.push(accept(f || null)); };
        var result = (0, decorators[i])(kind === "accessor" ? { get: descriptor.get, set: descriptor.set } : descriptor[key], context);
        if (kind === "accessor") {
            if (result === void 0) continue;
            if (result === null || typeof result !== "object") throw new TypeError("Object expected");
            if (_ = accept(result.get)) descriptor.get = _;
            if (_ = accept(result.set)) descriptor.set = _;
            if (_ = accept(result.init)) initializers.unshift(_);
        }
        else if (_ = accept(result)) {
            if (kind === "field") initializers.unshift(_);
            else descriptor[key] = _;
        }
    }
    if (target) Object.defineProperty(target, contextIn.name, descriptor);
    done = true;
};
var __runInitializers = (this && this.__runInitializers) || function (thisArg, initializers, value) {
    var useValue = arguments.length > 2;
    for (var i = 0; i < initializers.length; i++) {
        value = useValue ? initializers[i].call(thisArg, value) : initializers[i].call(thisArg);
    }
    return useValue ? value : void 0;
};
import { assertNever, removeIf } from "../../utils/index.js";
import { ReflectionSymbolId } from "../reflections/ReflectionSymbolId.js";
import { NonEnumerable } from "../../utils/general.js";
/**
 * A model that represents a single TypeDoc comment tag.
 *
 * Tags are stored in the {@link Comment.blockTags} property.
 * @category Comments
 */
export class CommentTag {
    /**
     * The name of this tag, e.g. `@returns`, `@example`
     */
    tag;
    /**
     * Some tags, (`@typedef`, `@param`, `@property`, etc.) may have a user defined identifier associated with them.
     * If this tag is one of those, it will be parsed out and included here.
     */
    name;
    /**
     * The actual body text of this tag.
     */
    content;
    /**
     * A flag which may be set by plugins to prevent TypeDoc from rendering this tag, if the plugin provides
     * custom rendering. Note: This flag is **not** serialized, it is expected to be set just before the comment
     * is rendered.
     */
    skipRendering = false;
    /**
     * Create a new CommentTag instance.
     */
    constructor(tag, text) {
        this.tag = tag;
        this.content = text;
    }
    /**
     * Checks if this block tag is roughly equal to the other tag.
     * This isn't exactly equal, but just "roughly equal" by the tag
     * text.
     */
    similarTo(other) {
        return (this.tag === other.tag &&
            this.name === other.tag &&
            Comment.combineDisplayParts(this.content) ===
                Comment.combineDisplayParts(other.content));
    }
    clone() {
        const tag = new CommentTag(this.tag, Comment.cloneDisplayParts(this.content));
        if (this.name) {
            tag.name = this.name;
        }
        return tag;
    }
    toObject(serializer) {
        return {
            tag: this.tag,
            name: this.name,
            content: Comment.serializeDisplayParts(serializer, this.content),
        };
    }
    fromObject(de, obj) {
        // tag already set by Comment.fromObject
        this.name = obj.name;
        this.content = Comment.deserializeDisplayParts(de, obj.content);
    }
}
/**
 * A model that represents a comment.
 *
 * Instances of this model are created by the CommentPlugin. You can retrieve comments
 * through the {@link DeclarationReflection.comment} property.
 * @category Comments
 */
let Comment = (() => {
    let _sourcePath_decorators;
    let _sourcePath_initializers = [];
    let _sourcePath_extraInitializers = [];
    let _discoveryId_decorators;
    let _discoveryId_initializers = [];
    let _discoveryId_extraInitializers = [];
    let _inheritedFromParentDeclaration_decorators;
    let _inheritedFromParentDeclaration_initializers = [];
    let _inheritedFromParentDeclaration_extraInitializers = [];
    return class Comment {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(null) : void 0;
            _sourcePath_decorators = [NonEnumerable];
            _discoveryId_decorators = [NonEnumerable];
            _inheritedFromParentDeclaration_decorators = [NonEnumerable];
            __esDecorate(null, null, _sourcePath_decorators, { kind: "field", name: "sourcePath", static: false, private: false, access: { has: obj => "sourcePath" in obj, get: obj => obj.sourcePath, set: (obj, value) => { obj.sourcePath = value; } }, metadata: _metadata }, _sourcePath_initializers, _sourcePath_extraInitializers);
            __esDecorate(null, null, _discoveryId_decorators, { kind: "field", name: "discoveryId", static: false, private: false, access: { has: obj => "discoveryId" in obj, get: obj => obj.discoveryId, set: (obj, value) => { obj.discoveryId = value; } }, metadata: _metadata }, _discoveryId_initializers, _discoveryId_extraInitializers);
            __esDecorate(null, null, _inheritedFromParentDeclaration_decorators, { kind: "field", name: "inheritedFromParentDeclaration", static: false, private: false, access: { has: obj => "inheritedFromParentDeclaration" in obj, get: obj => obj.inheritedFromParentDeclaration, set: (obj, value) => { obj.inheritedFromParentDeclaration = value; } }, metadata: _metadata }, _inheritedFromParentDeclaration_initializers, _inheritedFromParentDeclaration_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        /**
         * Debugging utility for combining parts into a simple string. Not suitable for
         * rendering, but can be useful in tests.
         */
        static combineDisplayParts(parts) {
            let result = "";
            for (const item of parts || []) {
                switch (item.kind) {
                    case "text":
                    case "code":
                    case "relative-link":
                        result += item.text;
                        break;
                    case "inline-tag":
                        result += `{${item.tag} ${item.text}}`;
                        break;
                    default:
                        assertNever(item);
                }
            }
            return result;
        }
        /**
         * Helper utility to clone {@link Comment.summary} or {@link CommentTag.content}
         */
        static cloneDisplayParts(parts) {
            return parts.map((p) => ({ ...p }));
        }
        static serializeDisplayParts(serializer, parts) {
            return parts?.map((part) => {
                switch (part.kind) {
                    case "text":
                    case "code":
                        return { ...part };
                    case "inline-tag": {
                        let target;
                        if (typeof part.target === "string") {
                            target = part.target;
                        }
                        else if (part.target) {
                            if ("id" in part.target) {
                                target = part.target.id;
                            }
                            else {
                                target = part.target.toObject(serializer);
                            }
                        }
                        return {
                            ...part,
                            target,
                        };
                    }
                    case "relative-link": {
                        return {
                            ...part,
                        };
                    }
                }
            });
        }
        //Since display parts are plain objects, this lives here
        static deserializeDisplayParts(de, parts) {
            const links = [];
            const files = [];
            const result = parts.map((part) => {
                switch (part.kind) {
                    case "text":
                    case "code":
                        return { ...part };
                    case "inline-tag": {
                        if (typeof part.target === "number") {
                            const part2 = {
                                kind: part.kind,
                                tag: part.tag,
                                text: part.text,
                                target: undefined,
                                tsLinkText: part.tsLinkText,
                            };
                            links.push([part.target, part2]);
                            return part2;
                        }
                        else if (typeof part.target === "string" ||
                            part.target === undefined) {
                            return {
                                kind: "inline-tag",
                                tag: part.tag,
                                text: part.text,
                                target: part.target,
                                tsLinkText: part.tsLinkText,
                            };
                        }
                        else if (typeof part.target === "object") {
                            return {
                                kind: "inline-tag",
                                tag: part.tag,
                                text: part.text,
                                target: new ReflectionSymbolId(part.target),
                                tsLinkText: part.tsLinkText,
                            };
                        }
                        else {
                            assertNever(part.target);
                        }
                        break;
                    }
                    case "relative-link": {
                        if (part.target) {
                            const part2 = {
                                kind: "relative-link",
                                text: part.text,
                                target: null,
                                targetAnchor: part.targetAnchor,
                            };
                            files.push([part.target, part2]);
                            return part2;
                        }
                        return {
                            ...part,
                            target: undefined,
                            targetAnchor: part.targetAnchor,
                        };
                    }
                }
            });
            if (links.length || files.length) {
                de.defer((project) => {
                    for (const [oldFileId, part] of files) {
                        part.target = de.oldFileIdToNewFileId[oldFileId];
                    }
                    for (const [oldId, part] of links) {
                        part.target = project.getReflectionById(de.oldIdToNewId[oldId] ?? -1);
                        if (!part.target) {
                            de.logger.warn(de.application.i18n.serialized_project_referenced_0_not_part_of_project(oldId.toString()));
                        }
                    }
                });
            }
            return result;
        }
        /**
         * Splits the provided parts into a header (first line, as a string)
         * and body (remaining lines). If the header line contains inline tags
         * they will be serialized to a string.
         */
        static splitPartsToHeaderAndBody(parts) {
            let index = parts.findIndex((part) => {
                switch (part.kind) {
                    case "text":
                    case "code":
                        return part.text.includes("\n");
                    case "inline-tag":
                    case "relative-link":
                        return false;
                }
            });
            if (index === -1) {
                return {
                    header: Comment.combineDisplayParts(parts),
                    body: [],
                };
            }
            // Do not split a code block, stop the header at the end of the previous block
            if (parts[index].kind === "code") {
                --index;
            }
            if (index === -1) {
                return { header: "", body: Comment.cloneDisplayParts(parts) };
            }
            let header = Comment.combineDisplayParts(parts.slice(0, index));
            const split = parts[index].text.indexOf("\n");
            let body;
            if (split === -1) {
                header += parts[index].text;
                body = Comment.cloneDisplayParts(parts.slice(index + 1));
            }
            else {
                header += parts[index].text.substring(0, split);
                body = Comment.cloneDisplayParts(parts.slice(index));
                body[0].text = body[0].text.substring(split + 1);
            }
            if (!body[0].text) {
                body.shift();
            }
            return { header: header.trim(), body };
        }
        /**
         * The content of the comment which is not associated with a block tag.
         */
        summary;
        /**
         * All associated block level tags.
         */
        blockTags = [];
        /**
         * All modifier tags present on the comment, e.g. `@alpha`, `@beta`.
         */
        modifierTags = new Set();
        /**
         * Label associated with this reflection, if any (https://tsdoc.org/pages/tags/label/)
         */
        label;
        /**
         * Full path to the file where this comment originated from, if any.
         * This field will not be serialized, so will not be present when handling JSON-revived reflections.
         *
         * Note: This field is non-enumerable to make testing comment contents with `deepEqual` easier.
         */
        sourcePath = __runInitializers(this, _sourcePath_initializers, void 0);
        /**
         * Internal discovery ID used to prevent symbol comments from
         * being duplicated on signatures. Only set when the comment was created
         * from a `ts.CommentRange`.
         * @internal
         */
        discoveryId = (__runInitializers(this, _sourcePath_extraInitializers), __runInitializers(this, _discoveryId_initializers, void 0));
        /**
         * If the comment was inherited from a different "parent" declaration
         * (see #2545), then it is desirable to know this as any `@param` tags
         * which do not apply should not cause warnings. This is not serialized,
         * and only set when the comment was created from a `ts.CommentRange`.
         */
        inheritedFromParentDeclaration = (__runInitializers(this, _discoveryId_extraInitializers), __runInitializers(this, _inheritedFromParentDeclaration_initializers, void 0));
        /**
         * Creates a new Comment instance.
         */
        constructor(summary = [], blockTags = [], modifierTags = new Set()) {
            __runInitializers(this, _inheritedFromParentDeclaration_extraInitializers);
            this.summary = summary;
            this.blockTags = blockTags;
            this.modifierTags = modifierTags;
            extractLabelTag(this);
        }
        /**
         * Gets either the `@summary` tag, or a short version of the comment summary
         * section for rendering in module/namespace pages.
         */
        getShortSummary(useFirstParagraph) {
            const tag = this.getTag("@summary");
            if (tag)
                return tag.content;
            if (!useFirstParagraph)
                return [];
            let partsEnd = this.summary.findIndex((part) => {
                switch (part.kind) {
                    case "text":
                        return part.text.includes("\n\n");
                    case "code":
                        return part.text.includes("\n");
                    case "inline-tag":
                    case "relative-link":
                        return false;
                    default:
                        assertNever(part);
                }
            });
            const foundEnd = partsEnd !== -1;
            if (partsEnd === -1) {
                partsEnd = this.summary.length - 1;
            }
            const summaryParts = this.summary.slice(0, partsEnd);
            if (partsEnd !== -1) {
                const text = this.summary[partsEnd].text;
                const paragraphEnd = text.indexOf("\n\n");
                if (paragraphEnd !== -1) {
                    summaryParts.push({
                        ...this.summary[partsEnd],
                        text: text.slice(0, paragraphEnd),
                    });
                }
                else if (!foundEnd) {
                    summaryParts.push(this.summary[partsEnd]);
                }
            }
            return summaryParts;
        }
        /**
         * Checks if this comment is roughly equal to the other comment.
         * This isn't exactly equal, but just "roughly equal" by the comment
         * text.
         */
        similarTo(other) {
            if (Comment.combineDisplayParts(this.summary) !==
                Comment.combineDisplayParts(other.summary)) {
                return false;
            }
            // Ignore modifier tags, as they could cause false negatives
            // if a cascaded modifier tag is present in one comment but not the other.
            if (this.blockTags.length !== other.blockTags.length) {
                return false;
            }
            for (let i = 0; i < this.blockTags.length; ++i) {
                if (!this.blockTags[i].similarTo(other.blockTags[i])) {
                    return false;
                }
            }
            return true;
        }
        /**
         * Create a deep clone of this comment.
         */
        clone() {
            const comment = new Comment(Comment.cloneDisplayParts(this.summary), this.blockTags.map((tag) => tag.clone()), new Set(this.modifierTags));
            comment.discoveryId = this.discoveryId;
            comment.sourcePath = this.sourcePath;
            comment.inheritedFromParentDeclaration =
                this.inheritedFromParentDeclaration;
            return comment;
        }
        /**
         * Returns true if this comment is completely empty.
         * @internal
         */
        isEmpty() {
            return !this.hasVisibleComponent() && this.modifierTags.size === 0;
        }
        /**
         * Has this comment a visible component?
         *
         * @returns TRUE when this comment has a visible component.
         */
        hasVisibleComponent() {
            return (this.summary.some((x) => x.kind !== "text" || x.text !== "") ||
                this.blockTags.length > 0);
        }
        /**
         * Test whether this comment contains a tag with the given name.
         *
         * @param tagName  The name of the tag to look for.
         * @returns TRUE when this comment contains a tag with the given name, otherwise FALSE.
         */
        hasModifier(tagName) {
            return this.modifierTags.has(tagName);
        }
        removeModifier(tagName) {
            this.modifierTags.delete(tagName);
        }
        /**
         * Return the first tag with the given name.
         *
         * @param tagName  The name of the tag to look for.
         * @returns The found tag or undefined.
         */
        getTag(tagName) {
            return this.blockTags.find((tag) => tag.tag === tagName);
        }
        /**
         * Get all tags with the given tag name.
         */
        getTags(tagName) {
            return this.blockTags.filter((tag) => tag.tag === tagName);
        }
        getIdentifiedTag(identifier, tagName) {
            return this.blockTags.find((tag) => tag.tag === tagName && tag.name === identifier);
        }
        /**
         * Removes all block tags with the given tag name from the comment.
         * @param tagName
         */
        removeTags(tagName) {
            removeIf(this.blockTags, (tag) => tag.tag === tagName);
        }
        toObject(serializer) {
            return {
                summary: Comment.serializeDisplayParts(serializer, this.summary),
                blockTags: serializer.toObjectsOptional(this.blockTags),
                modifierTags: this.modifierTags.size > 0
                    ? Array.from(this.modifierTags)
                    : undefined,
                label: this.label,
            };
        }
        fromObject(de, obj) {
            this.summary = Comment.deserializeDisplayParts(de, obj.summary);
            this.blockTags =
                obj.blockTags?.map((tagObj) => {
                    const tag = new CommentTag(tagObj.tag, []);
                    de.fromObject(tag, tagObj);
                    return tag;
                }) || [];
            this.modifierTags = new Set(obj.modifierTags);
            this.label = obj.label;
        }
    };
})();
export { Comment };
function extractLabelTag(comment) {
    const index = comment.summary.findIndex((part) => part.kind === "inline-tag" && part.tag === "@label");
    if (index !== -1) {
        comment.label = comment.summary.splice(index, 1)[0].text;
    }
}
