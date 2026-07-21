import { Comment } from "../comments/index.js";
import { Reflection, TraverseProperty, } from "./abstract.js";
import { ReflectionKind } from "./kind.js";
/**
 * Non-TS reflection type which is used to represent markdown documents included in the docs.
 */
export class DocumentReflection extends Reflection {
    variant = "document";
    /**
     * The content to be displayed on the page for this reflection.
     */
    content;
    /**
     * Frontmatter included in document
     */
    frontmatter;
    /**
     * A precomputed boost derived from the searchCategoryBoosts and searchGroupBoosts options, used when
     * boosting search relevance scores at runtime. May be modified by plugins.
     */
    relevanceBoost;
    /**
     * Child documents, if any are present.
     */
    children;
    constructor(name, parent, content, frontmatter) {
        super(name, ReflectionKind.Document, parent);
        this.content = content;
        this.frontmatter = frontmatter;
        if (typeof frontmatter["title"] === "string") {
            this.name = frontmatter["title"];
            delete frontmatter["title"];
        }
    }
    addChild(child) {
        this.children ||= [];
        this.children.push(child);
    }
    isDocument() {
        return true;
    }
    traverse(callback) {
        for (const child of this.children || []) {
            if (callback(child, TraverseProperty.Documents) === false) {
                return;
            }
        }
    }
    toObject(serializer) {
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            content: Comment.serializeDisplayParts(serializer, this.content),
            frontmatter: this.frontmatter,
            relevanceBoost: this.relevanceBoost,
            children: serializer.toObjectsOptional(this.children),
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        this.content = Comment.deserializeDisplayParts(de, obj.content);
        this.frontmatter = obj.frontmatter;
        this.relevanceBoost = obj.relevanceBoost;
        this.children = de.reviveMany(obj.children, (obj) => de.reflectionBuilders.document(this, obj));
    }
}
