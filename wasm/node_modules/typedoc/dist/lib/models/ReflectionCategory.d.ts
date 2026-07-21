import type { CommentDisplayPart, DeclarationReflection, DocumentReflection } from "./index.js";
import type { Serializer, JSONOutput, Deserializer } from "../serialization/index.js";
/**
 * A category of reflections.
 *
 * Reflection categories are created by the ´CategoryPlugin´ in the resolving phase
 * of the dispatcher. The main purpose of categories is to be able to more easily
 * render human readable children lists in templates.
 */
export declare class ReflectionCategory {
    /**
     * The title, a string representation of this category.
     */
    title: string;
    /**
     * The user specified description, if any, set with `@categoryDescription`
     */
    description?: CommentDisplayPart[];
    /**
     * All reflections of this category.
     */
    children: Array<DeclarationReflection | DocumentReflection>;
    /**
     * Create a new ReflectionCategory instance.
     *
     * @param title The title of this category.
     */
    constructor(title: string);
    /**
     * Do all children of this category have a separate document?
     */
    allChildrenHaveOwnDocument(): boolean;
    toObject(serializer: Serializer): JSONOutput.ReflectionCategory;
    fromObject(de: Deserializer, obj: JSONOutput.ReflectionCategory): void;
}
