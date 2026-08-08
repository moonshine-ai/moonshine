import { ReflectionCategory } from "./ReflectionCategory.js";
import type { CommentDisplayPart, DeclarationReflection, DocumentReflection, Reflection } from "./index.js";
import type { Serializer, JSONOutput, Deserializer } from "../serialization/index.js";
/**
 * A group of reflections. All reflections in a group are of the same kind.
 *
 * Reflection groups are created by the ´GroupHandler´ in the resolving phase
 * of the dispatcher. The main purpose of groups is to be able to more easily
 * render human readable children lists in templates.
 */
export declare class ReflectionGroup {
    readonly owningReflection: Reflection;
    /**
     * The title, a string representation of the typescript kind, of this group.
     */
    title: string;
    /**
     * User specified description via `@groupDescription`, if specified.
     */
    description?: CommentDisplayPart[];
    /**
     * All reflections of this group.
     */
    children: Array<DeclarationReflection | DocumentReflection>;
    /**
     * Categories contained within this group.
     */
    categories?: ReflectionCategory[];
    /**
     * Create a new ReflectionGroup instance.
     *
     * @param title The title of this group.
     * @param owningReflection The reflection containing this group, useful for changing rendering based on a comment on a reflection.
     */
    constructor(title: string, owningReflection: Reflection);
    /**
     * Do all children of this group have a separate document?
     */
    allChildrenHaveOwnDocument(): boolean;
    toObject(serializer: Serializer): JSONOutput.ReflectionGroup;
    fromObject(de: Deserializer, obj: JSONOutput.ReflectionGroup): void;
}
