import { Reflection, type TraverseCallback } from "./abstract.js";
import { ReflectionCategory } from "../ReflectionCategory.js";
import { ReflectionGroup } from "../ReflectionGroup.js";
import type { ReflectionKind } from "./kind.js";
import type { Serializer, JSONOutput, Deserializer } from "../../serialization/index.js";
import type { DocumentReflection } from "./document.js";
import type { DeclarationReflection } from "./declaration.js";
/**
 * @category Reflections
 */
export declare abstract class ContainerReflection extends Reflection {
    /**
     * The children of this reflection. Do not add reflections to this array
     * manually. Instead call {@link addChild}.
     */
    children?: Array<DeclarationReflection>;
    /**
     * Documents associated with this reflection.
     *
     * These are not children as including them as children requires code handle both
     * types, despite being mostly unrelated and handled separately.
     *
     * Including them here in a separate array neatly handles that problem, but also
     * introduces another one for rendering. When rendering, documents should really
     * actually be considered part of the "children" of a reflection. For this reason,
     * we also maintain a list of child declarations with child documents which is used
     * when rendering.
     */
    documents?: Array<DocumentReflection>;
    /**
     * Union of the {@link children} and {@link documents} arrays which dictates the
     * sort order for rendering.
     */
    childrenIncludingDocuments?: Array<DeclarationReflection | DocumentReflection>;
    /**
     * All children grouped by their kind.
     */
    groups?: ReflectionGroup[];
    /**
     * All children grouped by their category.
     */
    categories?: ReflectionCategory[];
    /**
     * Return a list of all children of a certain kind.
     *
     * @param kind  The desired kind of children.
     * @returns     An array containing all children with the desired kind.
     */
    getChildrenByKind(kind: ReflectionKind): DeclarationReflection[];
    addChild(child: DeclarationReflection | DocumentReflection): void;
    removeChild(child: DeclarationReflection | DocumentReflection): void;
    traverse(callback: TraverseCallback): void;
    toObject(serializer: Serializer): JSONOutput.ContainerReflection;
    fromObject(de: Deserializer, obj: JSONOutput.ContainerReflection): void;
}
