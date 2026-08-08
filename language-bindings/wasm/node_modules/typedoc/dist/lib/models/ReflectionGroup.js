import { ReflectionCategory } from "./ReflectionCategory.js";
import { Comment } from "./comments/index.js";
/**
 * A group of reflections. All reflections in a group are of the same kind.
 *
 * Reflection groups are created by the ´GroupHandler´ in the resolving phase
 * of the dispatcher. The main purpose of groups is to be able to more easily
 * render human readable children lists in templates.
 */
export class ReflectionGroup {
    owningReflection;
    /**
     * The title, a string representation of the typescript kind, of this group.
     */
    title;
    /**
     * User specified description via `@groupDescription`, if specified.
     */
    description;
    /**
     * All reflections of this group.
     */
    children = [];
    /**
     * Categories contained within this group.
     */
    categories;
    /**
     * Create a new ReflectionGroup instance.
     *
     * @param title The title of this group.
     * @param owningReflection The reflection containing this group, useful for changing rendering based on a comment on a reflection.
     */
    constructor(title, owningReflection) {
        this.owningReflection = owningReflection;
        this.title = title;
    }
    /**
     * Do all children of this group have a separate document?
     */
    allChildrenHaveOwnDocument() {
        return this.children.every((child) => child.hasOwnDocument);
    }
    toObject(serializer) {
        return {
            title: this.title,
            description: this.description
                ? Comment.serializeDisplayParts(serializer, this.description)
                : undefined,
            children: this.children.length > 0
                ? this.children.map((child) => child.id)
                : undefined,
            categories: serializer.toObjectsOptional(this.categories),
        };
    }
    fromObject(de, obj) {
        if (obj.description) {
            this.description = Comment.deserializeDisplayParts(de, obj.description);
        }
        if (obj.categories) {
            this.categories = obj.categories.map((catObj) => {
                const cat = new ReflectionCategory(catObj.title);
                de.fromObject(cat, catObj);
                return cat;
            });
        }
        if (obj.children) {
            de.defer((project) => {
                for (const childId of obj.children || []) {
                    const child = project.getReflectionById(de.oldIdToNewId[childId] ?? -1);
                    if (child?.isDeclaration() || child?.isDocument()) {
                        this.children.push(child);
                    }
                }
            });
        }
    }
}
