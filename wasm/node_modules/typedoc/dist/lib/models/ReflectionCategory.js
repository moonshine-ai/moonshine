import { Comment } from "./comments/index.js";
/**
 * A category of reflections.
 *
 * Reflection categories are created by the ´CategoryPlugin´ in the resolving phase
 * of the dispatcher. The main purpose of categories is to be able to more easily
 * render human readable children lists in templates.
 */
export class ReflectionCategory {
    /**
     * The title, a string representation of this category.
     */
    title;
    /**
     * The user specified description, if any, set with `@categoryDescription`
     */
    description;
    /**
     * All reflections of this category.
     */
    children = [];
    /**
     * Create a new ReflectionCategory instance.
     *
     * @param title The title of this category.
     */
    constructor(title) {
        this.title = title;
    }
    /**
     * Do all children of this category have a separate document?
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
        };
    }
    fromObject(de, obj) {
        if (obj.description) {
            this.description = Comment.deserializeDisplayParts(de, obj.description);
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
