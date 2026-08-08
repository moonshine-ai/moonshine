import { DeclarationReflection } from "./declaration.js";
import { ReflectionKind } from "./kind.js";
/**
 * Describes a reflection which does not exist at this location, but is referenced. Used for imported reflections.
 *
 * ```ts
 * // a.ts
 * export const a = 1;
 * // b.ts
 * import { a } from './a';
 * // Here to avoid extra work we create a reference to the original reflection in module a instead
 * // of copying the reflection.
 * export { a };
 * ```
 * @category Reflections
 */
export class ReferenceReflection extends DeclarationReflection {
    variant = "reference";
    _target;
    /**
     * Creates a reference reflection. Should only be used within the factory function.
     * @internal
     */
    constructor(name, reflection, parent) {
        super(name, ReflectionKind.Reference, parent);
        this._target = reflection.id;
    }
    /**
     * Tries to get the reflection that is referenced. This may be another reference reflection.
     * To fully resolve any references, use {@link tryGetTargetReflectionDeep}.
     */
    tryGetTargetReflection() {
        return this.project.getReflectionById(this._target);
    }
    /**
     * Tries to get the reflection that is referenced, this will fully resolve references.
     * To only resolve one reference, use {@link tryGetTargetReflection}.
     */
    tryGetTargetReflectionDeep() {
        let result = this.tryGetTargetReflection();
        while (result instanceof ReferenceReflection) {
            result = result.tryGetTargetReflection();
        }
        return result;
    }
    /**
     * Gets the reflection that is referenced. This may be another reference reflection.
     * To fully resolve any references, use {@link getTargetReflectionDeep}.
     */
    getTargetReflection() {
        const target = this.tryGetTargetReflection();
        if (!target) {
            throw new Error("Reference was unresolved.");
        }
        return target;
    }
    /**
     * Gets the reflection that is referenced, this will fully resolve references.
     * To only resolve one reference, use {@link getTargetReflection}.
     */
    getTargetReflectionDeep() {
        let result = this.getTargetReflection();
        while (result instanceof ReferenceReflection) {
            result = result.getTargetReflection();
        }
        return result;
    }
    getChildByName(arg) {
        return this.getTargetReflection().getChildByName(arg);
    }
    toObject(serializer) {
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            target: this.tryGetTargetReflection()?.id ?? -1,
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        de.defer((project) => {
            this._target =
                project.getReflectionById(de.oldIdToNewId[obj.target] ?? -1)
                    ?.id ?? -1;
        });
    }
}
