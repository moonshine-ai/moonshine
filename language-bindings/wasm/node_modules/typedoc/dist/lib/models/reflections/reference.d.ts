import { DeclarationReflection } from "./declaration.js";
import type { Serializer, JSONOutput, Deserializer } from "../../serialization/index.js";
import type { Reflection } from "./abstract.js";
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
export declare class ReferenceReflection extends DeclarationReflection {
    readonly variant = "reference";
    private _target;
    /**
     * Creates a reference reflection. Should only be used within the factory function.
     * @internal
     */
    constructor(name: string, reflection: Reflection, parent?: Reflection);
    /**
     * Tries to get the reflection that is referenced. This may be another reference reflection.
     * To fully resolve any references, use {@link tryGetTargetReflectionDeep}.
     */
    tryGetTargetReflection(): Reflection | undefined;
    /**
     * Tries to get the reflection that is referenced, this will fully resolve references.
     * To only resolve one reference, use {@link tryGetTargetReflection}.
     */
    tryGetTargetReflectionDeep(): Reflection | undefined;
    /**
     * Gets the reflection that is referenced. This may be another reference reflection.
     * To fully resolve any references, use {@link getTargetReflectionDeep}.
     */
    getTargetReflection(): Reflection;
    /**
     * Gets the reflection that is referenced, this will fully resolve references.
     * To only resolve one reference, use {@link getTargetReflection}.
     */
    getTargetReflectionDeep(): Reflection;
    getChildByName(arg: string | string[]): Reflection | undefined;
    toObject(serializer: Serializer): JSONOutput.ReferenceReflection;
    fromObject(de: Deserializer, obj: JSONOutput.ReferenceReflection): void;
}
