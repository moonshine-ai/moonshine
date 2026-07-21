import type { Serializer } from "./serializer.js";
import type { ModelToObject } from "./schema.js";
/**
 * Represents Serializer plugin component.
 *
 * Like {@link Converter} plugins each {@link Serializer} plugin defines a predicate that instructs if an
 * object can be serialized by it. This is done dynamically at runtime via a `supports` method.
 */
export interface SerializerComponent<T extends object> {
    /**
     * The priority this serializer should be executed with.
     * A higher priority means the {@link Serializer} will be applied earlier.
     */
    readonly priority: number;
    /**
     * Technically this should return `item is T`, but that doesn't play nicely
     * with inference, so allow the looser `boolean` return type.
     * @param item
     */
    supports(item: unknown): boolean;
    toObject(item: T, obj: Partial<ModelToObject<T>>, serializer: Serializer): Partial<ModelToObject<T>>;
}
