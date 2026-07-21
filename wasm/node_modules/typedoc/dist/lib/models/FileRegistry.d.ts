import type { Deserializer, Serializer } from "../serialization/index.js";
import type { FileRegistry as JSONFileRegistry } from "../serialization/schema.js";
import type { ProjectReflection, Reflection } from "./reflections/index.js";
import type { ReflectionId } from "./reflections/abstract.js";
export declare class FileRegistry {
    protected nextId: number;
    protected mediaToReflection: Map<number, ReflectionId>;
    protected mediaToPath: Map<number, string>;
    protected reflectionToPath: Map<ReflectionId, string>;
    protected pathToMedia: Map<string, number>;
    protected names: Map<number, string>;
    protected nameUsage: Map<string, number>;
    registerAbsolute(absolute: string): {
        target: number;
        anchor: string | undefined;
    };
    /** Called by {@link ProjectReflection.registerReflection} @internal*/
    registerReflection(absolute: string, reflection: Reflection): void;
    getReflectionPath(reflection: Reflection): string | undefined;
    register(sourcePath: string, relativePath: string): {
        target: number;
        anchor: string | undefined;
    } | undefined;
    removeReflection(reflection: Reflection): void;
    resolve(id: number, project: ProjectReflection): string | Reflection | undefined;
    getName(id: number): string | undefined;
    getNameToAbsoluteMap(): ReadonlyMap<string, string>;
    toObject(ser: Serializer): JSONFileRegistry;
    /**
     * Revive a file registry from disc.
     * Note that in the packages context this may be called multiple times on
     * a single object, and should merge in files from the other registries.
     */
    fromObject(de: Deserializer, obj: JSONFileRegistry): void;
}
export declare class ValidatingFileRegistry extends FileRegistry {
    register(sourcePath: string, relativePath: string): {
        target: number;
        anchor: string | undefined;
    } | undefined;
    fromObject(de: Deserializer, obj: JSONFileRegistry): void;
}
