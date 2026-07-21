import type { Application } from "../application.js";
import { ProjectReflection, type ReflectionVariant, type TypeKindMap } from "../models/index.js";
import type { Logger } from "../utils/loggers.js";
import type { JSONOutput } from "./index.js";
import type { FileRegistry } from "../models/FileRegistry.js";
export interface DeserializerComponent {
    priority: number;
    supports(model: unknown, obj: unknown): boolean;
    fromObject(model: unknown, obj: unknown): void;
}
export interface Deserializable<T> {
    fromObject(d: Deserializer, o: T): void;
}
/**
 * Deserializes TypeDoc's JSON output back to {@link Reflection} instances.
 *
 * @group Common
 * @summary Deserializes TypeDoc's JSON output
 */
export declare class Deserializer {
    readonly application: Application;
    private deferred;
    private deserializers;
    private activeReflection;
    constructor(application: Application);
    get logger(): Logger;
    reflectionBuilders: {
        [K in keyof ReflectionVariant]: (parent: NonNullable<ReflectionVariant[K]["parent"]>, obj: JSONOutput.ModelToObject<ReflectionVariant[K]>) => ReflectionVariant[K];
    };
    typeBuilders: {
        [K in keyof TypeKindMap]: (obj: JSONOutput.ModelToObject<TypeKindMap[K]>, de: Deserializer) => TypeKindMap[K];
    };
    /**
     * Only set when deserializing.
     */
    projectRoot: string;
    oldIdToNewId: Record<number, number | undefined>;
    oldFileIdToNewFileId: Record<number, number | undefined>;
    project: ProjectReflection | undefined;
    addDeserializer(de: DeserializerComponent): void;
    /**
     * Revive a single project into the structure it was originally created with.
     * This is generally not appropriate for merging multiple projects since projects may
     * contain reflections in their root, not inside a module.
     */
    reviveProject(name: string, projectObj: JSONOutput.ProjectReflection, options: {
        projectRoot: string;
        registry: FileRegistry;
        addProjectDocuments?: boolean;
    }): ProjectReflection;
    reviveProjects(name: string, projects: readonly JSONOutput.ProjectReflection[], options: {
        projectRoot: string;
        registry: FileRegistry;
        addProjectDocuments?: boolean;
    }): ProjectReflection;
    revive<T, U extends Deserializable<T>>(source: NonNullable<T>, creator: (obj: T) => U): U;
    revive<T, U extends Deserializable<T>>(source: T | undefined, creator: (obj: T) => U): U | undefined;
    reviveMany<T, U extends Deserializable<T>>(sourceArray: T[], creator: (obj: T) => U): U[];
    reviveMany<T, U extends Deserializable<T>>(sourceArray: T[] | undefined, creator: (obj: T) => U): U[] | undefined;
    reviveType<T extends JSONOutput.SomeType>(obj: T): TypeKindMap[T["type"]];
    reviveType<T extends JSONOutput.SomeType>(obj: T | undefined): TypeKindMap[T["type"]] | undefined;
    constructReflection<T extends JSONOutput.SomeReflection>(obj: T): ReflectionVariant[T["variant"]];
    constructType<T extends JSONOutput.SomeType>(obj: T): TypeKindMap[T["type"]];
    fromObject<T>(receiver: {
        fromObject(d: Deserializer, o: T): void;
    }, obj: T): void;
    /**
     * Defers work until the initial pass of serialization has been completed.
     * This can be used to set up references which cannot be immediately restored.
     *
     * May only be called when deserializing.
     */
    defer(cb: (project: ProjectReflection) => void): void;
}
