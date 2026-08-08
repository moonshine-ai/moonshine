import type ts from "typescript";
import { type Reflection } from "./abstract.js";
import { ContainerReflection } from "./container.js";
import type { DeclarationReflection } from "./declaration.js";
import { type Type } from "../types.js";
import { ReflectionKind } from "./kind.js";
import { type CommentDisplayPart } from "../comments/index.js";
import { ReflectionSymbolId } from "./ReflectionSymbolId.js";
import type { Serializer } from "../../serialization/serializer.js";
import type { Deserializer, JSONOutput } from "../../serialization/index.js";
import type { FileRegistry } from "../FileRegistry.js";
/**
 * A reflection that represents the root of the project.
 *
 * The project reflection acts as a global index, one may receive all reflections
 * and source files of the processed project through this reflection.
 * @category Reflections
 */
export declare class ProjectReflection extends ContainerReflection {
    readonly variant = "project";
    private symbolToReflectionIdMap;
    private reflectionIdToSymbolIdMap;
    private reflectionIdToSymbolMap;
    private removedSymbolIds;
    private referenceGraph?;
    private reflectionChildren;
    /**
     * A list of all reflections within the project. DO NOT MUTATE THIS OBJECT.
     * All mutation should be done via {@link registerReflection} and {@link removeReflection}
     * to ensure that links to reflections remain valid.
     *
     * This may be replaced with a `Map<number, Reflection>` someday.
     */
    reflections: {
        [id: number]: Reflection;
    };
    /**
     * The name of the package that this reflection documents according to package.json.
     */
    packageName?: string;
    /**
     * The version of the package that this reflection documents according to package.json.
     */
    packageVersion?: string;
    /**
     * The contents of the readme.md file of the project when found.
     */
    readme?: CommentDisplayPart[];
    /**
     * Object which describes where to find content for relative links.
     */
    readonly files: FileRegistry;
    constructor(name: string, registry: FileRegistry);
    /**
     * Return whether this reflection is the root / project reflection.
     */
    isProject(): this is ProjectReflection;
    /**
     * Return a list of all reflections in this project of a certain kind.
     *
     * @param kind  The desired kind of reflection.
     * @returns     An array containing all reflections with the desired kind.
     */
    getReflectionsByKind(kind: ReflectionKind): Reflection[];
    /**
     * Registers the given reflection so that it can be quickly looked up by helper methods.
     * Should be called for *every* reflection added to the project.
     */
    registerReflection(reflection: Reflection, symbol: ts.Symbol | undefined, filePath: string | undefined): void;
    /**
     * Removes references to reflections contained within the provided type.
     * Plugins which overwrite types on reflections should pass the type to this
     * method before overwriting the property.
     * @since 0.26.6
     */
    removeTypeReflections(type: Type | undefined): void;
    /**
     * Removes a reflection from the documentation. Can be used by plugins to filter reflections
     * out of the generated documentation. Has no effect if the reflection is not present in the
     * project.
     */
    removeReflection(reflection: Reflection): void;
    /** @internal */
    mergeModules(source: DeclarationReflection, target: DeclarationReflection | ProjectReflection): void;
    /**
     * Remove a reflection without updating the parent reflection to remove references to the removed reflection.
     */
    private _removeReflection;
    /**
     * Gets the reflection registered for the given reflection ID, or undefined if it is not present
     * in the project.
     */
    getReflectionById(id: number): Reflection | undefined;
    /**
     * Gets the reflection associated with the given symbol, if it exists.
     * @internal
     */
    getReflectionFromSymbol(symbol: ts.Symbol): Reflection | undefined;
    /**
     * Gets the reflection associated with the given symbol id, if it exists.
     * If there are multiple reflections associated with this symbol, gets the first one.
     * @internal
     */
    getReflectionFromSymbolId(symbolId: ReflectionSymbolId): Reflection | undefined;
    /** @internal */
    getReflectionsFromSymbolId(symbolId: ReflectionSymbolId): Reflection[];
    /** @internal */
    getSymbolIdFromReflection(reflection: Reflection): ReflectionSymbolId | undefined;
    /** @internal */
    registerSymbolId(reflection: Reflection, id: ReflectionSymbolId): void;
    symbolIdHasBeenRemoved(id: ReflectionSymbolId): boolean;
    /**
     * THIS MAY NOT BE USED AFTER CONVERSION HAS FINISHED.
     * @internal
     */
    getSymbolFromReflection(reflection: Reflection): ts.Symbol | undefined;
    private getReferenceGraph;
    toObject(serializer: Serializer): JSONOutput.ProjectReflection;
    fromObject(de: Deserializer, obj: JSONOutput.ProjectReflection): void;
}
