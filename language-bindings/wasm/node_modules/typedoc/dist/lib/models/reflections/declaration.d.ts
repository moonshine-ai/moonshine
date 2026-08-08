import type ts from "typescript";
import { type ReferenceType, type SomeType } from "../types.js";
import { type TraverseCallback } from "./abstract.js";
import { ContainerReflection } from "./container.js";
import type { SignatureReflection } from "./signature.js";
import type { TypeParameterReflection } from "./type-parameter.js";
import type { Serializer, JSONOutput, Deserializer } from "../../serialization/index.js";
import { type CommentDisplayPart } from "../comments/index.js";
import { SourceReference } from "../sources/file.js";
/**
 * Stores hierarchical type data.
 *
 * @see {@link DeclarationReflection.typeHierarchy}
 */
export interface DeclarationHierarchy {
    /**
     * The types represented by this node in the hierarchy.
     */
    types: SomeType[];
    /**
     * The next hierarchy level.
     */
    next?: DeclarationHierarchy;
    /**
     * Is this the entry containing the target type?
     */
    isTarget?: boolean;
}
/**
 * A reflection that represents a single declaration emitted by the TypeScript compiler.
 *
 * All parts of a project are represented by DeclarationReflection instances. The actual
 * kind of a reflection is stored in its ´kind´ member.
 * @category Reflections
 */
export declare class DeclarationReflection extends ContainerReflection {
    readonly variant: "declaration" | "reference";
    /**
     * A list of all source files that contributed to this reflection.
     */
    sources?: SourceReference[];
    /**
     * Precomputed boost for search results, may be less than 1 to de-emphasize this member in search results.
     * Does NOT include group/category values as they are computed when building the JS index.
     *
     * This is preserved for plugins, and may be removed in 0.28 if no plugins have used it yet.
     */
    relevanceBoost?: number;
    /**
     * The escaped name of this declaration assigned by the TS compiler if there is an associated symbol.
     * This is used to retrieve properties for analyzing inherited members.
     *
     * Not serialized, only useful during conversion.
     * @internal
     */
    escapedName?: ts.__String;
    /**
     * The type of the reflection.
     *
     * If the reflection represents a variable or a property, this is the value type.<br />
     * If the reflection represents a signature, this is the return type.
     */
    type?: SomeType;
    typeParameters?: TypeParameterReflection[];
    /**
     * A list of call signatures attached to this declaration.
     *
     * TypeDoc creates one declaration per function that may contain one or more
     * signature reflections.
     */
    signatures?: SignatureReflection[];
    /**
     * The index signature of this declaration.
     */
    indexSignatures?: SignatureReflection[];
    /**
     * The get signature of this declaration.
     */
    getSignature?: SignatureReflection;
    /**
     * The set signature of this declaration.
     */
    setSignature?: SignatureReflection;
    /**
     * The default value of this reflection.
     *
     * Applies to function parameters, variables, and properties.
     */
    defaultValue?: string;
    /**
     * A type that points to the reflection that has been overwritten by this reflection.
     *
     * Applies to interface and class members.
     */
    overwrites?: ReferenceType;
    /**
     * A type that points to the reflection this reflection has been inherited from.
     *
     * Applies to interface and class members.
     */
    inheritedFrom?: ReferenceType;
    /**
     * A type that points to the reflection this reflection is the implementation of.
     *
     * Applies to class members.
     */
    implementationOf?: ReferenceType;
    /**
     * A list of all types this reflection extends (e.g. the parent classes).
     */
    extendedTypes?: SomeType[];
    /**
     * A list of all types that extend this reflection (e.g. the subclasses).
     */
    extendedBy?: ReferenceType[];
    /**
     * A list of all types this reflection implements.
     */
    implementedTypes?: SomeType[];
    /**
     * A list of all types that implement this reflection.
     */
    implementedBy?: ReferenceType[];
    /**
     * Contains a simplified representation of the type hierarchy suitable for being
     * rendered in templates.
     */
    typeHierarchy?: DeclarationHierarchy;
    /**
     * The contents of the readme file of the module when found.
     */
    readme?: CommentDisplayPart[];
    /**
     * The version of the module when found.
     */
    packageVersion?: string;
    isDeclaration(): this is DeclarationReflection;
    hasGetterOrSetter(): boolean;
    getAllSignatures(): SignatureReflection[];
    getNonIndexSignatures(): SignatureReflection[];
    getProperties(): DeclarationReflection[];
    getChildOrTypePropertyByName(path: string[]): DeclarationReflection | undefined;
    traverse(callback: TraverseCallback): void;
    /**
     * Return a string representation of this reflection.
     */
    toString(): string;
    toObject(serializer: Serializer): JSONOutput.DeclarationReflection;
    fromObject(de: Deserializer, obj: JSONOutput.DeclarationReflection | JSONOutput.ProjectReflection): void;
}
