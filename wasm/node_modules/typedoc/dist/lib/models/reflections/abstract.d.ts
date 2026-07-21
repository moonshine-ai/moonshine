import { Comment } from "../comments/comment.js";
import type { ProjectReflection } from "./project.js";
import type { NeverIfInternal } from "../../utils/index.js";
import { ReflectionKind } from "./kind.js";
import type { Serializer, Deserializer, JSONOutput } from "../../serialization/index.js";
import type { ReflectionVariant } from "./variant.js";
import type { DeclarationReflection } from "./declaration.js";
import type { DocumentReflection } from "./document.js";
import type { Internationalization, TranslatedString } from "../../internationalization/index.js";
import type { ParameterReflection } from "./parameter.js";
import type { ReferenceReflection } from "./reference.js";
/**
 * Reset the reflection id.
 *
 * Used by the test cases to ensure the reflection ids won't change between runs.
 */
export declare function resetReflectionID(): void;
export declare enum ReflectionFlag {
    None = 0,
    Private = 1,
    Protected = 2,
    Public = 4,
    Static = 8,
    External = 16,
    Optional = 32,
    Rest = 64,
    Abstract = 128,
    Const = 256,
    Readonly = 512,
    Inherited = 1024
}
/**
 * This must extend Array in order to work with Handlebar's each helper.
 */
export declare class ReflectionFlags {
    private flags;
    hasFlag(flag: ReflectionFlag): boolean;
    /**
     * Is this a private member?
     */
    get isPrivate(): boolean;
    /**
     * Is this a protected member?
     */
    get isProtected(): boolean;
    /**
     * Is this a public member?
     */
    get isPublic(): boolean;
    /**
     * Is this a static member?
     */
    get isStatic(): boolean;
    /**
     * Is this a declaration from an external document?
     */
    get isExternal(): boolean;
    /**
     * Whether this reflection is an optional component or not.
     *
     * Applies to function parameters and object members.
     */
    get isOptional(): boolean;
    /**
     * Whether it's a rest parameter, like `foo(...params);`.
     */
    get isRest(): boolean;
    get isAbstract(): boolean;
    get isConst(): boolean;
    get isReadonly(): boolean;
    get isInherited(): boolean;
    setFlag(flag: ReflectionFlag, set: boolean): void;
    getFlagStrings(i18n: Internationalization): TranslatedString[];
    private setSingleFlag;
    private static serializedFlags;
    toObject(): JSONOutput.ReflectionFlags;
    fromObject(obj: JSONOutput.ReflectionFlags): void;
}
export declare enum TraverseProperty {
    Children = 0,
    Documents = 1,
    Parameters = 2,
    TypeLiteral = 3,
    TypeParameter = 4,
    Signatures = 5,
    IndexSignature = 6,
    GetSignature = 7,
    SetSignature = 8
}
export interface TraverseCallback {
    /**
     * May return false to bail out of any further iteration. To preserve backwards compatibility, if
     * a function returns undefined, iteration must continue.
     */
    (reflection: Reflection, property: TraverseProperty): boolean | NeverIfInternal<void>;
}
export type ReflectionVisitor = {
    [K in keyof ReflectionVariant]?: (refl: ReflectionVariant[K]) => void;
};
export type ReflectionId = number & {
    __reflectionIdBrand: never;
};
/**
 * Base class for all reflection classes.
 *
 * While generating a documentation, TypeDoc generates an instance of {@link ProjectReflection}
 * as the root for all reflections within the project. All other reflections are represented
 * by the {@link DeclarationReflection} class.
 *
 * This base class exposes the basic properties one may use to traverse the reflection tree.
 * You can use the {@link ContainerReflection.children} and {@link parent} properties to walk the tree. The {@link ContainerReflection.groups} property
 * contains a list of all children grouped and sorted for rendering.
 * @category Reflections
 */
export declare abstract class Reflection {
    /**
     * Discriminator representing the type of reflection represented by this object.
     */
    abstract readonly variant: keyof ReflectionVariant;
    /**
     * Unique id of this reflection.
     */
    id: ReflectionId;
    /**
     * The symbol name of this reflection.
     */
    name: string;
    /**
     * The kind of this reflection.
     */
    kind: ReflectionKind;
    flags: ReflectionFlags;
    /**
     * The reflection this reflection is a child of.
     */
    parent?: Reflection;
    project: ProjectReflection;
    /**
     * The parsed documentation comment attached to this reflection.
     */
    comment?: Comment;
    /**
     * The url of this reflection in the generated documentation.
     * TODO: Reflections shouldn't know urls exist. Move this to a serializer.
     */
    url?: string;
    /**
     * The name of the anchor of this child.
     * TODO: Reflections shouldn't know anchors exist. Move this to a serializer.
     */
    anchor?: string;
    /**
     * Is the url pointing to an individual document?
     *
     * When FALSE, the url points to an anchor tag on a page of a different reflection.
     * TODO: Reflections shouldn't know how they are rendered. Move this to the correct serializer.
     */
    hasOwnDocument?: boolean;
    constructor(name: string, kind: ReflectionKind, parent?: Reflection);
    /**
     * Test whether this reflection is of the given kind.
     */
    kindOf(kind: ReflectionKind | ReflectionKind[]): boolean;
    /**
     * Return the full name of this reflection. Intended for use in debugging. For log messages
     * intended to be displayed to the user for them to fix, prefer {@link getFriendlyFullName} instead.
     *
     * The full name contains the name of this reflection and the names of all parent reflections.
     *
     * @param separator  Separator used to join the names of the reflections.
     * @returns The full name of this reflection.
     */
    getFullName(separator?: string): string;
    /**
     * Return the full name of this reflection, with signature names dropped if possible without
     * introducing ambiguity in the name.
     */
    getFriendlyFullName(): string;
    /**
     * Set a flag on this reflection.
     */
    setFlag(flag: ReflectionFlag, value?: boolean): void;
    /**
     * Has this reflection a visible comment?
     *
     * @returns TRUE when this reflection has a visible comment.
     */
    hasComment(): boolean;
    hasGetterOrSetter(): boolean;
    /**
     * Return a child by its name.
     *
     * @param arg The name hierarchy of the child to look for.
     * @returns The found child or undefined.
     */
    getChildByName(arg: string | string[]): Reflection | undefined;
    /**
     * Return whether this reflection is the root / project reflection.
     */
    isProject(): this is ProjectReflection;
    isDeclaration(): this is DeclarationReflection;
    isParameter(): this is ParameterReflection;
    isDocument(): this is DocumentReflection;
    isReference(): this is ReferenceReflection;
    /**
     * Check if this reflection or any of its parents have been marked with the `@deprecated` tag.
     */
    isDeprecated(): boolean;
    /**
     * Traverse most potential child reflections of this reflection.
     *
     * Note: This may not necessarily traverse child reflections contained within the `type` property
     * of the reflection, and should not be relied on for this. Support for checking object types will likely be removed in v0.27.
     *
     * The given callback will be invoked for all children, signatures and type parameters
     * attached to this reflection.
     *
     * @param callback  The callback function that should be applied for each child reflection.
     */
    abstract traverse(callback: TraverseCallback): void;
    visit(visitor: ReflectionVisitor): void;
    /**
     * Return a string representation of this reflection.
     */
    toString(): string;
    /**
     * Return a string representation of this reflection and all of its children.
     *
     * Note: This is intended as a debug tool only, output may change between patch versions.
     *
     * @param indent  Used internally to indent child reflections.
     */
    toStringHierarchy(indent?: string): string;
    toObject(serializer: Serializer): JSONOutput.Reflection;
    fromObject(de: Deserializer, obj: JSONOutput.Reflection): void;
}
