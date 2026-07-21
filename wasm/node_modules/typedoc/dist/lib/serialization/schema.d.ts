/**
 * Contains interfaces which describe the JSON output. Each interface is related to a specific type of serializer.
 *
 * ## Plugins
 * Plugins which modify the serialization process can use declaration merging
 * to add custom properties to the exported interfaces.
 * For example, if your custom serializer adds a property to all {@link Reflection} objects:
 * ```ts
 * declare module 'typedoc/dist/lib/serialization/schema' {
 *     export interface AbstractReflection {
 *         myCustomProp: boolean
 *     }
 * }
 * ```
 *
 * If a plugin defines a new Model type, {@link ModelToObject} will not pick up the serializer type and
 * the resulting type will not be included in the return type of {@link Serializer.toObject}.
 * To fix this, use declaration merging to augment the {@link Serializer} class.
 * ```ts
 * declare module 'typedoc/dist/lib/serialization/serializer' {
 *     export interface Serializer {
 *         toObject(value: CustomModel, obj?: Partial<CustomModel>): CustomOutput
 *     }
 * }
 * ```
 *
 * For documentation on the JSON output properties, view the corresponding model.
 * @summary Contains interfaces which describe the JSON output.
 * @module
 */
import type * as M from "../models/index.js";
import type { IfInternal } from "../utils/index.js";
/**
 * Describes the mapping from Model types to the corresponding JSON output type.
 */
export type ModelToObject<T> = [T] extends [Array<infer U>] ? ModelToObject<U>[] : [M.SomeType] extends [T] ? SomeType : _ModelToObject<T>;
type _ModelToObject<T> = T extends Primitive ? T : Required<T> extends Required<M.ReflectionGroup> ? ReflectionGroup : Required<T> extends Required<M.ReflectionCategory> ? ReflectionCategory : T extends M.ReflectionVariant[keyof M.ReflectionVariant] ? ReflectionVariantMap[T["variant"]] : T extends M.SomeType ? TypeKindMap[T["type"]] : T extends M.Type ? SomeType : T extends M.Comment ? Comment : T extends M.CommentTag ? CommentTag : T extends M.CommentDisplayPart ? CommentDisplayPart : T extends M.SourceReference ? SourceReference : T extends M.FileRegistry ? FileRegistry : never;
type Primitive = string | number | undefined | null | boolean;
type ToSerialized<T> = T extends Primitive ? T : T extends bigint ? {
    value: string;
    negative: boolean;
} : ModelToObject<T>;
/**
 * Helper to describe a set of serialized properties. Primitive types are returned
 * directly, while other models are first passed through ModelToObject.
 * This helper removes the readonly modifier from properties since the result of serialization
 * is a plain object that consumers may modify as they choose, TypeDoc doesn't care.
 */
type S<T, K extends keyof T> = {
    -readonly [K2 in K]: ToSerialized<T[K2]>;
};
export interface ReflectionSymbolId {
    sourceFileName: string;
    qualifiedName: string;
}
export interface ReflectionGroup extends S<M.ReflectionGroup, "title" | "description" | "categories"> {
    children?: M.ReflectionGroup["children"][number]["id"][];
}
export interface ReflectionCategory extends S<M.ReflectionCategory, "title" | "description"> {
    children?: M.ReflectionCategory["children"][number]["id"][];
}
/** @category Reflections */
export interface ReflectionVariantMap {
    declaration: DeclarationReflection;
    param: ParameterReflection;
    project: ProjectReflection;
    reference: ReferenceReflection;
    signature: SignatureReflection;
    typeParam: TypeParameterReflection;
    document: DocumentReflection;
}
/** @category Reflections */
export type SomeReflection = ReflectionVariantMap[keyof ReflectionVariantMap];
/** @category Reflections */
export interface DocumentReflection extends Omit<Reflection, "variant">, S<M.DocumentReflection, "variant" | "content" | "relevanceBoost" | "children"> {
    frontmatter: Record<string, unknown>;
}
/** @category Reflections */
export interface ReferenceReflection extends Omit<DeclarationReflection, "variant">, S<M.ReferenceReflection, "variant"> {
    /**
     * -1 if the reference refers to a symbol that does not exist in the documentation.
     * Otherwise, the reflection ID.
     */
    target: number;
}
/** @category Reflections */
export interface SignatureReflection extends Omit<Reflection, "variant">, S<M.SignatureReflection, "variant" | "sources" | "parameters" | "typeParameters" | "type" | "overwrites" | "inheritedFrom" | "implementationOf"> {
    /** @deprecated in 0.26, replaced with {@link typeParameters} */
    typeParameter?: ModelToObject<M.TypeParameterReflection[]>;
}
/** @category Reflections */
export interface ParameterReflection extends Omit<Reflection, "variant">, S<M.ParameterReflection, "variant" | "type" | "defaultValue"> {
    variant: "param";
}
/** @category Reflections */
export interface DeclarationReflection extends Omit<ContainerReflection, "variant">, S<M.DeclarationReflection, "variant" | "packageVersion" | "sources" | "relevanceBoost" | "type" | "signatures" | "indexSignatures" | "defaultValue" | "overwrites" | "inheritedFrom" | "implementationOf" | "extendedTypes" | "extendedBy" | "implementedTypes" | "implementedBy" | "getSignature" | "setSignature" | "typeParameters" | "readme"> {
    /** @deprecated moved to {@link indexSignatures} with 0.26. */
    indexSignature?: SignatureReflection;
}
/** @category Reflections */
export interface TypeParameterReflection extends Omit<Reflection, "variant">, S<M.TypeParameterReflection, "variant" | "type" | "default" | "varianceModifier"> {
}
/** @category Reflections */
export interface ProjectReflection extends Omit<ContainerReflection, "variant">, S<M.ProjectReflection, "variant" | "packageName" | "packageVersion" | "readme"> {
    symbolIdMap: Record<number, ReflectionSymbolId> | IfInternal<undefined, never>;
    files: FileRegistry;
}
/** @category Reflections */
export interface ContainerReflection extends Reflection, S<M.ContainerReflection, "children" | "documents" | "groups" | "categories"> {
    childrenIncludingDocuments?: number[];
}
/** @category Reflections */
export interface Reflection extends S<M.Reflection, "id" | "variant" | "name" | "kind" | "comment"> {
    flags: ReflectionFlags;
}
/** @category Types */
export type SomeType = TypeKindMap[keyof TypeKindMap];
/** @category Types */
export type TypeKindMap = {
    array: ArrayType;
    conditional: ConditionalType;
    indexedAccess: IndexedAccessType;
    inferred: InferredType;
    intersection: IntersectionType;
    intrinsic: IntrinsicType;
    literal: LiteralType;
    mapped: MappedType;
    optional: OptionalType;
    predicate: PredicateType;
    query: QueryType;
    reference: ReferenceType;
    reflection: ReflectionType;
    rest: RestType;
    templateLiteral: TemplateLiteralType;
    tuple: TupleType;
    namedTupleMember: NamedTupleMemberType;
    typeOperator: TypeOperatorType;
    union: UnionType;
    unknown: UnknownType;
};
/** @category Types */
export interface ArrayType extends Type, S<M.ArrayType, "type" | "elementType"> {
}
/** @category Types */
export interface ConditionalType extends Type, S<M.ConditionalType, "type" | "checkType" | "extendsType" | "trueType" | "falseType"> {
}
/** @category Types */
export interface IndexedAccessType extends Type, S<M.IndexedAccessType, "type" | "indexType" | "objectType"> {
}
/** @category Types */
export interface InferredType extends Type, S<M.InferredType, "type" | "name" | "constraint"> {
}
/** @category Types */
export interface IntersectionType extends Type, S<M.IntersectionType, "type" | "types"> {
}
/** @category Types */
export interface IntrinsicType extends Type, S<M.IntrinsicType, "type" | "name"> {
}
/** @category Types */
export interface OptionalType extends Type, S<M.OptionalType, "type" | "elementType"> {
}
/** @category Types */
export interface PredicateType extends Type, S<M.PredicateType, "type" | "name" | "asserts" | "targetType"> {
}
/** @category Types */
export interface QueryType extends Type, S<M.QueryType, "type" | "queryType"> {
}
/** @category Types */
export interface ReferenceType extends Type, S<M.ReferenceType, "type" | "name" | "typeArguments" | "package" | "externalUrl"> {
    target: number | ReflectionSymbolId;
    qualifiedName?: string;
    refersToTypeParameter?: boolean;
    preferValues?: boolean;
    highlightedProperties?: Record<string, CommentDisplayPart[]>;
}
/** @category Types */
export interface ReflectionType extends Type, S<M.ReflectionType, "type" | "declaration"> {
}
/** @category Types */
export interface RestType extends Type, S<M.RestType, "type" | "elementType"> {
}
/** @category Types */
export interface LiteralType extends Type, S<M.LiteralType, "type" | "value"> {
}
/** @category Types */
export interface TupleType extends Type, S<M.TupleType, "type"> {
    elements?: ModelToObject<M.TupleType["elements"]>;
}
/** @category Types */
export interface NamedTupleMemberType extends Type, S<M.NamedTupleMember, "type" | "name" | "isOptional" | "element"> {
}
/** @category Types */
export interface TemplateLiteralType extends Type, S<M.TemplateLiteralType, "type" | "head"> {
    tail: [SomeType, string][];
}
/** @category Types */
export interface MappedType extends Type, S<M.MappedType, "type" | "parameter" | "parameterType" | "templateType" | "readonlyModifier" | "optionalModifier" | "nameType"> {
}
/** @category Types */
export interface TypeOperatorType extends Type, S<M.TypeOperatorType, "type" | "operator" | "target"> {
}
/** @category Types */
export interface UnionType extends Type, S<M.UnionType, "type" | "types" | "elementSummaries"> {
}
/** @category Types */
export interface UnknownType extends Type, S<M.UnknownType, "type" | "name"> {
}
/** @category Types */
export interface Type {
}
type BoolKeys<T> = {
    [K in keyof T]-?: T[K] extends boolean ? K : never;
}[keyof T];
export interface ReflectionFlags extends Partial<S<M.ReflectionFlags, BoolKeys<M.ReflectionFlags>>> {
}
/** @category Comments */
export interface Comment extends Partial<S<M.Comment, "blockTags" | "label">> {
    summary: CommentDisplayPart[];
    modifierTags?: `@${string}`[];
}
/** @category Comments */
export interface CommentTag extends S<M.CommentTag, "tag" | "name"> {
    content: CommentDisplayPart[];
}
/**
 * @see {@link M.CommentDisplayPart}
 * @category Comments
 */
export type CommentDisplayPart = {
    kind: "text";
    text: string;
} | {
    kind: "code";
    text: string;
} | InlineTagDisplayPart | RelativeLinkDisplayPart;
/**
 * If `target` is a number, it is a reflection ID. If a string, it is a URL.
 * `target` will only be set for `@link`, `@linkcode`, and `@linkplain` tags.
 * @category Comments
 */
export interface InlineTagDisplayPart {
    kind: "inline-tag";
    tag: `@${string}`;
    text: string;
    target?: string | number | ReflectionSymbolId;
    tsLinkText?: string;
}
/**
 * This is used for relative links within comments/documents.
 * It is used to mark pieces of text which need to be replaced
 * to make links work properly.
 */
export interface RelativeLinkDisplayPart {
    kind: "relative-link";
    /**
     * The original relative text from the parsed comment.
     */
    text: string;
    /**
     * File ID, if present
     */
    target?: number;
    /**
     * Anchor within the target file, if present
     */
    targetAnchor?: string;
}
export interface SourceReference extends S<M.SourceReference, "fileName" | "line" | "character" | "url"> {
}
export interface FileRegistry {
    /** Relative path according to the serialization root */
    entries: Record<number, string>;
    /** File ID to reflection ID */
    reflections: Record<number, number>;
}
export {};
