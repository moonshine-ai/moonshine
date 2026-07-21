import type { EnumKeys } from "../../utils/index.js";
/**
 * Defines the available reflection kinds.
 * @category Reflections
 */
export declare enum ReflectionKind {
    Project = 1,
    Module = 2,
    Namespace = 4,
    Enum = 8,
    EnumMember = 16,
    Variable = 32,
    Function = 64,
    Class = 128,
    Interface = 256,
    Constructor = 512,
    Property = 1024,
    Method = 2048,
    CallSignature = 4096,
    IndexSignature = 8192,
    ConstructorSignature = 16384,
    Parameter = 32768,
    TypeLiteral = 65536,
    TypeParameter = 131072,
    Accessor = 262144,
    GetSignature = 524288,
    SetSignature = 1048576,
    TypeAlias = 2097152,
    Reference = 4194304,
    /**
     * Generic non-ts content to be included in the generated docs as its own page.
     */
    Document = 8388608
}
/** @category Reflections */
export declare namespace ReflectionKind {
    type KindString = EnumKeys<typeof ReflectionKind>;
    /** @internal */
    const All: number;
    /** @internal */
    const ClassOrInterface: number;
    /** @internal */
    const VariableOrProperty: number;
    /** @internal */
    const FunctionOrMethod: number;
    /** @internal */
    const ClassMember: number;
    /** @internal */
    const SomeSignature: number;
    /** @internal */
    const SomeModule: number;
    /** @internal */
    const SomeType: number;
    /** @internal */
    const SomeValue: number;
    /** @internal */
    const SomeMember: number;
    /** @internal */
    const SomeExport: number;
    /** @internal */
    const MayContainDocuments: number;
    /** @internal */
    const ExportContainer: number;
    /** @internal */
    const Inheritable: number;
    /** @internal */
    const ContainsCallSignatures: number;
    /** @internal */
    const TypeReferenceTarget: number;
    /** @internal */
    const ValueReferenceTarget: number;
    /**
     * Note: This does not include Class/Interface, even though they technically could contain index signatures
     * @internal
     */
    const SignatureContainer: number;
    /** @internal */
    const VariableContainer: number;
    /** @internal */
    const MethodContainer: number;
    /**
     * Get a non-localized kind string. For the localized string, use `app.internationalization.kindSingularString(kind)`
     */
    function singularString(kind: ReflectionKind): string;
    /**
     * Get a non-localized kind string. For the localized string, use `app.internationalization.kindPluralString(kind)`
     */
    function pluralString(kind: ReflectionKind): string;
    function classString(kind: ReflectionKind): string;
}
