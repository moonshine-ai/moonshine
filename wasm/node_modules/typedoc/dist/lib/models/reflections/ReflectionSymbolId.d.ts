import ts from "typescript";
import type { JSONOutput, Serializer } from "../../serialization/index.js";
import type { DeclarationReference } from "../../converter/index.js";
/**
 * See {@link ReflectionSymbolId}
 */
export type ReflectionSymbolIdString = string & {
    readonly __reflectionSymbolId: unique symbol;
};
/**
 * This exists so that TypeDoc can store a unique identifier for a `ts.Symbol` without
 * keeping a reference to the `ts.Symbol` itself. This identifier should be stable across
 * runs so long as the symbol is exported from the same file.
 */
export declare class ReflectionSymbolId {
    readonly fileName: string;
    readonly qualifiedName: string;
    /**
     * Note: This is **not** serialized. It exists for sorting by declaration order, but
     * should not be needed when deserializing from JSON.
     * Will be set to `Infinity` if the ID was deserialized from JSON.
     */
    pos: number;
    /**
     * Note: This is **not** serialized. It exists to support detection of the differences between
     * symbols which share declarations, but are instantiated with different type parameters.
     * This will be `NaN` if the symbol reference is not transient.
     * Note: This can only be non-NaN if {@link pos} is finite.
     */
    transientId: number;
    constructor(symbol: ts.Symbol, declaration?: ts.Declaration);
    constructor(json: JSONOutput.ReflectionSymbolId);
    getStableKey(): ReflectionSymbolIdString;
    toDeclarationReference(): DeclarationReference;
    toObject(serializer: Serializer): {
        sourceFileName: string;
        qualifiedName: string;
    };
}
export declare function addInferredDeclarationMapPaths(opts: ts.CompilerOptions, files: readonly string[]): void;
