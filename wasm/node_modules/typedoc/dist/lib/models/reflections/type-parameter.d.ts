import type { SomeType } from "../types.js";
import { Reflection, type TraverseCallback } from "./abstract.js";
import type { DeclarationReflection } from "./declaration.js";
import type { Serializer, JSONOutput, Deserializer } from "../../serialization/index.js";
import type { SignatureReflection } from "./signature.js";
/**
 * Modifier flags for type parameters, added in TS 4.7
 * @enum
 */
export declare const VarianceModifier: {
    readonly in: "in";
    readonly out: "out";
    readonly inOut: "in out";
};
export type VarianceModifier = (typeof VarianceModifier)[keyof typeof VarianceModifier];
/**
 * @category Reflections
 */
export declare class TypeParameterReflection extends Reflection {
    readonly variant = "typeParam";
    parent?: DeclarationReflection | SignatureReflection;
    type?: SomeType;
    default?: SomeType;
    varianceModifier?: VarianceModifier;
    constructor(name: string, parent: Reflection, varianceModifier: VarianceModifier | undefined);
    toObject(serializer: Serializer): JSONOutput.TypeParameterReflection;
    fromObject(de: Deserializer, obj: JSONOutput.TypeParameterReflection): void;
    traverse(_callback: TraverseCallback): void;
}
