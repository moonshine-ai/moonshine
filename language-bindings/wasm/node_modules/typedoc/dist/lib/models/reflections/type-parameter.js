import { Reflection } from "./abstract.js";
import { ReflectionKind } from "./kind.js";
/**
 * Modifier flags for type parameters, added in TS 4.7
 * @enum
 */
export const VarianceModifier = {
    in: "in",
    out: "out",
    inOut: "in out",
};
/**
 * @category Reflections
 */
export class TypeParameterReflection extends Reflection {
    variant = "typeParam";
    type;
    default;
    varianceModifier;
    constructor(name, parent, varianceModifier) {
        super(name, ReflectionKind.TypeParameter, parent);
        this.varianceModifier = varianceModifier;
    }
    toObject(serializer) {
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            type: serializer.toObject(this.type),
            default: serializer.toObject(this.default),
            varianceModifier: this.varianceModifier,
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        this.type = de.reviveType(obj.type);
        this.default = de.reviveType(obj.default);
        this.varianceModifier = obj.varianceModifier;
    }
    traverse(_callback) {
        // do nothing, no child reflections.
    }
}
