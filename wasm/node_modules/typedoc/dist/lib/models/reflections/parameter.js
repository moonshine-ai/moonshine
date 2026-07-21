import { ReflectionType } from "../types.js";
import { Reflection, TraverseProperty, } from "./abstract.js";
/**
 * @category Reflections
 */
export class ParameterReflection extends Reflection {
    variant = "param";
    defaultValue;
    type;
    traverse(callback) {
        if (this.type instanceof ReflectionType) {
            if (callback(this.type.declaration, TraverseProperty.TypeLiteral) === false) {
                return;
            }
        }
    }
    isParameter() {
        return true;
    }
    /**
     * Return a string representation of this reflection.
     */
    toString() {
        return (super.toString() + (this.type ? ": " + this.type.toString() : ""));
    }
    toObject(serializer) {
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            type: serializer.toObject(this.type),
            defaultValue: this.defaultValue,
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        this.type = de.reviveType(obj.type);
        this.defaultValue = obj.defaultValue;
    }
}
