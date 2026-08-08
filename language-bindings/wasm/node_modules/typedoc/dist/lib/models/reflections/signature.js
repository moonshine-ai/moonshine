import { ReflectionType } from "../types.js";
import { Reflection, TraverseProperty, } from "./abstract.js";
import { SourceReference } from "../sources/file.js";
/**
 * @category Reflections
 */
export class SignatureReflection extends Reflection {
    variant = "signature";
    // ESLint is wrong, we're restricting types to be more narrow.
    // eslint-disable-next-line @typescript-eslint/no-useless-constructor
    constructor(name, kind, parent) {
        super(name, kind, parent);
    }
    /**
     * A list of all source files that contributed to this reflection.
     */
    sources;
    parameters;
    typeParameters;
    type;
    /**
     * A type that points to the reflection that has been overwritten by this reflection.
     *
     * Applies to interface and class members.
     */
    overwrites;
    /**
     * A type that points to the reflection this reflection has been inherited from.
     *
     * Applies to interface and class members.
     */
    inheritedFrom;
    /**
     * A type that points to the reflection this reflection is the implementation of.
     *
     * Applies to class members.
     */
    implementationOf;
    traverse(callback) {
        if (this.type instanceof ReflectionType) {
            if (callback(this.type.declaration, TraverseProperty.TypeLiteral) === false) {
                return;
            }
        }
        for (const parameter of this.typeParameters?.slice() || []) {
            if (callback(parameter, TraverseProperty.TypeParameter) === false) {
                return;
            }
        }
        for (const parameter of this.parameters?.slice() || []) {
            if (callback(parameter, TraverseProperty.Parameters) === false) {
                return;
            }
        }
    }
    /**
     * Return a string representation of this reflection.
     */
    toString() {
        let result = super.toString();
        if (this.typeParameters) {
            const parameters = this.typeParameters.map((parameter) => parameter.name);
            result += "<" + parameters.join(", ") + ">";
        }
        if (this.type) {
            result += ": " + this.type.toString();
        }
        return result;
    }
    toObject(serializer) {
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            sources: serializer.toObjectsOptional(this.sources),
            typeParameters: serializer.toObjectsOptional(this.typeParameters),
            parameters: serializer.toObjectsOptional(this.parameters),
            type: serializer.toObject(this.type),
            overwrites: serializer.toObject(this.overwrites),
            inheritedFrom: serializer.toObject(this.inheritedFrom),
            implementationOf: serializer.toObject(this.implementationOf),
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        this.sources = de.reviveMany(obj.sources, (t) => new SourceReference(t.fileName, t.line, t.character));
        // eslint-disable-next-line @typescript-eslint/no-deprecated
        if (obj.typeParameter) {
            // eslint-disable-next-line @typescript-eslint/no-deprecated
            this.typeParameters = de.reviveMany(obj.typeParameter, (t) => de.constructReflection(t));
        }
        else {
            this.typeParameters = de.reviveMany(obj.typeParameters, (t) => de.constructReflection(t));
        }
        this.parameters = de.reviveMany(obj.parameters, (t) => de.constructReflection(t));
        this.type = de.reviveType(obj.type);
        this.overwrites = de.reviveType(obj.overwrites);
        this.inheritedFrom = de.reviveType(obj.inheritedFrom);
        this.implementationOf = de.reviveType(obj.implementationOf);
    }
}
