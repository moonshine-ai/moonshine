import ts from "typescript";
import { getQualifiedName } from "../utils/tsutils.js";
import { ReflectionSymbolId } from "./reflections/ReflectionSymbolId.js";
import { findPackageForPath } from "../utils/fs.js";
import { ReflectionKind } from "./reflections/kind.js";
import { Comment } from "./comments/index.js";
import { joinArray } from "../utils/array.js";
/**
 * Base class of all type definitions.
 * @category Types
 */
export class Type {
    /**
     * Return a string representation of this type.
     */
    toString() {
        return this.stringify(TypeContext.none);
    }
    visit(visitor, ...args) {
        return visitor[this.type]?.(this, ...args);
    }
    stringify(context) {
        if (this.needsParenthesis(context)) {
            return `(${this.getTypeString()})`;
        }
        return this.getTypeString();
    }
    // Nothing to do for the majority of types.
    fromObject(_de, _obj) { }
    /**
     * Return the estimated size of the type if it was all printed on one line.
     */
    estimatePrintWidth() {
        return this.getTypeString().length;
    }
}
export function makeRecursiveVisitor(visitor) {
    const recursiveVisitor = {
        namedTupleMember(type) {
            visitor.namedTupleMember?.(type);
            type.element.visit(recursiveVisitor);
        },
        templateLiteral(type) {
            visitor.templateLiteral?.(type);
            for (const [h] of type.tail) {
                h.visit(recursiveVisitor);
            }
        },
        array(type) {
            visitor.array?.(type);
            type.elementType.visit(recursiveVisitor);
        },
        conditional(type) {
            visitor.conditional?.(type);
            type.checkType.visit(recursiveVisitor);
            type.extendsType.visit(recursiveVisitor);
            type.trueType.visit(recursiveVisitor);
            type.falseType.visit(recursiveVisitor);
        },
        indexedAccess(type) {
            visitor.indexedAccess?.(type);
            type.indexType.visit(recursiveVisitor);
            type.objectType.visit(recursiveVisitor);
        },
        inferred(type) {
            visitor.inferred?.(type);
            type.constraint?.visit(recursiveVisitor);
        },
        intersection(type) {
            visitor.intersection?.(type);
            type.types.forEach((t) => t.visit(recursiveVisitor));
        },
        intrinsic(type) {
            visitor.intrinsic?.(type);
        },
        literal(type) {
            visitor.literal?.(type);
        },
        mapped(type) {
            visitor.mapped?.(type);
            type.nameType?.visit(recursiveVisitor);
            type.parameterType.visit(recursiveVisitor);
            type.templateType.visit(recursiveVisitor);
        },
        optional(type) {
            visitor.optional?.(type);
            type.elementType.visit(recursiveVisitor);
        },
        predicate(type) {
            visitor.predicate?.(type);
            type.targetType?.visit(recursiveVisitor);
        },
        query(type) {
            visitor.query?.(type);
            type.queryType.visit(recursiveVisitor);
        },
        reference(type) {
            visitor.reference?.(type);
            type.typeArguments?.forEach((t) => t.visit(recursiveVisitor));
        },
        reflection(type) {
            visitor.reflection?.(type);
            // Future: This should maybe recurse too?
            // See the validator in exports.ts for how to do it.
        },
        rest(type) {
            visitor.rest?.(type);
            type.elementType.visit(recursiveVisitor);
        },
        tuple(type) {
            visitor.tuple?.(type);
            type.elements.forEach((t) => t.visit(recursiveVisitor));
        },
        typeOperator(type) {
            visitor.typeOperator?.(type);
            type.target.visit(recursiveVisitor);
        },
        union(type) {
            visitor.union?.(type);
            type.types.forEach((t) => t.visit(recursiveVisitor));
        },
        unknown(type) {
            visitor.unknown?.(type);
        },
    };
    return recursiveVisitor;
}
/**
 * Enumeration that can be used when traversing types to track the location of recursion.
 * Used by TypeDoc internally to track when to output parenthesis when rendering.
 * @enum
 */
export const TypeContext = {
    none: "none",
    templateLiteralElement: "templateLiteralElement", // `${here}`
    arrayElement: "arrayElement", // here[]
    indexedAccessElement: "indexedAccessElement", // {}[here]
    conditionalCheck: "conditionalCheck", // here extends 1 ? 2 : 3
    conditionalExtends: "conditionalExtends", // 1 extends here ? 2 : 3
    conditionalTrue: "conditionalTrue", // 1 extends 2 ? here : 3
    conditionalFalse: "conditionalFalse", // 1 extends 2 ? 3 : here
    indexedIndex: "indexedIndex", // {}[here]
    indexedObject: "indexedObject", // here[1]
    inferredConstraint: "inferredConstraint", // 1 extends infer X extends here ? 1 : 2
    intersectionElement: "intersectionElement", // here & 1
    mappedName: "mappedName", // { [k in string as here]: 1 }
    mappedParameter: "mappedParameter", // { [k in here]: 1 }
    mappedTemplate: "mappedTemplate", // { [k in string]: here }
    optionalElement: "optionalElement", // [here?]
    predicateTarget: "predicateTarget", // (): X is here
    queryTypeTarget: "queryTypeTarget", // typeof here, can only ever be a ReferenceType
    typeOperatorTarget: "typeOperatorTarget", // keyof here
    referenceTypeArgument: "referenceTypeArgument", // X<here>
    restElement: "restElement", // [...here]
    tupleElement: "tupleElement", // [here]
    unionElement: "unionElement", // here | 1
};
/**
 * Represents an array type.
 *
 * ```ts
 * let value: string[];
 * ```
 * @category Types
 */
export class ArrayType extends Type {
    elementType;
    type = "array";
    /**
     * @param elementType The type of the elements in the array.
     */
    constructor(elementType) {
        super();
        this.elementType = elementType;
    }
    getTypeString() {
        return this.elementType.stringify(TypeContext.arrayElement) + "[]";
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            elementType: serializer.toObject(this.elementType),
        };
    }
}
/**
 * Represents a conditional type.
 *
 * ```ts
 * let value: Check extends Extends ? True : False;
 * ```
 * @category Types
 */
export class ConditionalType extends Type {
    checkType;
    extendsType;
    trueType;
    falseType;
    type = "conditional";
    constructor(checkType, extendsType, trueType, falseType) {
        super();
        this.checkType = checkType;
        this.extendsType = extendsType;
        this.trueType = trueType;
        this.falseType = falseType;
    }
    getTypeString() {
        return [
            this.checkType.stringify(TypeContext.conditionalCheck),
            "extends",
            this.extendsType.stringify(TypeContext.conditionalExtends),
            "?",
            this.trueType.stringify(TypeContext.conditionalTrue),
            ":",
            this.falseType.stringify(TypeContext.conditionalFalse),
        ].join(" ");
    }
    needsParenthesis(context) {
        const map = {
            none: false,
            templateLiteralElement: false,
            arrayElement: true,
            indexedAccessElement: false,
            conditionalCheck: true,
            conditionalExtends: true,
            conditionalTrue: false,
            conditionalFalse: false,
            indexedIndex: false,
            indexedObject: true,
            inferredConstraint: true,
            intersectionElement: true,
            mappedName: false,
            mappedParameter: false,
            mappedTemplate: false,
            optionalElement: true,
            predicateTarget: false,
            queryTypeTarget: false,
            typeOperatorTarget: true,
            referenceTypeArgument: false,
            restElement: true,
            tupleElement: false,
            unionElement: true,
        };
        return map[context];
    }
    toObject(serializer) {
        return {
            type: this.type,
            checkType: serializer.toObject(this.checkType),
            extendsType: serializer.toObject(this.extendsType),
            trueType: serializer.toObject(this.trueType),
            falseType: serializer.toObject(this.falseType),
        };
    }
}
/**
 * Represents an indexed access type.
 * @category Types
 */
export class IndexedAccessType extends Type {
    objectType;
    indexType;
    type = "indexedAccess";
    constructor(objectType, indexType) {
        super();
        this.objectType = objectType;
        this.indexType = indexType;
    }
    getTypeString() {
        return [
            this.objectType.stringify(TypeContext.indexedObject),
            "[",
            this.indexType.stringify(TypeContext.indexedIndex),
            "]",
        ].join("");
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            indexType: serializer.toObject(this.indexType),
            objectType: serializer.toObject(this.objectType),
        };
    }
}
/**
 * Represents an inferred type, U in the example below.
 *
 * ```ts
 * type Z = Promise<string> extends Promise<infer U> : never
 * ```
 * @category Types
 */
export class InferredType extends Type {
    name;
    constraint;
    type = "inferred";
    constructor(name, constraint) {
        super();
        this.name = name;
        this.constraint = constraint;
    }
    getTypeString() {
        if (this.constraint) {
            return `infer ${this.name} extends ${this.constraint.stringify(TypeContext.inferredConstraint)}`;
        }
        return `infer ${this.name}`;
    }
    needsParenthesis(context) {
        const map = {
            none: false,
            templateLiteralElement: false,
            arrayElement: true,
            indexedAccessElement: false,
            conditionalCheck: false,
            conditionalExtends: false,
            conditionalTrue: false,
            conditionalFalse: false,
            indexedIndex: false,
            indexedObject: true,
            inferredConstraint: false,
            intersectionElement: false,
            mappedName: false,
            mappedParameter: false,
            mappedTemplate: false,
            optionalElement: true,
            predicateTarget: false,
            queryTypeTarget: false,
            typeOperatorTarget: false,
            referenceTypeArgument: false,
            restElement: true,
            tupleElement: false,
            unionElement: false,
        };
        return map[context];
    }
    toObject(serializer) {
        return {
            type: this.type,
            name: this.name,
            constraint: serializer.toObject(this.constraint),
        };
    }
}
/**
 * Represents an intersection type.
 *
 * ```ts
 * let value: A & B;
 * ```
 * @category Types
 */
export class IntersectionType extends Type {
    types;
    type = "intersection";
    constructor(types) {
        super();
        this.types = types;
    }
    getTypeString() {
        return this.types
            .map((t) => t.stringify(TypeContext.intersectionElement))
            .join(" & ");
    }
    needsParenthesis(context) {
        const map = {
            none: false,
            templateLiteralElement: false,
            arrayElement: true,
            indexedAccessElement: false,
            conditionalCheck: true,
            conditionalExtends: false,
            conditionalTrue: false,
            conditionalFalse: false,
            indexedIndex: false,
            indexedObject: true,
            inferredConstraint: false,
            intersectionElement: false,
            mappedName: false,
            mappedParameter: false,
            mappedTemplate: false,
            optionalElement: true,
            predicateTarget: false,
            queryTypeTarget: false,
            typeOperatorTarget: true,
            referenceTypeArgument: false,
            restElement: true,
            tupleElement: false,
            unionElement: false,
        };
        return map[context];
    }
    toObject(serializer) {
        return {
            type: this.type,
            types: this.types.map((t) => serializer.toObject(t)),
        };
    }
}
/**
 * Represents an intrinsic type like `string` or `boolean`.
 *
 * ```ts
 * let value: number;
 * ```
 * @category Types
 */
export class IntrinsicType extends Type {
    name;
    type = "intrinsic";
    constructor(name) {
        super();
        this.name = name;
    }
    getTypeString() {
        return this.name;
    }
    toObject() {
        return {
            type: this.type,
            name: this.name,
        };
    }
    needsParenthesis() {
        return false;
    }
}
/**
 * Represents a literal type.
 *
 * ```ts
 * type A = "A"
 * type B = 1
 * ```
 * @category Types
 */
export class LiteralType extends Type {
    value;
    type = "literal";
    constructor(value) {
        super();
        this.value = value;
    }
    /**
     * Return a string representation of this type.
     */
    getTypeString() {
        if (typeof this.value === "bigint") {
            return this.value.toString() + "n";
        }
        return JSON.stringify(this.value);
    }
    needsParenthesis() {
        return false;
    }
    toObject() {
        if (typeof this.value === "bigint") {
            return {
                type: this.type,
                value: {
                    value: this.value.toString().replace("-", ""),
                    negative: this.value < BigInt("0"),
                },
            };
        }
        return {
            type: this.type,
            value: this.value,
        };
    }
}
/**
 * Represents a mapped type.
 *
 * ```ts
 * { -readonly [K in Parameter as Name]?: Template }
 * ```
 * @category Types
 */
export class MappedType extends Type {
    parameter;
    parameterType;
    templateType;
    readonlyModifier;
    optionalModifier;
    nameType;
    type = "mapped";
    constructor(parameter, parameterType, templateType, readonlyModifier, optionalModifier, nameType) {
        super();
        this.parameter = parameter;
        this.parameterType = parameterType;
        this.templateType = templateType;
        this.readonlyModifier = readonlyModifier;
        this.optionalModifier = optionalModifier;
        this.nameType = nameType;
    }
    getTypeString() {
        const read = {
            "+": "readonly ",
            "-": "-readonly ",
            "": "",
        }[this.readonlyModifier ?? ""];
        const opt = {
            "+": "?",
            "-": "-?",
            "": "",
        }[this.optionalModifier ?? ""];
        const parts = [
            "{ ",
            read,
            "[",
            this.parameter,
            " in ",
            this.parameterType.stringify(TypeContext.mappedParameter),
        ];
        if (this.nameType) {
            parts.push(" as ", this.nameType.stringify(TypeContext.mappedName));
        }
        parts.push("]", opt, ": ", this.templateType.stringify(TypeContext.mappedTemplate), " }");
        return parts.join("");
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            parameter: this.parameter,
            parameterType: serializer.toObject(this.parameterType),
            templateType: serializer.toObject(this.templateType),
            readonlyModifier: this.readonlyModifier,
            optionalModifier: this.optionalModifier,
            nameType: serializer.toObject(this.nameType),
        };
    }
}
/**
 * Represents an optional type
 * ```ts
 * type Z = [1, 2?]
 * //           ^^
 * ```
 * @category Types
 */
export class OptionalType extends Type {
    elementType;
    type = "optional";
    constructor(elementType) {
        super();
        this.elementType = elementType;
    }
    getTypeString() {
        return this.elementType.stringify(TypeContext.optionalElement) + "?";
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            elementType: serializer.toObject(this.elementType),
        };
    }
}
/**
 * Represents a type predicate.
 *
 * ```ts
 * function isString(x: unknown): x is string {}
 * function assert(condition: boolean): asserts condition {}
 * ```
 * @category Types
 */
export class PredicateType extends Type {
    name;
    asserts;
    targetType;
    type = "predicate";
    /**
     * Create a new PredicateType instance.
     *
     * @param name The identifier name which is tested by the predicate.
     * @param asserts True if the type is of the form `asserts val is string`,
     *                false if the type is of the form `val is string`
     * @param targetType The type that the identifier is tested to be.
     *                   May be undefined if the type is of the form `asserts val`.
     *                   Will be defined if the type is of the form `asserts val is string` or `val is string`.
     */
    constructor(name, asserts, targetType) {
        super();
        this.name = name;
        this.asserts = asserts;
        this.targetType = targetType;
    }
    /**
     * Return a string representation of this type.
     */
    getTypeString() {
        const out = this.asserts ? ["asserts", this.name] : [this.name];
        if (this.targetType) {
            out.push("is", this.targetType.stringify(TypeContext.predicateTarget));
        }
        return out.join(" ");
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            name: this.name,
            asserts: this.asserts,
            targetType: serializer.toObject(this.targetType),
        };
    }
}
/**
 * Represents a type that is constructed by querying the type of a reflection.
 * ```ts
 * const x = 1
 * type Z = typeof x // query on reflection for x
 * ```
 * @category Types
 */
export class QueryType extends Type {
    queryType;
    type = "query";
    constructor(queryType) {
        super();
        this.queryType = queryType;
    }
    getTypeString() {
        return `typeof ${this.queryType.stringify(TypeContext.queryTypeTarget)}`;
    }
    /**
     * @privateRemarks
     * An argument could be made that this ought to return true for indexedObject
     * since precedence is different than on the value side... if someone really cares
     * they can easily use a custom theme to change this.
     */
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            queryType: serializer.toObject(this.queryType),
        };
    }
}
/**
 * Represents a type that refers to another reflection like a class, interface or enum.
 *
 * ```ts
 * let value: MyClass<T>;
 * ```
 * @category Types
 */
export class ReferenceType extends Type {
    type = "reference";
    /**
     * The name of the referenced type.
     *
     * If the symbol cannot be found because it's not part of the documentation this
     * can be used to represent the type.
     */
    name;
    /**
     * The type arguments of this reference.
     */
    typeArguments;
    /**
     * The resolved reflection.
     */
    get reflection() {
        if (typeof this._target === "number") {
            return this._project?.getReflectionById(this._target);
        }
        const resolvePotential = this._project?.getReflectionsFromSymbolId(this._target);
        if (!resolvePotential?.length) {
            return;
        }
        const kind = this.preferValues
            ? ReflectionKind.ValueReferenceTarget
            : ReflectionKind.TypeReferenceTarget;
        const resolved = resolvePotential.find((refl) => refl.kindOf(kind)) ||
            resolvePotential.find((refl) => refl.kindOf(~kind));
        // Do not mark the type as resolved at this point so that if it
        // points to a member which is removed (e.g. by typedoc-plugin-zod)
        // and then replaced it still ends up pointing at the right reflection.
        // We will lock type reference resolution when serializing to JSON.
        // this._target = resolved.id;
        return resolved;
    }
    /**
     * Sometimes a few properties are more important than the rest
     * of the properties within a type. This occurs most often with
     * object parameters, where users want to specify `@param foo.bar`
     * to highlight something about the `bar` property.
     *
     * Does NOT support nested properties.
     */
    highlightedProperties;
    /**
     * If not resolved, the symbol id of the reflection, otherwise undefined.
     */
    get symbolId() {
        if (!this.reflection && typeof this._target === "object") {
            return this._target;
        }
    }
    /**
     * Checks if this type is a reference type because it uses a name, but is intentionally not pointing
     * to a reflection. This happens for type parameters and when representing a mapped type.
     */
    isIntentionallyBroken() {
        if (typeof this._target === "object" &&
            this._project?.symbolIdHasBeenRemoved(this._target)) {
            return true;
        }
        return this._target === -1 || this.refersToTypeParameter;
    }
    /**
     * Convert this reference type to a declaration reference used for resolution of external types.
     */
    toDeclarationReference() {
        return {
            resolutionStart: "global",
            moduleSource: this.package,
            symbolReference: {
                path: this.qualifiedName
                    .split(".")
                    .map((p) => ({ path: p, navigation: "." })),
            },
        };
    }
    /**
     * The fully qualified name of the referenced type, relative to the file it is defined in.
     * This will usually be the same as `name`, unless namespaces are used.
     */
    qualifiedName;
    /**
     * The package that this type is referencing.
     */
    package;
    /**
     * If this reference type refers to a reflection defined by a project not being rendered,
     * points to the url that this type should be linked to.
     */
    externalUrl;
    /**
     * If set, no warnings about something not being exported should be created
     * since this may be referring to a type created with `infer X` which will not
     * be registered on the project.
     */
    refersToTypeParameter = false;
    /**
     * If set, will prefer reflections with {@link ReflectionKind | ReflectionKinds} which represent
     * values rather than those which represent types.
     */
    preferValues = false;
    _target;
    _project;
    constructor(name, target, project, qualifiedName) {
        super();
        this.name = name;
        if (typeof target === "number") {
            this._target = target;
        }
        else {
            this._target = "variant" in target ? target.id : target;
        }
        this._project = project;
        this.qualifiedName = qualifiedName;
    }
    static createResolvedReference(name, target, project) {
        return new ReferenceType(name, target, project, name);
    }
    static createSymbolReference(symbol, context, name) {
        const ref = new ReferenceType(name ?? symbol.name, new ReflectionSymbolId(symbol), context.project, getQualifiedName(symbol, name ?? symbol.name));
        ref.refersToTypeParameter = !!(symbol.flags & ts.SymbolFlags.TypeParameter);
        const symbolPath = symbol.declarations?.[0]
            ?.getSourceFile()
            .fileName.replace(/\\/g, "/");
        if (!symbolPath)
            return ref;
        ref.package = findPackageForPath(symbolPath);
        return ref;
    }
    /**
     * This is used for type parameters, which don't actually point to something,
     * and also for temporary references which will be cleaned up with real references
     * later during conversion.
     * @internal
     */
    static createBrokenReference(name, project) {
        return new ReferenceType(name, -1, project, name);
    }
    getTypeString() {
        const name = this.reflection ? this.reflection.name : this.name;
        let typeArgs = "";
        if (this.typeArguments && this.typeArguments.length > 0) {
            typeArgs += "<";
            typeArgs += this.typeArguments
                .map((arg) => arg.stringify(TypeContext.referenceTypeArgument))
                .join(", ");
            typeArgs += ">";
        }
        return name + typeArgs;
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        let target;
        if (typeof this._target === "number") {
            target = this._target;
        }
        else if (this._project?.symbolIdHasBeenRemoved(this._target)) {
            target = -1;
        }
        else if (this.reflection) {
            target = this.reflection.id;
        }
        else {
            target = this._target.toObject(serializer);
        }
        const result = {
            type: this.type,
            target,
            typeArguments: serializer.toObjectsOptional(this.typeArguments),
            name: this.name,
            package: this.package,
            externalUrl: this.externalUrl,
        };
        if (this.name !== this.qualifiedName) {
            result.qualifiedName = this.qualifiedName;
        }
        if (this.refersToTypeParameter) {
            result.refersToTypeParameter = true;
        }
        if (typeof target !== "number" && this.preferValues) {
            result.preferValues = true;
        }
        if (this.highlightedProperties) {
            result.highlightedProperties = Object.fromEntries(Array.from(this.highlightedProperties.entries(), ([key, parts]) => {
                return [
                    key,
                    Comment.serializeDisplayParts(serializer, parts),
                ];
            }));
        }
        return result;
    }
    fromObject(de, obj) {
        this.typeArguments = de.reviveMany(obj.typeArguments, (t) => de.constructType(t));
        if (typeof obj.target === "number" && obj.target !== -1) {
            de.defer((project) => {
                const target = project.getReflectionById(de.oldIdToNewId[obj.target] ?? -1);
                if (target) {
                    this._project = project;
                    this._target = target.id;
                }
                else {
                    de.logger.warn(de.application.i18n.serialized_project_referenced_0_not_part_of_project(JSON.stringify(obj.target)));
                }
            });
        }
        else if (obj.target === -1) {
            this._target = -1;
        }
        else {
            this._project = de.project;
            this._target = new ReflectionSymbolId(obj.target);
        }
        this.qualifiedName = obj.qualifiedName ?? obj.name;
        this.package = obj.package;
        this.refersToTypeParameter = !!obj.refersToTypeParameter;
        this.preferValues = !!obj.preferValues;
        if (obj.highlightedProperties) {
            this.highlightedProperties = new Map();
            for (const [key, parts] of Object.entries(obj.highlightedProperties)) {
                this.highlightedProperties.set(key, Comment.deserializeDisplayParts(de, parts));
            }
        }
    }
}
/**
 * Represents a type which has it's own reflection like literal types.
 * This type will likely go away at some point and be replaced by a dedicated
 * `ObjectType`. Allowing reflections to be nested within types causes much
 * pain in the rendering code.
 *
 * ```ts
 * let value: { a: string, b: number };
 * ```
 * @category Types
 */
export class ReflectionType extends Type {
    declaration;
    type = "reflection";
    constructor(declaration) {
        super();
        this.declaration = declaration;
    }
    getTypeString() {
        const parts = [];
        const sigs = this.declaration.getAllSignatures();
        for (const sig of sigs) {
            parts.push(sigStr(sig, ": "));
        }
        if (this.declaration.children) {
            for (const p of this.declaration.children) {
                parts.push(`${p.name}${propertySep(p)} ${typeStr(p.type)}`);
            }
            return `{ ${parts.join("; ")} }`;
        }
        if (sigs.length === 1) {
            return sigStr(sigs[0], " => ");
        }
        if (parts.length === 0) {
            return "{}";
        }
        return `{ ${parts.join("; ")} }`;
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            declaration: serializer.toObject(this.declaration),
        };
    }
}
/**
 * Represents a rest type
 * ```ts
 * type Z = [1, ...2[]]
 * //           ^^^^^^
 * ```
 * @category Types
 */
export class RestType extends Type {
    elementType;
    type = "rest";
    constructor(elementType) {
        super();
        this.elementType = elementType;
    }
    getTypeString() {
        return `...${this.elementType.stringify(TypeContext.restElement)}`;
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            elementType: serializer.toObject(this.elementType),
        };
    }
}
/**
 * TS 4.1 template literal types
 * ```ts
 * type Z = `${'a' | 'b'}${'a' | 'b'}`
 * ```
 * @category Types
 */
export class TemplateLiteralType extends Type {
    head;
    tail;
    type = "templateLiteral";
    constructor(head, tail) {
        super();
        this.head = head;
        this.tail = tail;
    }
    getTypeString() {
        return [
            "`",
            this.head,
            ...this.tail.map(([type, text]) => {
                return ("${" +
                    type.stringify(TypeContext.templateLiteralElement) +
                    "}" +
                    text);
            }),
            "`",
        ].join("");
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            head: this.head,
            tail: this.tail.map(([type, text]) => [
                serializer.toObject(type),
                text,
            ]),
        };
    }
}
/**
 * Represents a tuple type.
 *
 * ```ts
 * let value: [string, boolean];
 * ```
 * @category Types
 */
export class TupleType extends Type {
    elements;
    type = "tuple";
    /**
     * @param elements The ordered type elements of the tuple type.
     */
    constructor(elements) {
        super();
        this.elements = elements;
    }
    getTypeString() {
        return ("[" +
            this.elements
                .map((t) => t.stringify(TypeContext.tupleElement))
                .join(", ") +
            "]");
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            elements: serializer.toObjectsOptional(this.elements),
        };
    }
}
/**
 * Represents a named member of a tuple type.
 *
 * ```ts
 * let value: [name: string];
 * ```
 * @category Types
 */
export class NamedTupleMember extends Type {
    name;
    isOptional;
    element;
    type = "namedTupleMember";
    constructor(name, isOptional, element) {
        super();
        this.name = name;
        this.isOptional = isOptional;
        this.element = element;
    }
    /**
     * Return a string representation of this type.
     */
    getTypeString() {
        return `${this.name}${this.isOptional ? "?" : ""}: ${this.element.stringify(TypeContext.tupleElement)}`;
    }
    needsParenthesis() {
        return false;
    }
    toObject(serializer) {
        return {
            type: this.type,
            name: this.name,
            isOptional: this.isOptional,
            element: serializer.toObject(this.element),
        };
    }
}
/**
 * Represents a type operator type.
 *
 * ```ts
 * class A {}
 * class B<T extends keyof A> {}
 * ```
 * @category Types
 */
export class TypeOperatorType extends Type {
    target;
    operator;
    type = "typeOperator";
    constructor(target, operator) {
        super();
        this.target = target;
        this.operator = operator;
    }
    getTypeString() {
        return `${this.operator} ${this.target.stringify(TypeContext.typeOperatorTarget)}`;
    }
    needsParenthesis(context) {
        const map = {
            none: false,
            templateLiteralElement: false,
            arrayElement: true,
            indexedAccessElement: false,
            conditionalCheck: false,
            conditionalExtends: false,
            conditionalTrue: false,
            conditionalFalse: false,
            indexedIndex: false,
            indexedObject: true,
            inferredConstraint: false,
            intersectionElement: false,
            mappedName: false,
            mappedParameter: false,
            mappedTemplate: false,
            optionalElement: true,
            predicateTarget: false,
            queryTypeTarget: false,
            typeOperatorTarget: false,
            referenceTypeArgument: false,
            restElement: false,
            tupleElement: false,
            unionElement: false,
        };
        return map[context];
    }
    toObject(serializer) {
        return {
            type: this.type,
            operator: this.operator,
            target: serializer.toObject(this.target),
        };
    }
}
/**
 * Represents an union type.
 *
 * ```ts
 * let value: string | string[];
 * ```
 * @category Types
 */
export class UnionType extends Type {
    types;
    type = "union";
    /**
     * If present, there should be as many items in this array as there are items in the {@link types} array.
     *
     * This member is only valid on unions which are on {@link DeclarationReflection.type | DeclarationReflection.type} with a
     * {@link ReflectionKind} `kind` of `TypeAlias`. Specifying it on any other union is undefined behavior.
     */
    elementSummaries;
    constructor(types) {
        super();
        this.types = types;
    }
    getTypeString() {
        return this.types
            .map((t) => t.stringify(TypeContext.unionElement))
            .join(" | ");
    }
    needsParenthesis(context) {
        const map = {
            none: false,
            templateLiteralElement: false,
            arrayElement: true,
            indexedAccessElement: false,
            conditionalCheck: true,
            conditionalExtends: false,
            conditionalTrue: false,
            conditionalFalse: false,
            indexedIndex: false,
            indexedObject: true,
            inferredConstraint: false,
            intersectionElement: true,
            mappedName: false,
            mappedParameter: false,
            mappedTemplate: false,
            optionalElement: true,
            predicateTarget: false,
            queryTypeTarget: false,
            typeOperatorTarget: true,
            referenceTypeArgument: false,
            restElement: false,
            tupleElement: false,
            unionElement: false,
        };
        return map[context];
    }
    fromObject(de, obj) {
        if (obj.elementSummaries) {
            this.elementSummaries = obj.elementSummaries.map((parts) => Comment.deserializeDisplayParts(de, parts));
        }
    }
    toObject(serializer) {
        return {
            type: this.type,
            types: this.types.map((t) => serializer.toObject(t)),
            elementSummaries: this.elementSummaries?.map((parts) => Comment.serializeDisplayParts(serializer, parts)),
        };
    }
}
/**
 * Represents all unknown types that cannot be converted by TypeDoc.
 * @category Types
 */
export class UnknownType extends Type {
    type = "unknown";
    /**
     * A string representation of the type as returned from TypeScript compiler.
     */
    name;
    constructor(name) {
        super();
        this.name = name;
    }
    getTypeString() {
        return this.name;
    }
    /**
     * Always returns true if not at the root level, we have no idea what's in here, so wrap it in parenthesis
     * to be extra safe.
     */
    needsParenthesis(context) {
        return context !== TypeContext.none;
    }
    toObject() {
        return {
            type: this.type,
            name: this.name,
        };
    }
}
function propertySep(refl) {
    return refl.flags.isOptional ? "?:" : ":";
}
function typeStr(type) {
    return type?.toString() ?? "any";
}
function sigStr(sig, sep) {
    const params = joinArray(sig.parameters, ", ", (p) => `${p.name}${propertySep(p)} ${typeStr(p.type)}`);
    return `(${params})${sep}${typeStr(sig.type)}`;
}
