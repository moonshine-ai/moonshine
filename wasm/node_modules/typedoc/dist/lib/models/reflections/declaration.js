import { ReflectionType } from "../types.js";
import { TraverseProperty } from "./abstract.js";
import { ContainerReflection } from "./container.js";
import { Comment } from "../comments/index.js";
import { SourceReference } from "../sources/file.js";
import { ReflectionSymbolId } from "./ReflectionSymbolId.js";
import { ReflectionKind } from "./kind.js";
/**
 * A reflection that represents a single declaration emitted by the TypeScript compiler.
 *
 * All parts of a project are represented by DeclarationReflection instances. The actual
 * kind of a reflection is stored in its ´kind´ member.
 * @category Reflections
 */
export class DeclarationReflection extends ContainerReflection {
    variant = "declaration";
    /**
     * A list of all source files that contributed to this reflection.
     */
    sources;
    /**
     * Precomputed boost for search results, may be less than 1 to de-emphasize this member in search results.
     * Does NOT include group/category values as they are computed when building the JS index.
     *
     * This is preserved for plugins, and may be removed in 0.28 if no plugins have used it yet.
     */
    relevanceBoost;
    /**
     * The escaped name of this declaration assigned by the TS compiler if there is an associated symbol.
     * This is used to retrieve properties for analyzing inherited members.
     *
     * Not serialized, only useful during conversion.
     * @internal
     */
    escapedName;
    /**
     * The type of the reflection.
     *
     * If the reflection represents a variable or a property, this is the value type.<br />
     * If the reflection represents a signature, this is the return type.
     */
    type;
    typeParameters;
    /**
     * A list of call signatures attached to this declaration.
     *
     * TypeDoc creates one declaration per function that may contain one or more
     * signature reflections.
     */
    signatures;
    /**
     * The index signature of this declaration.
     */
    indexSignatures;
    /**
     * The get signature of this declaration.
     */
    getSignature;
    /**
     * The set signature of this declaration.
     */
    setSignature;
    /**
     * The default value of this reflection.
     *
     * Applies to function parameters, variables, and properties.
     */
    defaultValue;
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
    /**
     * A list of all types this reflection extends (e.g. the parent classes).
     */
    extendedTypes;
    /**
     * A list of all types that extend this reflection (e.g. the subclasses).
     */
    extendedBy;
    /**
     * A list of all types this reflection implements.
     */
    implementedTypes;
    /**
     * A list of all types that implement this reflection.
     */
    implementedBy;
    /**
     * Contains a simplified representation of the type hierarchy suitable for being
     * rendered in templates.
     */
    typeHierarchy;
    /**
     * The contents of the readme file of the module when found.
     */
    readme;
    /**
     * The version of the module when found.
     */
    packageVersion;
    isDeclaration() {
        return true;
    }
    hasGetterOrSetter() {
        return !!this.getSignature || !!this.setSignature;
    }
    getAllSignatures() {
        let result = [];
        if (this.signatures) {
            result = result.concat(this.signatures);
        }
        if (this.indexSignatures) {
            result = result.concat(this.indexSignatures);
        }
        if (this.getSignature) {
            result.push(this.getSignature);
        }
        if (this.setSignature) {
            result.push(this.setSignature);
        }
        return result;
    }
    getNonIndexSignatures() {
        return [].concat(this.signatures ?? [], this.setSignature ?? [], this.getSignature ?? []);
    }
    getProperties() {
        if (this.children?.length) {
            return this.children;
        }
        if (this.type?.type === "reflection") {
            return this.type.declaration.children ?? [];
        }
        return [];
    }
    getChildOrTypePropertyByName(path) {
        if (this.type?.type === "reflection") {
            for (const child of this.type.declaration.children || []) {
                if (path[0] === child.name) {
                    if (path.length === 1) {
                        return child;
                    }
                    return child.getChildOrTypePropertyByName(path.slice(1));
                }
            }
        }
        for (const child of this.children || []) {
            if (path[0] === child.name) {
                if (path.length === 1) {
                    return child;
                }
                return child.getChildOrTypePropertyByName(path.slice(1));
            }
        }
        return undefined;
    }
    traverse(callback) {
        for (const parameter of this.typeParameters?.slice() || []) {
            if (callback(parameter, TraverseProperty.TypeParameter) === false) {
                return;
            }
        }
        if (this.type instanceof ReflectionType) {
            if (callback(this.type.declaration, TraverseProperty.TypeLiteral) === false) {
                return;
            }
        }
        for (const signature of this.signatures?.slice() || []) {
            if (callback(signature, TraverseProperty.Signatures) === false) {
                return;
            }
        }
        for (const signature of this.indexSignatures?.slice() || []) {
            if (callback(signature, TraverseProperty.IndexSignature) === false) {
                return;
            }
        }
        if (this.getSignature) {
            if (callback(this.getSignature, TraverseProperty.GetSignature) ===
                false) {
                return;
            }
        }
        if (this.setSignature) {
            if (callback(this.setSignature, TraverseProperty.SetSignature) ===
                false) {
                return;
            }
        }
        super.traverse(callback);
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
            packageVersion: this.packageVersion,
            sources: serializer.toObjectsOptional(this.sources),
            relevanceBoost: this.relevanceBoost === 1 ? undefined : this.relevanceBoost,
            typeParameters: serializer.toObjectsOptional(this.typeParameters),
            type: serializer.toObject(this.type),
            signatures: serializer.toObjectsOptional(this.signatures),
            indexSignatures: serializer.toObjectsOptional(this.indexSignatures),
            getSignature: serializer.toObject(this.getSignature),
            setSignature: serializer.toObject(this.setSignature),
            defaultValue: this.defaultValue,
            overwrites: serializer.toObject(this.overwrites),
            inheritedFrom: serializer.toObject(this.inheritedFrom),
            implementationOf: serializer.toObject(this.implementationOf),
            extendedTypes: serializer.toObjectsOptional(this.extendedTypes),
            extendedBy: serializer.toObjectsOptional(this.extendedBy),
            implementedTypes: serializer.toObjectsOptional(this.implementedTypes),
            implementedBy: serializer.toObjectsOptional(this.implementedBy),
            readme: Comment.serializeDisplayParts(serializer, this.readme),
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        if (obj.readme) {
            this.readme = Comment.deserializeDisplayParts(de, obj.readme);
        }
        // This happens when merging multiple projects together.
        // If updating this, also check ProjectReflection.fromObject.
        if (obj.variant === "project") {
            this.kind = ReflectionKind.Module;
            this.packageVersion = obj.packageVersion;
            this.project.files.fromObject(de, obj.files || {});
            de.defer(() => {
                for (const [id, sid] of Object.entries(obj.symbolIdMap || {})) {
                    const refl = this.project.getReflectionById(de.oldIdToNewId[+id] ?? -1);
                    if (refl) {
                        this.project.registerSymbolId(refl, new ReflectionSymbolId(sid));
                    }
                    else {
                        de.logger.warn(de.application.i18n.serialized_project_referenced_0_not_part_of_project(id.toString()));
                    }
                }
            });
            return;
        }
        this.packageVersion = obj.packageVersion;
        this.sources = de.reviveMany(obj.sources, (src) => new SourceReference(src.fileName, src.line, src.character));
        this.relevanceBoost = obj.relevanceBoost;
        this.typeParameters = de.reviveMany(obj.typeParameters, (tp) => de.constructReflection(tp));
        this.type = de.revive(obj.type, (t) => de.constructType(t));
        this.signatures = de.reviveMany(obj.signatures, (r) => de.constructReflection(r));
        // TypeDoc 0.25, remove check with 0.28.
        // eslint-disable-next-line @typescript-eslint/no-deprecated
        if (obj.indexSignature) {
            this.indexSignatures = [
                // eslint-disable-next-line @typescript-eslint/no-deprecated
                de.revive(obj.indexSignature, (r) => de.constructReflection(r)),
            ];
        }
        else {
            this.indexSignatures = de.reviveMany(obj.indexSignatures, (r) => de.constructReflection(r));
        }
        this.getSignature = de.revive(obj.getSignature, (r) => de.constructReflection(r));
        this.setSignature = de.revive(obj.setSignature, (r) => de.constructReflection(r));
        this.defaultValue = obj.defaultValue;
        this.overwrites = de.reviveType(obj.overwrites);
        this.inheritedFrom = de.reviveType(obj.inheritedFrom);
        this.implementationOf = de.reviveType(obj.implementationOf);
        this.extendedTypes = de.reviveMany(obj.extendedTypes, (t) => de.reviveType(t));
        this.extendedBy = de.reviveMany(obj.extendedBy, (t) => de.reviveType(t));
        this.implementedTypes = de.reviveMany(obj.implementedTypes, (t) => de.reviveType(t));
        this.implementedBy = de.reviveMany(obj.implementedBy, (t) => de.reviveType(t));
    }
}
