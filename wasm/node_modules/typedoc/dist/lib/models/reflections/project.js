import { TraverseProperty } from "./abstract.js";
import { ContainerReflection } from "./container.js";
import { ReferenceReflection } from "./reference.js";
import { IntrinsicType, makeRecursiveVisitor } from "../types.js";
import { assertNever, removeIf, removeIfPresent } from "../../utils/index.js";
import { ReflectionKind } from "./kind.js";
import { Comment } from "../comments/index.js";
import { ReflectionSymbolId } from "./ReflectionSymbolId.js";
import { DefaultMap, StableKeyMap } from "../../utils/map.js";
/**
 * A reflection that represents the root of the project.
 *
 * The project reflection acts as a global index, one may receive all reflections
 * and source files of the processed project through this reflection.
 * @category Reflections
 */
export class ProjectReflection extends ContainerReflection {
    variant = "project";
    // Used to resolve references.
    symbolToReflectionIdMap = new StableKeyMap();
    reflectionIdToSymbolIdMap = new Map();
    reflectionIdToSymbolMap = new Map();
    removedSymbolIds = new StableKeyMap();
    // Maps a reflection ID to all references eventually referring to it.
    referenceGraph;
    // Maps a reflection ID to all reflections with it as their parent.
    reflectionChildren = new DefaultMap(() => []);
    /**
     * A list of all reflections within the project. DO NOT MUTATE THIS OBJECT.
     * All mutation should be done via {@link registerReflection} and {@link removeReflection}
     * to ensure that links to reflections remain valid.
     *
     * This may be replaced with a `Map<number, Reflection>` someday.
     */
    reflections = { [this.id]: this };
    /**
     * The name of the package that this reflection documents according to package.json.
     */
    packageName;
    /**
     * The version of the package that this reflection documents according to package.json.
     */
    packageVersion;
    /**
     * The contents of the readme.md file of the project when found.
     */
    readme;
    /**
     * Object which describes where to find content for relative links.
     */
    files;
    constructor(name, registry) {
        super(name, ReflectionKind.Project);
        this.reflections[this.id] = this;
        this.files = registry;
    }
    /**
     * Return whether this reflection is the root / project reflection.
     */
    isProject() {
        return true;
    }
    /**
     * Return a list of all reflections in this project of a certain kind.
     *
     * @param kind  The desired kind of reflection.
     * @returns     An array containing all reflections with the desired kind.
     */
    getReflectionsByKind(kind) {
        return Object.values(this.reflections).filter((reflection) => reflection.kindOf(kind));
    }
    /**
     * Registers the given reflection so that it can be quickly looked up by helper methods.
     * Should be called for *every* reflection added to the project.
     */
    registerReflection(reflection, symbol, filePath) {
        this.referenceGraph = undefined;
        if (reflection.parent) {
            this.reflectionChildren
                .get(reflection.parent.id)
                .push(reflection.id);
        }
        this.reflections[reflection.id] = reflection;
        if (symbol) {
            this.reflectionIdToSymbolMap.set(reflection.id, symbol);
            const id = new ReflectionSymbolId(symbol);
            this.registerSymbolId(reflection, id);
            // #2466
            // If we just registered a member of a class or interface, then we need to check if
            // we've registered this symbol before under the wrong parent reflection.
            // This can happen because the compiler API will use non-dependently-typed symbols
            // for properties of classes/interfaces which inherit them, so we can't rely on the
            // property being unique for each class.
            if (reflection.parent?.kindOf(ReflectionKind.ClassOrInterface) &&
                reflection.kindOf(ReflectionKind.SomeMember)) {
                const saved = this.symbolToReflectionIdMap.get(id);
                const parentSymbolReflection = symbol.parent &&
                    this.getReflectionFromSymbol(symbol.parent);
                if (typeof saved === "object" &&
                    saved.length > 1 &&
                    parentSymbolReflection) {
                    removeIf(saved, (item) => this.getReflectionById(item)?.parent !==
                        parentSymbolReflection);
                }
            }
        }
        if (filePath) {
            this.files.registerReflection(filePath, reflection);
        }
    }
    /**
     * Removes references to reflections contained within the provided type.
     * Plugins which overwrite types on reflections should pass the type to this
     * method before overwriting the property.
     * @since 0.26.6
     */
    removeTypeReflections(type) {
        type?.visit(makeRecursiveVisitor({
            reflection: (type) => {
                this.removeReflection(type.declaration);
            },
        }));
    }
    /**
     * Removes a reflection from the documentation. Can be used by plugins to filter reflections
     * out of the generated documentation. Has no effect if the reflection is not present in the
     * project.
     */
    removeReflection(reflection) {
        // Remove the reflection...
        this._removeReflection(reflection);
        // And now try to remove references to it in the parent reflection.
        // This might not find anything if someone called removeReflection on a member of a union
        // but I think that could only be caused by a plugin doing something weird, not by a regular
        // user... so this is probably good enough for now. Reflections that live on types are
        // kind of half-real anyways.
        const parent = reflection.parent;
        parent?.traverse((child, property) => {
            if (child !== reflection) {
                return true; // Continue iteration
            }
            switch (property) {
                case TraverseProperty.Children:
                case TraverseProperty.Documents:
                    parent.removeChild(reflection);
                    break;
                case TraverseProperty.GetSignature:
                    delete parent.getSignature;
                    break;
                case TraverseProperty.IndexSignature:
                    removeIfPresent(parent.indexSignatures, reflection);
                    if (!parent.indexSignatures?.length) {
                        delete parent.indexSignatures;
                    }
                    break;
                case TraverseProperty.Parameters:
                    removeIfPresent(reflection.parent.parameters, reflection);
                    if (!reflection.parent.parameters
                        ?.length) {
                        delete reflection.parent
                            .parameters;
                    }
                    break;
                case TraverseProperty.SetSignature:
                    delete parent.setSignature;
                    break;
                case TraverseProperty.Signatures:
                    removeIfPresent(parent.signatures, reflection);
                    if (!parent.signatures?.length) {
                        delete parent.signatures;
                    }
                    break;
                case TraverseProperty.TypeLiteral:
                    parent.type = new IntrinsicType("Object");
                    break;
                case TraverseProperty.TypeParameter:
                    removeIfPresent(parent.typeParameters, reflection);
                    if (!parent.typeParameters?.length) {
                        delete parent.typeParameters;
                    }
                    break;
                default:
                    assertNever(property);
            }
            return false; // Stop iteration
        });
    }
    /** @internal */
    mergeModules(source, target) {
        // First, tell the children about their new parent
        delete this.referenceGraph;
        const oldChildrenIds = this.reflectionChildren.getNoInsert(source.id) || [];
        const newChildren = this.reflectionChildren.get(target.id);
        for (const childId of oldChildrenIds) {
            const childRefl = this.getReflectionById(childId);
            // To avoid conflicting with some plugins which do this surgery somewhat incorrectly
            // (typedoc-plugin-merge-modules and likely others I'm not aware of) only move children
            // which are still children
            if (childRefl?.parent === source) {
                childRefl.parent = target;
                newChildren.push(childId);
                target.addChild(childRefl);
            }
        }
        // Then remove the now-empty parent
        this.reflectionChildren.delete(source.id);
        this.removeReflection(source);
        // And remove any outdated collections of children on the new parent.
        // So long as this is used before REVIVE(-100) or EVENT_BEGIN_RESOLVE(-100)
        // this will make the appropriate plugin rebuild the lists.
        delete target.groups;
        delete target.categories;
    }
    /**
     * Remove a reflection without updating the parent reflection to remove references to the removed reflection.
     */
    _removeReflection(reflection) {
        this.files.removeReflection(reflection);
        // Remove references pointing to this reflection
        const graph = this.getReferenceGraph();
        for (const id of graph.get(reflection.id) ?? []) {
            const ref = this.getReflectionById(id);
            if (ref) {
                this.removeReflection(ref);
            }
        }
        graph.delete(reflection.id);
        // Remove children of this reflection
        for (const childId of this.reflectionChildren.getNoInsert(reflection.id) || []) {
            const child = this.getReflectionById(childId);
            // Only remove if the child's parent is still actually this reflection.
            // This might not be the case if a plugin has moved this reflection to another parent.
            // (typedoc-plugin-merge-modules)
            if (child?.parent === reflection) {
                this._removeReflection(child);
            }
        }
        this.reflectionChildren.delete(reflection.id);
        // Remove references from the TS symbol to this reflection.
        const symbolId = this.reflectionIdToSymbolIdMap.get(reflection.id);
        if (symbolId) {
            const saved = this.symbolToReflectionIdMap.get(symbolId);
            if (saved === reflection.id) {
                this.symbolToReflectionIdMap.delete(symbolId);
                this.removedSymbolIds.set(symbolId, true);
            }
            else if (typeof saved === "object") {
                removeIfPresent(saved, reflection.id);
                if (saved.length === 0) {
                    this.removedSymbolIds.set(symbolId, true);
                }
            }
        }
        this.reflectionIdToSymbolMap.delete(reflection.id);
        this.reflectionIdToSymbolIdMap.delete(reflection.id);
        delete this.reflections[reflection.id];
    }
    /**
     * Gets the reflection registered for the given reflection ID, or undefined if it is not present
     * in the project.
     */
    getReflectionById(id) {
        return this.reflections[id];
    }
    /**
     * Gets the reflection associated with the given symbol, if it exists.
     * @internal
     */
    getReflectionFromSymbol(symbol) {
        return this.getReflectionFromSymbolId(new ReflectionSymbolId(symbol));
    }
    /**
     * Gets the reflection associated with the given symbol id, if it exists.
     * If there are multiple reflections associated with this symbol, gets the first one.
     * @internal
     */
    getReflectionFromSymbolId(symbolId) {
        return this.getReflectionsFromSymbolId(symbolId)[0];
    }
    /** @internal */
    getReflectionsFromSymbolId(symbolId) {
        const id = this.symbolToReflectionIdMap.get(symbolId);
        if (typeof id === "number") {
            return [this.getReflectionById(id)];
        }
        else if (typeof id === "object") {
            return id.map((id) => this.getReflectionById(id));
        }
        return [];
    }
    /** @internal */
    getSymbolIdFromReflection(reflection) {
        return this.reflectionIdToSymbolIdMap.get(reflection.id);
    }
    /** @internal */
    registerSymbolId(reflection, id) {
        this.removedSymbolIds.delete(id);
        this.reflectionIdToSymbolIdMap.set(reflection.id, id);
        const previous = this.symbolToReflectionIdMap.get(id);
        if (previous) {
            if (typeof previous === "number") {
                this.symbolToReflectionIdMap.set(id, [previous, reflection.id]);
            }
            else {
                previous.push(reflection.id);
            }
        }
        else {
            this.symbolToReflectionIdMap.set(id, reflection.id);
        }
    }
    symbolIdHasBeenRemoved(id) {
        return this.removedSymbolIds.has(id);
    }
    /**
     * THIS MAY NOT BE USED AFTER CONVERSION HAS FINISHED.
     * @internal
     */
    getSymbolFromReflection(reflection) {
        return this.reflectionIdToSymbolMap.get(reflection.id);
    }
    getReferenceGraph() {
        if (!this.referenceGraph) {
            this.referenceGraph = new Map();
            for (const id in this.reflections) {
                const ref = this.reflections[id];
                if (ref instanceof ReferenceReflection) {
                    const target = ref.tryGetTargetReflection();
                    if (target) {
                        const refs = this.referenceGraph.get(target.id) ?? [];
                        refs.push(ref.id);
                        this.referenceGraph.set(target.id, refs);
                    }
                }
            }
        }
        return this.referenceGraph;
    }
    toObject(serializer) {
        const symbolIdMap = {};
        this.reflectionIdToSymbolIdMap.forEach((sid, id) => {
            symbolIdMap[id] = sid.toObject(serializer);
        });
        return {
            ...super.toObject(serializer),
            variant: this.variant,
            packageName: this.packageName,
            packageVersion: this.packageVersion,
            readme: Comment.serializeDisplayParts(serializer, this.readme),
            symbolIdMap,
            files: serializer.toObject(this.files),
        };
    }
    fromObject(de, obj) {
        super.fromObject(de, obj);
        // If updating this, also check the block in DeclarationReflection.fromObject.
        this.packageName = obj.packageName;
        this.packageVersion = obj.packageVersion;
        if (obj.readme) {
            this.readme = Comment.deserializeDisplayParts(de, obj.readme);
        }
        this.files.fromObject(de, obj.files || {});
        de.defer(() => {
            // Unnecessary conditional in release
            for (const [id, sid] of Object.entries(obj.symbolIdMap || {})) {
                const refl = this.getReflectionById(de.oldIdToNewId[+id] ?? -1);
                if (refl) {
                    this.registerSymbolId(refl, new ReflectionSymbolId(sid));
                }
                else {
                    de.logger.warn(de.application.i18n.serialized_project_referenced_0_not_part_of_project(id.toString()));
                }
            }
        });
    }
}
