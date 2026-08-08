var __esDecorate = (this && this.__esDecorate) || function (ctor, descriptorIn, decorators, contextIn, initializers, extraInitializers) {
    function accept(f) { if (f !== void 0 && typeof f !== "function") throw new TypeError("Function expected"); return f; }
    var kind = contextIn.kind, key = kind === "getter" ? "get" : kind === "setter" ? "set" : "value";
    var target = !descriptorIn && ctor ? contextIn["static"] ? ctor : ctor.prototype : null;
    var descriptor = descriptorIn || (target ? Object.getOwnPropertyDescriptor(target, contextIn.name) : {});
    var _, done = false;
    for (var i = decorators.length - 1; i >= 0; i--) {
        var context = {};
        for (var p in contextIn) context[p] = p === "access" ? {} : contextIn[p];
        for (var p in contextIn.access) context.access[p] = contextIn.access[p];
        context.addInitializer = function (f) { if (done) throw new TypeError("Cannot add initializers after decoration has completed"); extraInitializers.push(accept(f || null)); };
        var result = (0, decorators[i])(kind === "accessor" ? { get: descriptor.get, set: descriptor.set } : descriptor[key], context);
        if (kind === "accessor") {
            if (result === void 0) continue;
            if (result === null || typeof result !== "object") throw new TypeError("Object expected");
            if (_ = accept(result.get)) descriptor.get = _;
            if (_ = accept(result.set)) descriptor.set = _;
            if (_ = accept(result.init)) initializers.unshift(_);
        }
        else if (_ = accept(result)) {
            if (kind === "field") initializers.unshift(_);
            else descriptor[key] = _;
        }
    }
    if (target) Object.defineProperty(target, contextIn.name, descriptor);
    done = true;
};
var __runInitializers = (this && this.__runInitializers) || function (thisArg, initializers, value) {
    var useValue = arguments.length > 2;
    for (var i = 0; i < initializers.length; i++) {
        value = useValue ? initializers[i].call(thisArg, value) : initializers[i].call(thisArg);
    }
    return useValue ? value : void 0;
};
import { Comment } from "../comments/comment.js";
import { splitUnquotedString } from "./utils.js";
import { ReflectionKind } from "./kind.js";
import { NonEnumerable } from "../../utils/general.js";
/**
 * Current reflection id.
 */
let REFLECTION_ID = 0;
/**
 * Reset the reflection id.
 *
 * Used by the test cases to ensure the reflection ids won't change between runs.
 */
export function resetReflectionID() {
    REFLECTION_ID = 0;
}
export var ReflectionFlag;
(function (ReflectionFlag) {
    ReflectionFlag[ReflectionFlag["None"] = 0] = "None";
    ReflectionFlag[ReflectionFlag["Private"] = 1] = "Private";
    ReflectionFlag[ReflectionFlag["Protected"] = 2] = "Protected";
    ReflectionFlag[ReflectionFlag["Public"] = 4] = "Public";
    ReflectionFlag[ReflectionFlag["Static"] = 8] = "Static";
    ReflectionFlag[ReflectionFlag["External"] = 16] = "External";
    ReflectionFlag[ReflectionFlag["Optional"] = 32] = "Optional";
    ReflectionFlag[ReflectionFlag["Rest"] = 64] = "Rest";
    ReflectionFlag[ReflectionFlag["Abstract"] = 128] = "Abstract";
    ReflectionFlag[ReflectionFlag["Const"] = 256] = "Const";
    ReflectionFlag[ReflectionFlag["Readonly"] = 512] = "Readonly";
    ReflectionFlag[ReflectionFlag["Inherited"] = 1024] = "Inherited";
})(ReflectionFlag || (ReflectionFlag = {}));
const relevantFlags = [
    ReflectionFlag.Private,
    ReflectionFlag.Protected,
    ReflectionFlag.Static,
    ReflectionFlag.Optional,
    ReflectionFlag.Abstract,
    ReflectionFlag.Const,
    ReflectionFlag.Readonly,
];
/**
 * This must extend Array in order to work with Handlebar's each helper.
 */
export class ReflectionFlags {
    flags = ReflectionFlag.None;
    hasFlag(flag) {
        return (flag & this.flags) !== 0;
    }
    /**
     * Is this a private member?
     */
    get isPrivate() {
        return this.hasFlag(ReflectionFlag.Private);
    }
    /**
     * Is this a protected member?
     */
    get isProtected() {
        return this.hasFlag(ReflectionFlag.Protected);
    }
    /**
     * Is this a public member?
     */
    get isPublic() {
        return this.hasFlag(ReflectionFlag.Public);
    }
    /**
     * Is this a static member?
     */
    get isStatic() {
        return this.hasFlag(ReflectionFlag.Static);
    }
    /**
     * Is this a declaration from an external document?
     */
    get isExternal() {
        return this.hasFlag(ReflectionFlag.External);
    }
    /**
     * Whether this reflection is an optional component or not.
     *
     * Applies to function parameters and object members.
     */
    get isOptional() {
        return this.hasFlag(ReflectionFlag.Optional);
    }
    /**
     * Whether it's a rest parameter, like `foo(...params);`.
     */
    get isRest() {
        return this.hasFlag(ReflectionFlag.Rest);
    }
    get isAbstract() {
        return this.hasFlag(ReflectionFlag.Abstract);
    }
    get isConst() {
        return this.hasFlag(ReflectionFlag.Const);
    }
    get isReadonly() {
        return this.hasFlag(ReflectionFlag.Readonly);
    }
    get isInherited() {
        return this.hasFlag(ReflectionFlag.Inherited);
    }
    setFlag(flag, set) {
        switch (flag) {
            case ReflectionFlag.Private:
                this.setSingleFlag(ReflectionFlag.Private, set);
                if (set) {
                    this.setFlag(ReflectionFlag.Protected, false);
                    this.setFlag(ReflectionFlag.Public, false);
                }
                break;
            case ReflectionFlag.Protected:
                this.setSingleFlag(ReflectionFlag.Protected, set);
                if (set) {
                    this.setFlag(ReflectionFlag.Private, false);
                    this.setFlag(ReflectionFlag.Public, false);
                }
                break;
            case ReflectionFlag.Public:
                this.setSingleFlag(ReflectionFlag.Public, set);
                if (set) {
                    this.setFlag(ReflectionFlag.Private, false);
                    this.setFlag(ReflectionFlag.Protected, false);
                }
                break;
            default:
                this.setSingleFlag(flag, set);
        }
    }
    getFlagStrings(i18n) {
        const strings = [];
        for (const flag of relevantFlags) {
            if (this.hasFlag(flag)) {
                strings.push(i18n.flagString(flag));
            }
        }
        return strings;
    }
    setSingleFlag(flag, set) {
        if (!set && this.hasFlag(flag)) {
            this.flags ^= flag;
        }
        else if (set && !this.hasFlag(flag)) {
            this.flags |= flag;
        }
    }
    static serializedFlags = [
        "isPrivate",
        "isProtected",
        "isPublic",
        "isStatic",
        "isExternal",
        "isOptional",
        "isRest",
        "isAbstract",
        "isConst",
        "isReadonly",
        "isInherited",
    ];
    toObject() {
        return Object.fromEntries(ReflectionFlags.serializedFlags
            .filter((flag) => this[flag])
            .map((flag) => [flag, true]));
    }
    fromObject(obj) {
        for (const key of Object.keys(obj)) {
            const flagName = key.substring(2); // isPublic => Public
            if (flagName in ReflectionFlag) {
                this.setFlag(ReflectionFlag[flagName], true);
            }
        }
    }
}
export var TraverseProperty;
(function (TraverseProperty) {
    TraverseProperty[TraverseProperty["Children"] = 0] = "Children";
    TraverseProperty[TraverseProperty["Documents"] = 1] = "Documents";
    TraverseProperty[TraverseProperty["Parameters"] = 2] = "Parameters";
    TraverseProperty[TraverseProperty["TypeLiteral"] = 3] = "TypeLiteral";
    TraverseProperty[TraverseProperty["TypeParameter"] = 4] = "TypeParameter";
    TraverseProperty[TraverseProperty["Signatures"] = 5] = "Signatures";
    TraverseProperty[TraverseProperty["IndexSignature"] = 6] = "IndexSignature";
    TraverseProperty[TraverseProperty["GetSignature"] = 7] = "GetSignature";
    TraverseProperty[TraverseProperty["SetSignature"] = 8] = "SetSignature";
})(TraverseProperty || (TraverseProperty = {}));
/**
 * Base class for all reflection classes.
 *
 * While generating a documentation, TypeDoc generates an instance of {@link ProjectReflection}
 * as the root for all reflections within the project. All other reflections are represented
 * by the {@link DeclarationReflection} class.
 *
 * This base class exposes the basic properties one may use to traverse the reflection tree.
 * You can use the {@link ContainerReflection.children} and {@link parent} properties to walk the tree. The {@link ContainerReflection.groups} property
 * contains a list of all children grouped and sorted for rendering.
 * @category Reflections
 */
let Reflection = (() => {
    let _parent_decorators;
    let _parent_initializers = [];
    let _parent_extraInitializers = [];
    let _project_decorators;
    let _project_initializers = [];
    let _project_extraInitializers = [];
    return class Reflection {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(null) : void 0;
            _parent_decorators = [NonEnumerable];
            _project_decorators = [NonEnumerable];
            __esDecorate(null, null, _parent_decorators, { kind: "field", name: "parent", static: false, private: false, access: { has: obj => "parent" in obj, get: obj => obj.parent, set: (obj, value) => { obj.parent = value; } }, metadata: _metadata }, _parent_initializers, _parent_extraInitializers);
            __esDecorate(null, null, _project_decorators, { kind: "field", name: "project", static: false, private: false, access: { has: obj => "project" in obj, get: obj => obj.project, set: (obj, value) => { obj.project = value; } }, metadata: _metadata }, _project_initializers, _project_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        /**
         * Unique id of this reflection.
         */
        id;
        /**
         * The symbol name of this reflection.
         */
        name;
        /**
         * The kind of this reflection.
         */
        kind;
        flags = new ReflectionFlags();
        /**
         * The reflection this reflection is a child of.
         */
        parent = __runInitializers(this, _parent_initializers, void 0);
        project = (__runInitializers(this, _parent_extraInitializers), __runInitializers(this, _project_initializers, void 0));
        /**
         * The parsed documentation comment attached to this reflection.
         */
        comment = __runInitializers(this, _project_extraInitializers);
        /**
         * The url of this reflection in the generated documentation.
         * TODO: Reflections shouldn't know urls exist. Move this to a serializer.
         */
        url;
        /**
         * The name of the anchor of this child.
         * TODO: Reflections shouldn't know anchors exist. Move this to a serializer.
         */
        anchor;
        /**
         * Is the url pointing to an individual document?
         *
         * When FALSE, the url points to an anchor tag on a page of a different reflection.
         * TODO: Reflections shouldn't know how they are rendered. Move this to the correct serializer.
         */
        hasOwnDocument;
        constructor(name, kind, parent) {
            this.id = REFLECTION_ID++;
            this.parent = parent;
            this.project = parent?.project || this;
            this.name = name;
            this.kind = kind;
            // If our parent is external, we are too.
            if (parent?.flags.isExternal) {
                this.setFlag(ReflectionFlag.External);
            }
        }
        /**
         * Test whether this reflection is of the given kind.
         */
        kindOf(kind) {
            const kindFlags = Array.isArray(kind)
                ? kind.reduce((a, b) => a | b, 0)
                : kind;
            return (this.kind & kindFlags) !== 0;
        }
        /**
         * Return the full name of this reflection. Intended for use in debugging. For log messages
         * intended to be displayed to the user for them to fix, prefer {@link getFriendlyFullName} instead.
         *
         * The full name contains the name of this reflection and the names of all parent reflections.
         *
         * @param separator  Separator used to join the names of the reflections.
         * @returns The full name of this reflection.
         */
        getFullName(separator = ".") {
            if (this.parent && !this.parent.isProject()) {
                return this.parent.getFullName(separator) + separator + this.name;
            }
            else {
                return this.name;
            }
        }
        /**
         * Return the full name of this reflection, with signature names dropped if possible without
         * introducing ambiguity in the name.
         */
        getFriendlyFullName() {
            if (this.parent && !this.parent.isProject()) {
                if (this.kindOf(ReflectionKind.ConstructorSignature |
                    ReflectionKind.CallSignature |
                    ReflectionKind.GetSignature |
                    ReflectionKind.SetSignature)) {
                    return this.parent.getFriendlyFullName();
                }
                return this.parent.getFriendlyFullName() + "." + this.name;
            }
            else {
                return this.name;
            }
        }
        /**
         * Set a flag on this reflection.
         */
        setFlag(flag, value = true) {
            this.flags.setFlag(flag, value);
        }
        /**
         * Has this reflection a visible comment?
         *
         * @returns TRUE when this reflection has a visible comment.
         */
        hasComment() {
            return this.comment ? this.comment.hasVisibleComponent() : false;
        }
        hasGetterOrSetter() {
            return false;
        }
        /**
         * Return a child by its name.
         *
         * @param arg The name hierarchy of the child to look for.
         * @returns The found child or undefined.
         */
        getChildByName(arg) {
            const names = Array.isArray(arg)
                ? arg
                : splitUnquotedString(arg, ".");
            const name = names[0];
            let result;
            this.traverse((child) => {
                if (child.name === name) {
                    if (names.length <= 1) {
                        result = child;
                    }
                    else {
                        result = child.getChildByName(names.slice(1));
                    }
                    return false;
                }
                return true;
            });
            return result;
        }
        /**
         * Return whether this reflection is the root / project reflection.
         */
        isProject() {
            return false;
        }
        isDeclaration() {
            return false;
        }
        isParameter() {
            return false;
        }
        isDocument() {
            return false;
        }
        isReference() {
            return this.variant === "reference";
        }
        /**
         * Check if this reflection or any of its parents have been marked with the `@deprecated` tag.
         */
        isDeprecated() {
            let signaturesDeprecated = false;
            this.visit({
                declaration(decl) {
                    if (decl.signatures?.length &&
                        decl.signatures.every((sig) => sig.comment?.getTag("@deprecated"))) {
                        signaturesDeprecated = true;
                    }
                },
            });
            if (signaturesDeprecated || this.comment?.getTag("@deprecated")) {
                return true;
            }
            return this.parent?.isDeprecated() ?? false;
        }
        visit(visitor) {
            visitor[this.variant]?.(this);
        }
        /**
         * Return a string representation of this reflection.
         */
        toString() {
            return ReflectionKind[this.kind] + " " + this.name;
        }
        /**
         * Return a string representation of this reflection and all of its children.
         *
         * Note: This is intended as a debug tool only, output may change between patch versions.
         *
         * @param indent  Used internally to indent child reflections.
         */
        toStringHierarchy(indent = "") {
            const lines = [indent + this.toString()];
            indent += "  ";
            this.traverse((child) => {
                lines.push(child.toStringHierarchy(indent));
                return true;
            });
            return lines.join("\n");
        }
        toObject(serializer) {
            return {
                id: this.id,
                name: this.name,
                variant: this.variant,
                kind: this.kind,
                flags: this.flags.toObject(),
                comment: this.comment && !this.comment.isEmpty()
                    ? serializer.toObject(this.comment)
                    : undefined,
            };
        }
        fromObject(de, obj) {
            // DO NOT copy id from obj. When deserializing reflections
            // they should be given new ids since they belong to a different project.
            this.name = obj.name;
            // Skip copying variant, we know it's already the correct value because the deserializer
            // will construct the correct class type.
            this.kind = obj.kind;
            this.flags.fromObject(obj.flags);
            // Parent is set during construction, so we don't need to do it here.
            this.comment = de.revive(obj.comment, () => new Comment());
            // url, anchor, hasOwnDocument, _alias, _aliases are set during rendering and only relevant during render.
            // It doesn't make sense to serialize them to json, or restore them.
        }
    };
})();
export { Reflection };
