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
import { Comment, CommentTag, DeclarationReflection, ParameterReflection, ReflectionFlag, ReflectionKind, ReflectionType, SignatureReflection, } from "../../models/index.js";
import { Option, filterMap, partition, removeIf, removeIfPresent, unique, } from "../../utils/index.js";
import { setIntersection } from "../../utils/set.js";
import { ConverterComponent } from "../components.js";
import { ConverterEvents } from "../converter-events.js";
import { CategoryPlugin } from "./CategoryPlugin.js";
/**
 * These tags are not useful to display in the generated documentation.
 * They should be ignored when parsing comments. Any relevant type information
 * (for JS users) will be consumed by TypeScript and need not be preserved
 * in the comment.
 *
 * Note that param/arg/argument/return/returns are not present.
 * These tags will have their type information stripped when parsing, but still
 * provide useful information for documentation.
 */
const NEVER_RENDERED = [
    "@augments",
    "@callback",
    "@class",
    "@constructor",
    "@enum",
    "@extends",
    "@this",
    "@type",
    "@typedef",
    "@jsx",
];
// We might make this user configurable at some point, but for now,
// this set is configured here.
const MUTUALLY_EXCLUSIVE_MODIFIERS = [
    new Set([
        "@alpha",
        "@beta",
        "@experimental",
        "@internal",
        "@public",
    ]),
];
/**
 * Handles most behavior triggered by comments. `@group` and `@category` are handled by their respective plugins, but everything else is here.
 *
 * How it works today
 * ==================
 * During conversion:
 * - Handle visibility flags (`@private`, `@protected`. `@public`)
 * - Handle module renames (`@module`)
 * - Remove excluded tags & comment discovery tags (`@module`, `@packageDocumentation`)
 * - Copy comments for type parameters from the parent container (for classes/interfaces)
 *
 * Resolve begin:
 * - Remove hidden reflections
 *
 * Resolve:
 * - Apply `@label` tag
 * - Copy comments on signature containers to the signature if signatures don't already have a comment
 *   and then remove the comment on the container.
 * - Copy comments to parameters and type parameters (for signatures)
 * - Apply `@group` and `@category` tags
 *
 * Resolve end:
 * - Copy auto inherited comments from heritage clauses
 * - Handle `@inheritDoc`
 * - Resolve `@link` tags to point to target reflections
 *
 * How it should work
 * ==================
 * During conversion:
 * - Handle visibility flags (`@private`, `@protected`. `@public`)
 * - Handle module renames (`@module`)
 * - Remove excluded tags & comment discovery tags (`@module`, `@packageDocumentation`)
 *
 * Resolve begin (100):
 * - Copy auto inherited comments from heritage clauses
 * - Apply `@label` tag
 *
 * Resolve begin (75)
 * - Handle `@inheritDoc`
 *
 * Resolve begin (50)
 * - Copy comments on signature containers to the signature if signatures don't already have a comment
 *   and then remove the comment on the container.
 * - Copy comments for type parameters from the parent container (for classes/interfaces)
 *
 * Resolve begin (25)
 * - Remove hidden reflections
 *
 * Resolve:
 * - Copy comments to parameters and type parameters (for signatures)
 * - Apply `@group` and `@category` tags
 *
 * Resolve end:
 * - Resolve `@link` tags to point to target reflections
 *
 */
let CommentPlugin = (() => {
    let _classSuper = ConverterComponent;
    let _excludeTags_decorators;
    let _excludeTags_initializers = [];
    let _excludeTags_extraInitializers = [];
    let _cascadedModifierTags_decorators;
    let _cascadedModifierTags_initializers = [];
    let _cascadedModifierTags_extraInitializers = [];
    let _excludeInternal_decorators;
    let _excludeInternal_initializers = [];
    let _excludeInternal_extraInitializers = [];
    let _excludePrivate_decorators;
    let _excludePrivate_initializers = [];
    let _excludePrivate_extraInitializers = [];
    let _excludeProtected_decorators;
    let _excludeProtected_initializers = [];
    let _excludeProtected_extraInitializers = [];
    let _excludeNotDocumented_decorators;
    let _excludeNotDocumented_initializers = [];
    let _excludeNotDocumented_extraInitializers = [];
    let _excludeCategories_decorators;
    let _excludeCategories_initializers = [];
    let _excludeCategories_extraInitializers = [];
    let _defaultCategory_decorators;
    let _defaultCategory_initializers = [];
    let _defaultCategory_extraInitializers = [];
    return class CommentPlugin extends _classSuper {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(_classSuper[Symbol.metadata] ?? null) : void 0;
            _excludeTags_decorators = [Option("excludeTags")];
            _cascadedModifierTags_decorators = [Option("cascadedModifierTags")];
            _excludeInternal_decorators = [Option("excludeInternal")];
            _excludePrivate_decorators = [Option("excludePrivate")];
            _excludeProtected_decorators = [Option("excludeProtected")];
            _excludeNotDocumented_decorators = [Option("excludeNotDocumented")];
            _excludeCategories_decorators = [Option("excludeCategories")];
            _defaultCategory_decorators = [Option("defaultCategory")];
            __esDecorate(this, null, _excludeTags_decorators, { kind: "accessor", name: "excludeTags", static: false, private: false, access: { has: obj => "excludeTags" in obj, get: obj => obj.excludeTags, set: (obj, value) => { obj.excludeTags = value; } }, metadata: _metadata }, _excludeTags_initializers, _excludeTags_extraInitializers);
            __esDecorate(this, null, _cascadedModifierTags_decorators, { kind: "accessor", name: "cascadedModifierTags", static: false, private: false, access: { has: obj => "cascadedModifierTags" in obj, get: obj => obj.cascadedModifierTags, set: (obj, value) => { obj.cascadedModifierTags = value; } }, metadata: _metadata }, _cascadedModifierTags_initializers, _cascadedModifierTags_extraInitializers);
            __esDecorate(this, null, _excludeInternal_decorators, { kind: "accessor", name: "excludeInternal", static: false, private: false, access: { has: obj => "excludeInternal" in obj, get: obj => obj.excludeInternal, set: (obj, value) => { obj.excludeInternal = value; } }, metadata: _metadata }, _excludeInternal_initializers, _excludeInternal_extraInitializers);
            __esDecorate(this, null, _excludePrivate_decorators, { kind: "accessor", name: "excludePrivate", static: false, private: false, access: { has: obj => "excludePrivate" in obj, get: obj => obj.excludePrivate, set: (obj, value) => { obj.excludePrivate = value; } }, metadata: _metadata }, _excludePrivate_initializers, _excludePrivate_extraInitializers);
            __esDecorate(this, null, _excludeProtected_decorators, { kind: "accessor", name: "excludeProtected", static: false, private: false, access: { has: obj => "excludeProtected" in obj, get: obj => obj.excludeProtected, set: (obj, value) => { obj.excludeProtected = value; } }, metadata: _metadata }, _excludeProtected_initializers, _excludeProtected_extraInitializers);
            __esDecorate(this, null, _excludeNotDocumented_decorators, { kind: "accessor", name: "excludeNotDocumented", static: false, private: false, access: { has: obj => "excludeNotDocumented" in obj, get: obj => obj.excludeNotDocumented, set: (obj, value) => { obj.excludeNotDocumented = value; } }, metadata: _metadata }, _excludeNotDocumented_initializers, _excludeNotDocumented_extraInitializers);
            __esDecorate(this, null, _excludeCategories_decorators, { kind: "accessor", name: "excludeCategories", static: false, private: false, access: { has: obj => "excludeCategories" in obj, get: obj => obj.excludeCategories, set: (obj, value) => { obj.excludeCategories = value; } }, metadata: _metadata }, _excludeCategories_initializers, _excludeCategories_extraInitializers);
            __esDecorate(this, null, _defaultCategory_decorators, { kind: "accessor", name: "defaultCategory", static: false, private: false, access: { has: obj => "defaultCategory" in obj, get: obj => obj.defaultCategory, set: (obj, value) => { obj.defaultCategory = value; } }, metadata: _metadata }, _defaultCategory_initializers, _defaultCategory_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        #excludeTags_accessor_storage = __runInitializers(this, _excludeTags_initializers, void 0);
        get excludeTags() { return this.#excludeTags_accessor_storage; }
        set excludeTags(value) { this.#excludeTags_accessor_storage = value; }
        #cascadedModifierTags_accessor_storage = (__runInitializers(this, _excludeTags_extraInitializers), __runInitializers(this, _cascadedModifierTags_initializers, void 0));
        get cascadedModifierTags() { return this.#cascadedModifierTags_accessor_storage; }
        set cascadedModifierTags(value) { this.#cascadedModifierTags_accessor_storage = value; }
        #excludeInternal_accessor_storage = (__runInitializers(this, _cascadedModifierTags_extraInitializers), __runInitializers(this, _excludeInternal_initializers, void 0));
        get excludeInternal() { return this.#excludeInternal_accessor_storage; }
        set excludeInternal(value) { this.#excludeInternal_accessor_storage = value; }
        #excludePrivate_accessor_storage = (__runInitializers(this, _excludeInternal_extraInitializers), __runInitializers(this, _excludePrivate_initializers, void 0));
        get excludePrivate() { return this.#excludePrivate_accessor_storage; }
        set excludePrivate(value) { this.#excludePrivate_accessor_storage = value; }
        #excludeProtected_accessor_storage = (__runInitializers(this, _excludePrivate_extraInitializers), __runInitializers(this, _excludeProtected_initializers, void 0));
        get excludeProtected() { return this.#excludeProtected_accessor_storage; }
        set excludeProtected(value) { this.#excludeProtected_accessor_storage = value; }
        #excludeNotDocumented_accessor_storage = (__runInitializers(this, _excludeProtected_extraInitializers), __runInitializers(this, _excludeNotDocumented_initializers, void 0));
        get excludeNotDocumented() { return this.#excludeNotDocumented_accessor_storage; }
        set excludeNotDocumented(value) { this.#excludeNotDocumented_accessor_storage = value; }
        #excludeCategories_accessor_storage = (__runInitializers(this, _excludeNotDocumented_extraInitializers), __runInitializers(this, _excludeCategories_initializers, void 0));
        get excludeCategories() { return this.#excludeCategories_accessor_storage; }
        set excludeCategories(value) { this.#excludeCategories_accessor_storage = value; }
        #defaultCategory_accessor_storage = (__runInitializers(this, _excludeCategories_extraInitializers), __runInitializers(this, _defaultCategory_initializers, void 0));
        get defaultCategory() { return this.#defaultCategory_accessor_storage; }
        set defaultCategory(value) { this.#defaultCategory_accessor_storage = value; }
        _excludeKinds = __runInitializers(this, _defaultCategory_extraInitializers);
        get excludeNotDocumentedKinds() {
            this._excludeKinds ??= this.application.options
                .getValue("excludeNotDocumentedKinds")
                .reduce((a, b) => a | ReflectionKind[b], 0);
            return this._excludeKinds;
        }
        constructor(owner) {
            super(owner);
            this.owner.on(ConverterEvents.CREATE_DECLARATION, this.onDeclaration.bind(this));
            this.owner.on(ConverterEvents.CREATE_SIGNATURE, this.onDeclaration.bind(this));
            this.owner.on(ConverterEvents.CREATE_TYPE_PARAMETER, this.onCreateTypeParameter.bind(this));
            this.owner.on(ConverterEvents.RESOLVE_BEGIN, this.onBeginResolve.bind(this));
            this.owner.on(ConverterEvents.RESOLVE, this.onResolve.bind(this));
            this.owner.on(ConverterEvents.END, () => {
                this._excludeKinds = undefined;
            });
        }
        /**
         * Apply all comment tag modifiers to the given reflection.
         *
         * @param reflection  The reflection the modifiers should be applied to.
         * @param comment  The comment that should be searched for modifiers.
         */
        applyModifiers(reflection, comment) {
            if (reflection.kindOf(ReflectionKind.SomeModule)) {
                comment.removeModifier("@namespace");
            }
            if (reflection.kindOf(ReflectionKind.Interface)) {
                comment.removeModifier("@interface");
            }
            if (comment.hasModifier("@abstract")) {
                if (reflection.kindOf(ReflectionKind.SomeSignature)) {
                    reflection.parent.setFlag(ReflectionFlag.Abstract);
                }
                else {
                    reflection.setFlag(ReflectionFlag.Abstract);
                }
                comment.removeModifier("@abstract");
            }
            if (comment.hasModifier("@private")) {
                reflection.setFlag(ReflectionFlag.Private);
                if (reflection.kindOf(ReflectionKind.CallSignature)) {
                    reflection.parent?.setFlag(ReflectionFlag.Private);
                }
                comment.removeModifier("@private");
            }
            if (comment.hasModifier("@protected")) {
                reflection.setFlag(ReflectionFlag.Protected);
                if (reflection.kindOf(ReflectionKind.CallSignature)) {
                    reflection.parent?.setFlag(ReflectionFlag.Protected);
                }
                comment.removeModifier("@protected");
            }
            if (comment.hasModifier("@public")) {
                reflection.setFlag(ReflectionFlag.Public);
                if (reflection.kindOf(ReflectionKind.CallSignature)) {
                    reflection.parent?.setFlag(ReflectionFlag.Public);
                }
                comment.removeModifier("@public");
            }
            if (comment.hasModifier("@readonly")) {
                const target = reflection.kindOf(ReflectionKind.GetSignature)
                    ? reflection.parent
                    : reflection;
                target.setFlag(ReflectionFlag.Readonly);
                comment.removeModifier("@readonly");
            }
            if (comment.hasModifier("@event") ||
                comment.hasModifier("@eventProperty")) {
                comment.blockTags.push(new CommentTag("@group", [{ kind: "text", text: "Events" }]));
                comment.removeModifier("@event");
                comment.removeModifier("@eventProperty");
            }
            if (reflection.kindOf(ReflectionKind.Project | ReflectionKind.SomeModule)) {
                comment.removeTags("@module");
                comment.removeModifier("@packageDocumentation");
            }
        }
        /**
         * Triggered when the converter has created a type parameter reflection.
         *
         * @param context  The context object describing the current state the converter is in.
         * @param reflection  The reflection that is currently processed.
         */
        onCreateTypeParameter(_context, reflection) {
            const comment = reflection.parent?.comment;
            if (comment) {
                let tag = comment.getIdentifiedTag(reflection.name, "@typeParam");
                if (!tag) {
                    tag = comment.getIdentifiedTag(reflection.name, "@template");
                }
                if (!tag) {
                    tag = comment.getIdentifiedTag(`<${reflection.name}>`, "@param");
                }
                if (!tag) {
                    tag = comment.getIdentifiedTag(reflection.name, "@param");
                }
                if (tag) {
                    reflection.comment = new Comment(tag.content);
                    reflection.comment.sourcePath = comment.sourcePath;
                    removeIfPresent(comment.blockTags, tag);
                }
            }
        }
        /**
         * Triggered when the converter has created a declaration or signature reflection.
         *
         * Invokes the comment parser.
         *
         * @param context  The context object describing the current state the converter is in.
         * @param reflection  The reflection that is currently processed.
         * @param node  The node that is currently processed if available.
         */
        onDeclaration(_context, reflection) {
            this.cascadeModifiers(reflection);
            const comment = reflection.comment;
            if (!comment)
                return;
            if (reflection.kindOf(ReflectionKind.SomeModule)) {
                const tag = comment.getTag("@module");
                if (tag) {
                    // If no name is specified, this is a flag to mark a comment as a module comment
                    // and should not result in a reflection rename.
                    const newName = Comment.combineDisplayParts(tag.content).trim();
                    if (newName.length && !newName.includes("\n")) {
                        reflection.name = newName;
                    }
                    removeIfPresent(comment.blockTags, tag);
                }
            }
            this.applyModifiers(reflection, comment);
            this.removeExcludedTags(comment);
        }
        /**
         * Triggered when the converter begins resolving a project.
         *
         * @param context  The context object describing the current state the converter is in.
         */
        onBeginResolve(context) {
            if (context.project.comment) {
                this.applyModifiers(context.project, context.project.comment);
                this.removeExcludedTags(context.project.comment);
            }
            const project = context.project;
            const reflections = Object.values(project.reflections);
            // Remove hidden reflections
            const hidden = new Set();
            for (const ref of reflections) {
                if (ref.kindOf(ReflectionKind.Accessor) && ref.flags.isReadonly) {
                    const decl = ref;
                    if (decl.setSignature) {
                        hidden.add(decl.setSignature);
                    }
                    // Clear flag set by @readonly since it shouldn't be rendered.
                    ref.setFlag(ReflectionFlag.Readonly, false);
                }
                if (this.isHidden(ref)) {
                    hidden.add(ref);
                }
            }
            hidden.forEach((reflection) => project.removeReflection(reflection));
            // remove functions with empty signatures after their signatures have been removed
            const [allRemoved, someRemoved] = partition(unique(filterMap(hidden, (reflection) => reflection.parent?.kindOf(ReflectionKind.SignatureContainer)
                ? reflection.parent
                : void 0)), (method) => method.getNonIndexSignatures().length === 0);
            allRemoved.forEach((reflection) => {
                project.removeReflection(reflection);
            });
            someRemoved.forEach((reflection) => {
                reflection.sources = reflection
                    .getNonIndexSignatures()
                    .flatMap((s) => s.sources ?? []);
            });
        }
        /**
         * Triggered when the converter resolves a reflection.
         *
         * Cleans up comment tags related to signatures like `@param` or `@returns`
         * and moves their data to the corresponding parameter reflections.
         *
         * This hook also copies over the comment of function implementations to their
         * signatures.
         *
         * @param context  The context object describing the current state the converter is in.
         * @param reflection  The reflection that is currently resolved.
         */
        onResolve(context, reflection) {
            if (reflection.comment) {
                if (reflection.comment.label &&
                    !/[A-Z_][A-Z0-9_]/.test(reflection.comment.label)) {
                    context.logger.warn(context.i18n.label_0_for_1_cannot_be_referenced(reflection.comment.label, reflection.getFriendlyFullName()));
                }
                for (const group of MUTUALLY_EXCLUSIVE_MODIFIERS) {
                    const intersect = setIntersection(group, reflection.comment.modifierTags);
                    if (intersect.size > 1) {
                        const [a, b] = intersect;
                        context.logger.warn(context.i18n.modifier_tag_0_is_mutually_exclusive_with_1_in_comment_for_2(a, b, reflection.getFriendlyFullName()));
                    }
                }
                mergeSeeTags(reflection.comment);
                movePropertyTags(reflection.comment, reflection);
                // Unlike other modifiers, this one has to wait until resolution to be removed
                // as it needs to remain present so that it can be checked when `@hidden` tags are
                // being processed.
                if (reflection.kindOf(ReflectionKind.Class)) {
                    reflection.comment.removeModifier("@hideconstructor");
                }
            }
            if ((reflection instanceof DeclarationReflection ||
                reflection instanceof ParameterReflection) &&
                reflection.comment) {
                let sigs = [];
                if (reflection.type instanceof ReflectionType) {
                    sigs = reflection.type.declaration.getNonIndexSignatures();
                }
                else if (reflection instanceof DeclarationReflection) {
                    sigs = reflection.getNonIndexSignatures();
                }
                // For variables and properties, the symbol might own the comment but we might also
                // have @param and @returns comments for an owned signature. Only do this if there is
                // exactly one signature as otherwise we have no hope of doing validation right.
                if (sigs.length === 1 && !sigs[0].comment) {
                    this.moveSignatureParamComments(sigs[0], reflection.comment);
                    const returnsTag = reflection.comment.getTag("@returns");
                    if (returnsTag) {
                        sigs[0].comment = new Comment();
                        sigs[0].comment.sourcePath = reflection.comment.sourcePath;
                        sigs[0].comment.blockTags.push(returnsTag);
                        reflection.comment.removeTags("@returns");
                    }
                }
                // Any cascaded tags will show up twice, once on this and once on our signatures
                // This is completely redundant, so remove them from the wrapping function.
                if (sigs.length && reflection.type?.type !== "reflection") {
                    for (const mod of this.cascadedModifierTags) {
                        reflection.comment.modifierTags.delete(mod);
                    }
                }
            }
            if (reflection instanceof SignatureReflection) {
                this.moveSignatureParamComments(reflection);
            }
        }
        moveSignatureParamComments(signature, comment = signature.comment) {
            if (!comment)
                return;
            signature.parameters?.forEach((parameter, index) => {
                if (parameter.name === "__namedParameters") {
                    const commentParams = comment.blockTags.filter((tag) => tag.tag === "@param" && !tag.name?.includes("."));
                    if (signature.parameters?.length === commentParams.length &&
                        commentParams[index].name) {
                        parameter.name = commentParams[index].name;
                    }
                }
                const tag = comment.getIdentifiedTag(parameter.name, "@param");
                if (tag) {
                    parameter.comment = new Comment(Comment.cloneDisplayParts(tag.content));
                    parameter.comment.sourcePath = comment.sourcePath;
                }
            });
            for (const parameter of signature.typeParameters || []) {
                const tag = comment.getIdentifiedTag(parameter.name, "@typeParam") ||
                    comment.getIdentifiedTag(parameter.name, "@template") ||
                    comment.getIdentifiedTag(`<${parameter.name}>`, "@param");
                if (tag) {
                    parameter.comment = new Comment(Comment.cloneDisplayParts(tag.content));
                    parameter.comment.sourcePath = comment.sourcePath;
                }
            }
            this.validateParamTags(signature, comment, signature.parameters || []);
            comment.removeTags("@param");
            comment.removeTags("@typeParam");
            comment.removeTags("@template");
        }
        removeExcludedTags(comment) {
            for (const tag of NEVER_RENDERED) {
                comment.removeTags(tag);
                comment.removeModifier(tag);
            }
            for (const tag of this.excludeTags) {
                comment.removeTags(tag);
                comment.removeModifier(tag);
            }
        }
        cascadeModifiers(reflection) {
            const parentComment = reflection.parent?.comment;
            if (!parentComment || reflection.kindOf(ReflectionKind.TypeLiteral)) {
                return;
            }
            const childMods = reflection.comment?.modifierTags ?? new Set();
            for (const mod of this.cascadedModifierTags) {
                if (parentComment.hasModifier(mod)) {
                    const exclusiveSet = MUTUALLY_EXCLUSIVE_MODIFIERS.find((tags) => tags.has(mod));
                    if (!exclusiveSet ||
                        Array.from(exclusiveSet).every((tag) => !childMods.has(tag))) {
                        reflection.comment ||= new Comment();
                        reflection.comment.modifierTags.add(mod);
                    }
                }
            }
        }
        /**
         * Determines whether or not a reflection has been hidden
         *
         * @param reflection Reflection to check if hidden
         */
        isHidden(reflection) {
            const comment = reflection.comment;
            if (reflection.flags.hasFlag(ReflectionFlag.Private) &&
                this.excludePrivate) {
                return true;
            }
            if (reflection.flags.hasFlag(ReflectionFlag.Protected) &&
                this.excludeProtected) {
                return true;
            }
            if (this.excludedByCategory(reflection)) {
                return true;
            }
            if (reflection.kindOf(ReflectionKind.ConstructorSignature |
                ReflectionKind.Constructor)) {
                if (comment?.hasModifier("@hideconstructor"))
                    return true;
                const cls = reflection.parent?.kindOf(ReflectionKind.Class)
                    ? reflection.parent
                    : reflection.parent?.parent?.kindOf(ReflectionKind.Class)
                        ? reflection.parent.parent
                        : undefined;
                if (cls?.comment?.hasModifier("@hideconstructor")) {
                    return true;
                }
            }
            if (!comment) {
                // We haven't moved comments from the parent for signatures without a direct
                // comment, so don't exclude those due to not being documented.
                if (reflection.kindOf(ReflectionKind.CallSignature |
                    ReflectionKind.ConstructorSignature) &&
                    reflection.parent?.comment) {
                    return false;
                }
                if (this.excludeNotDocumented) {
                    // Don't let excludeNotDocumented remove parameters.
                    if (!(reflection instanceof DeclarationReflection) &&
                        !(reflection instanceof SignatureReflection)) {
                        return false;
                    }
                    if (!reflection.kindOf(this.excludeNotDocumentedKinds)) {
                        return false;
                    }
                    // excludeNotDocumented should hide a module only if it has no visible children
                    if (reflection.kindOf(ReflectionKind.SomeModule)) {
                        if (!reflection.children) {
                            return true;
                        }
                        return reflection.children.every((child) => this.isHidden(child));
                    }
                    // signature containers should only be hidden if all their signatures are hidden
                    if (reflection.kindOf(ReflectionKind.SignatureContainer)) {
                        return reflection
                            .getAllSignatures()
                            .every((child) => this.isHidden(child));
                    }
                    // excludeNotDocumented should never hide parts of "type" reflections
                    return inTypeLiteral(reflection) === false;
                }
                return false;
            }
            const isHidden = comment.hasModifier("@hidden") ||
                comment.hasModifier("@ignore") ||
                (comment.hasModifier("@internal") && this.excludeInternal);
            if (!isHidden &&
                reflection.kindOf(ReflectionKind.ContainsCallSignatures)) {
                return reflection
                    .getNonIndexSignatures()
                    .every((sig) => this.isHidden(sig));
            }
            return isHidden;
        }
        excludedByCategory(reflection) {
            const excludeCategories = this.excludeCategories;
            let target;
            if (reflection instanceof DeclarationReflection) {
                target = reflection;
            }
            else if (reflection instanceof SignatureReflection) {
                target = reflection.parent;
            }
            if (!target || !excludeCategories.length)
                return false;
            const categories = CategoryPlugin.getCategories(target);
            if (categories.size === 0) {
                categories.add(this.defaultCategory);
            }
            return excludeCategories.some((cat) => categories.has(cat));
        }
        validateParamTags(signature, comment, params) {
            const paramTags = comment.blockTags.filter((tag) => tag.tag === "@param");
            removeIf(paramTags, (tag) => params.some((param) => param.name === tag.name));
            moveNestedParamTags(/* in-out */ paramTags, params, comment.sourcePath);
            if (!comment.inheritedFromParentDeclaration) {
                for (const tag of paramTags) {
                    this.application.logger.warn(this.application.i18n.signature_0_has_unused_param_with_name_1(signature.getFriendlyFullName(), tag.name ?? "(missing)"));
                }
            }
        }
    };
})();
export { CommentPlugin };
function inTypeLiteral(refl) {
    while (refl) {
        if (refl.kind === ReflectionKind.TypeLiteral) {
            return true;
        }
        refl = refl.parent;
    }
    return false;
}
function validHighlightedName(ref, name) {
    const refl = ref.reflection;
    // Assume external types are documented properly
    if (!refl)
        return true;
    // If it is a direct child, it is valid.
    if (refl.getChildByName([name]))
        return true;
    // Or if it is the child of the referenced reflection's type
    if (refl.isDeclaration() && refl.type?.type === "reflection") {
        if (refl.type.declaration.getChildByName([name])) {
            return true;
        }
    }
    return false;
}
// Moves tags like `@param foo.bar docs for bar` into the `bar` property of the `foo` parameter.
function moveNestedParamTags(
/* in-out */ paramTags, parameters, sourcePath) {
    const used = new Set();
    for (const param of parameters) {
        const visitor = {
            reflection(target) {
                const tags = paramTags.filter((t) => t.name?.startsWith(`${param.name}.`));
                for (const tag of tags) {
                    const path = tag.name.split(".");
                    path.shift();
                    const child = target.declaration.getChildOrTypePropertyByName(path);
                    if (child && !child.comment) {
                        child.comment = new Comment(Comment.cloneDisplayParts(tag.content));
                        child.comment.sourcePath = sourcePath;
                        used.add(paramTags.indexOf(tag));
                    }
                }
            },
            // #1876, also do this for unions/intersections.
            union(u) {
                u.types.forEach((t) => t.visit(visitor));
            },
            intersection(i) {
                i.types.forEach((t) => t.visit(visitor));
            },
            // #2147, support highlighting parts of a referenced type
            reference(ref) {
                for (let i = 0; i < paramTags.length; ++i) {
                    const tag = paramTags[i];
                    if (tag.name?.startsWith(`${param.name}.`)) {
                        const childName = tag.name.substring(param.name.length + 1);
                        if (!validHighlightedName(ref, childName)) {
                            continue;
                        }
                        ref.highlightedProperties ??= new Map();
                        ref.highlightedProperties.set(childName, paramTags[i].content);
                        used.add(i);
                    }
                }
            },
        };
        param.type?.visit(visitor);
    }
    const toRemove = Array.from(used)
        .sort((a, b) => a - b)
        .reverse();
    for (const index of toRemove) {
        paramTags.splice(index, 1);
    }
}
function movePropertyTags(comment, container) {
    const propTags = comment.blockTags.filter((tag) => tag.tag === "@prop" || tag.tag === "@property");
    comment.removeTags("@prop");
    comment.removeTags("@property");
    for (const prop of propTags) {
        if (!prop.name)
            continue;
        const child = container.getChildByName(prop.name);
        if (child) {
            child.comment = new Comment(Comment.cloneDisplayParts(prop.content));
            child.comment.sourcePath = comment.sourcePath;
            if (child instanceof DeclarationReflection && child.signatures) {
                for (const sig of child.signatures) {
                    sig.comment = new Comment(Comment.cloneDisplayParts(prop.content));
                    sig.comment.sourcePath = comment.sourcePath;
                }
            }
        }
    }
}
function mergeSeeTags(comment) {
    const see = comment.getTags("@see");
    if (see.length < 2)
        return;
    const index = comment.blockTags.indexOf(see[0]);
    comment.removeTags("@see");
    see[0].content = see.flatMap((part) => [
        { kind: "text", text: " - " },
        ...part.content,
        { kind: "text", text: "\n" },
    ]);
    comment.blockTags.splice(index, 0, see[0]);
}
