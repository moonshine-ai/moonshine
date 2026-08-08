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
import { ReflectionKind, ContainerReflection, ReferenceReflection, } from "../../models/reflections/index.js";
import { ReflectionGroup } from "../../models/ReflectionGroup.js";
import { ConverterComponent } from "../components.js";
import { getSortFunction } from "../../utils/sort.js";
import { Option } from "../../utils/index.js";
import { Comment } from "../../models/index.js";
import { ConverterEvents } from "../converter-events.js";
import { ApplicationEvents } from "../../application-events.js";
import assert from "assert";
// Same as the defaultKindSortOrder in sort.ts
const defaultGroupOrder = [
    ReflectionKind.Document,
    // project is never a child so never added to a group
    ReflectionKind.Module,
    ReflectionKind.Namespace,
    ReflectionKind.Enum,
    ReflectionKind.EnumMember,
    ReflectionKind.Class,
    ReflectionKind.Interface,
    ReflectionKind.TypeAlias,
    ReflectionKind.Constructor,
    ReflectionKind.Property,
    ReflectionKind.Variable,
    ReflectionKind.Function,
    ReflectionKind.Accessor,
    ReflectionKind.Method,
    ReflectionKind.Reference,
    // others are never added to groups
];
/**
 * A handler that sorts and groups the found reflections in the resolving phase.
 *
 * The handler sets the `groups` property of all container reflections.
 */
let GroupPlugin = (() => {
    let _classSuper = ConverterComponent;
    let _groupOrder_decorators;
    let _groupOrder_initializers = [];
    let _groupOrder_extraInitializers = [];
    let _sortEntryPoints_decorators;
    let _sortEntryPoints_initializers = [];
    let _sortEntryPoints_extraInitializers = [];
    let _groupReferencesByType_decorators;
    let _groupReferencesByType_initializers = [];
    let _groupReferencesByType_extraInitializers = [];
    return class GroupPlugin extends _classSuper {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(_classSuper[Symbol.metadata] ?? null) : void 0;
            _groupOrder_decorators = [Option("groupOrder")];
            _sortEntryPoints_decorators = [Option("sortEntryPoints")];
            _groupReferencesByType_decorators = [Option("groupReferencesByType")];
            __esDecorate(this, null, _groupOrder_decorators, { kind: "accessor", name: "groupOrder", static: false, private: false, access: { has: obj => "groupOrder" in obj, get: obj => obj.groupOrder, set: (obj, value) => { obj.groupOrder = value; } }, metadata: _metadata }, _groupOrder_initializers, _groupOrder_extraInitializers);
            __esDecorate(this, null, _sortEntryPoints_decorators, { kind: "accessor", name: "sortEntryPoints", static: false, private: false, access: { has: obj => "sortEntryPoints" in obj, get: obj => obj.sortEntryPoints, set: (obj, value) => { obj.sortEntryPoints = value; } }, metadata: _metadata }, _sortEntryPoints_initializers, _sortEntryPoints_extraInitializers);
            __esDecorate(this, null, _groupReferencesByType_decorators, { kind: "accessor", name: "groupReferencesByType", static: false, private: false, access: { has: obj => "groupReferencesByType" in obj, get: obj => obj.groupReferencesByType, set: (obj, value) => { obj.groupReferencesByType = value; } }, metadata: _metadata }, _groupReferencesByType_initializers, _groupReferencesByType_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        sortFunction;
        #groupOrder_accessor_storage = __runInitializers(this, _groupOrder_initializers, void 0);
        get groupOrder() { return this.#groupOrder_accessor_storage; }
        set groupOrder(value) { this.#groupOrder_accessor_storage = value; }
        #sortEntryPoints_accessor_storage = (__runInitializers(this, _groupOrder_extraInitializers), __runInitializers(this, _sortEntryPoints_initializers, void 0));
        get sortEntryPoints() { return this.#sortEntryPoints_accessor_storage; }
        set sortEntryPoints(value) { this.#sortEntryPoints_accessor_storage = value; }
        #groupReferencesByType_accessor_storage = (__runInitializers(this, _sortEntryPoints_extraInitializers), __runInitializers(this, _groupReferencesByType_initializers, void 0));
        get groupReferencesByType() { return this.#groupReferencesByType_accessor_storage; }
        set groupReferencesByType(value) { this.#groupReferencesByType_accessor_storage = value; }
        static WEIGHTS = [];
        constructor(owner) {
            super(owner);
            __runInitializers(this, _groupReferencesByType_extraInitializers);
            this.owner.on(ConverterEvents.RESOLVE_END, this.onEndResolve.bind(this), -100);
            this.application.on(ApplicationEvents.REVIVE, this.onRevive.bind(this), -100);
        }
        /**
         * Triggered when the converter has finished resolving a project.
         *
         * @param context  The context object describing the current state the converter is in.
         */
        onEndResolve(context) {
            this.setup();
            this.group(context.project);
            for (const id in context.project.reflections) {
                const reflection = context.project.reflections[id];
                if (reflection instanceof ContainerReflection) {
                    this.group(reflection);
                }
            }
        }
        onRevive(project) {
            this.setup();
            this.group(project);
            for (const refl of project.getReflectionsByKind(ReflectionKind.SomeModule)) {
                assert(refl.isDeclaration());
                this.group(refl);
            }
        }
        setup() {
            this.sortFunction = getSortFunction(this.application.options);
            GroupPlugin.WEIGHTS = this.groupOrder;
            if (GroupPlugin.WEIGHTS.length === 0) {
                GroupPlugin.WEIGHTS = defaultGroupOrder.map((kind) => this.application.internationalization.kindPluralString(kind));
            }
        }
        group(reflection) {
            if (reflection.childrenIncludingDocuments && !reflection.groups) {
                if (reflection.children) {
                    if (this.sortEntryPoints ||
                        !reflection.children.some((c) => c.kindOf(ReflectionKind.Module))) {
                        this.sortFunction(reflection.children);
                        this.sortFunction(reflection.documents || []);
                        this.sortFunction(reflection.childrenIncludingDocuments);
                    }
                }
                else if (reflection.documents) {
                    this.sortFunction(reflection.documents);
                    this.sortFunction(reflection.childrenIncludingDocuments);
                }
                reflection.groups = this.getReflectionGroups(reflection, reflection.childrenIncludingDocuments);
            }
        }
        /**
         * Extracts the groups for a given reflection.
         *
         * @privateRemarks
         * If you change this, also update extractCategories in CategoryPlugin accordingly.
         */
        getGroups(reflection) {
            return GroupPlugin.getGroups(reflection, this.groupReferencesByType, this.application.internationalization);
        }
        static getGroups(reflection, groupReferencesByType, internationalization) {
            const groups = new Set();
            function extractGroupTags(comment) {
                if (!comment)
                    return;
                for (const tag of comment.blockTags) {
                    if (tag.tag === "@group") {
                        groups.add(Comment.combineDisplayParts(tag.content).trim());
                    }
                }
            }
            if (reflection.isDeclaration()) {
                extractGroupTags(reflection.comment);
                for (const sig of reflection.getNonIndexSignatures()) {
                    extractGroupTags(sig.comment);
                }
                if (reflection.type?.type === "reflection") {
                    extractGroupTags(reflection.type.declaration.comment);
                    for (const sig of reflection.type.declaration.getNonIndexSignatures()) {
                        extractGroupTags(sig.comment);
                    }
                }
            }
            if (reflection.isDocument() && "group" in reflection.frontmatter) {
                groups.add(String(reflection.frontmatter["group"]));
            }
            groups.delete("");
            if (groups.size === 0) {
                if (reflection instanceof ReferenceReflection &&
                    groupReferencesByType) {
                    groups.add(internationalization.kindPluralString(reflection.getTargetReflectionDeep().kind));
                }
                else {
                    groups.add(internationalization.kindPluralString(reflection.kind));
                }
            }
            return groups;
        }
        /**
         * Create a grouped representation of the given list of reflections.
         *
         * Reflections are grouped by kind and sorted by weight and name.
         *
         * @param reflections  The reflections that should be grouped.
         * @returns An array containing all children of the given reflection grouped by their kind.
         */
        getReflectionGroups(parent, reflections) {
            const groups = new Map();
            reflections.forEach((child) => {
                for (const name of this.getGroups(child)) {
                    let group = groups.get(name);
                    if (!group) {
                        group = new ReflectionGroup(name, child);
                        groups.set(name, group);
                    }
                    group.children.push(child);
                }
            });
            if (parent.comment) {
                for (const tag of parent.comment.blockTags) {
                    if (tag.tag === "@groupDescription") {
                        const { header, body } = Comment.splitPartsToHeaderAndBody(tag.content);
                        const cat = groups.get(header);
                        if (cat) {
                            cat.description = body;
                        }
                        else {
                            this.application.logger.warn(this.application.i18n.comment_for_0_includes_groupDescription_for_1_but_no_child_in_group(parent.getFriendlyFullName(), header));
                        }
                    }
                }
            }
            return Array.from(groups.values()).sort(GroupPlugin.sortGroupCallback);
        }
        /**
         * Callback used to sort groups by name.
         */
        static sortGroupCallback(a, b) {
            let aWeight = GroupPlugin.WEIGHTS.indexOf(a.title);
            let bWeight = GroupPlugin.WEIGHTS.indexOf(b.title);
            if (aWeight === -1 || bWeight === -1) {
                let asteriskIndex = GroupPlugin.WEIGHTS.indexOf("*");
                if (asteriskIndex === -1) {
                    asteriskIndex = GroupPlugin.WEIGHTS.length;
                }
                if (aWeight === -1) {
                    aWeight = asteriskIndex;
                }
                if (bWeight === -1) {
                    bWeight = asteriskIndex;
                }
            }
            if (aWeight === bWeight) {
                return a.title > b.title ? 1 : -1;
            }
            return aWeight - bWeight;
        }
    };
})();
export { GroupPlugin };
