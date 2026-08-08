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
import assert from "assert";
import { ApplicationEvents } from "../../application-events.js";
import { Comment, ContainerReflection, ReflectionCategory, ReflectionKind, } from "../../models/index.js";
import { Option, getSortFunction } from "../../utils/index.js";
import { ConverterComponent } from "../components.js";
import { ConverterEvents } from "../converter-events.js";
/**
 * A handler that sorts and categorizes the found reflections in the resolving phase.
 *
 * The handler sets the ´category´ property of all reflections.
 */
let CategoryPlugin = (() => {
    let _classSuper = ConverterComponent;
    let _defaultCategory_decorators;
    let _defaultCategory_initializers = [];
    let _defaultCategory_extraInitializers = [];
    let _categoryOrder_decorators;
    let _categoryOrder_initializers = [];
    let _categoryOrder_extraInitializers = [];
    let _categorizeByGroup_decorators;
    let _categorizeByGroup_initializers = [];
    let _categorizeByGroup_extraInitializers = [];
    return class CategoryPlugin extends _classSuper {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(_classSuper[Symbol.metadata] ?? null) : void 0;
            _defaultCategory_decorators = [Option("defaultCategory")];
            _categoryOrder_decorators = [Option("categoryOrder")];
            _categorizeByGroup_decorators = [Option("categorizeByGroup")];
            __esDecorate(this, null, _defaultCategory_decorators, { kind: "accessor", name: "defaultCategory", static: false, private: false, access: { has: obj => "defaultCategory" in obj, get: obj => obj.defaultCategory, set: (obj, value) => { obj.defaultCategory = value; } }, metadata: _metadata }, _defaultCategory_initializers, _defaultCategory_extraInitializers);
            __esDecorate(this, null, _categoryOrder_decorators, { kind: "accessor", name: "categoryOrder", static: false, private: false, access: { has: obj => "categoryOrder" in obj, get: obj => obj.categoryOrder, set: (obj, value) => { obj.categoryOrder = value; } }, metadata: _metadata }, _categoryOrder_initializers, _categoryOrder_extraInitializers);
            __esDecorate(this, null, _categorizeByGroup_decorators, { kind: "accessor", name: "categorizeByGroup", static: false, private: false, access: { has: obj => "categorizeByGroup" in obj, get: obj => obj.categorizeByGroup, set: (obj, value) => { obj.categorizeByGroup = value; } }, metadata: _metadata }, _categorizeByGroup_initializers, _categorizeByGroup_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        sortFunction;
        #defaultCategory_accessor_storage = __runInitializers(this, _defaultCategory_initializers, void 0);
        get defaultCategory() { return this.#defaultCategory_accessor_storage; }
        set defaultCategory(value) { this.#defaultCategory_accessor_storage = value; }
        #categoryOrder_accessor_storage = (__runInitializers(this, _defaultCategory_extraInitializers), __runInitializers(this, _categoryOrder_initializers, void 0));
        get categoryOrder() { return this.#categoryOrder_accessor_storage; }
        set categoryOrder(value) { this.#categoryOrder_accessor_storage = value; }
        #categorizeByGroup_accessor_storage = (__runInitializers(this, _categoryOrder_extraInitializers), __runInitializers(this, _categorizeByGroup_initializers, void 0));
        get categorizeByGroup() { return this.#categorizeByGroup_accessor_storage; }
        set categorizeByGroup(value) { this.#categorizeByGroup_accessor_storage = value; }
        // For use in static methods
        static defaultCategory = "Other";
        static WEIGHTS = [];
        constructor(owner) {
            super(owner);
            __runInitializers(this, _categorizeByGroup_extraInitializers);
            this.owner.on(ConverterEvents.RESOLVE_END, this.onEndResolve.bind(this), -200);
            this.application.on(ApplicationEvents.REVIVE, this.onRevive.bind(this), -200);
        }
        onRevive(project) {
            this.setup();
            this.categorize(project);
            for (const refl of project.getReflectionsByKind(ReflectionKind.SomeModule)) {
                assert(refl.isDeclaration());
                this.categorize(refl);
            }
        }
        /**
         * Triggered when the converter begins converting a project.
         */
        setup() {
            this.sortFunction = getSortFunction(this.application.options);
            // Set up static properties
            if (this.defaultCategory) {
                CategoryPlugin.defaultCategory = this.defaultCategory;
            }
            CategoryPlugin.WEIGHTS = this.categoryOrder;
        }
        /**
         * Triggered when the converter has finished resolving a project.
         *
         * @param context  The context object describing the current state the converter is in.
         */
        onEndResolve(context) {
            this.setup();
            const project = context.project;
            this.categorize(project);
            for (const id in project.reflections) {
                const reflection = project.reflections[id];
                if (reflection instanceof ContainerReflection) {
                    this.categorize(reflection);
                }
            }
        }
        categorize(obj) {
            if (this.categorizeByGroup) {
                this.groupCategorize(obj);
            }
            else {
                this.lumpCategorize(obj);
            }
        }
        groupCategorize(obj) {
            if (!obj.groups || obj.groups.length === 0) {
                return;
            }
            obj.groups.forEach((group) => {
                if (group.categories)
                    return;
                group.categories = this.getReflectionCategories(obj, group.children);
                if (group.categories.length > 1) {
                    group.categories.sort(CategoryPlugin.sortCatCallback);
                }
                else if (group.categories.length === 1 &&
                    group.categories[0].title === CategoryPlugin.defaultCategory) {
                    // no categories if everything is uncategorized
                    group.categories = undefined;
                }
            });
        }
        lumpCategorize(obj) {
            if (!obj.childrenIncludingDocuments || obj.categories) {
                return;
            }
            obj.categories = this.getReflectionCategories(obj, obj.childrenIncludingDocuments);
            if (obj.categories.length > 1) {
                obj.categories.sort(CategoryPlugin.sortCatCallback);
            }
            else if (obj.categories.length === 1 &&
                obj.categories[0].title === CategoryPlugin.defaultCategory) {
                // no categories if everything is uncategorized
                obj.categories = undefined;
            }
        }
        /**
         * Create a categorized representation of the given list of reflections.
         *
         * @param reflections  The reflections that should be categorized.
         * @returns An array containing all children of the given reflection categorized
         */
        getReflectionCategories(parent, reflections) {
            const categories = new Map();
            for (const child of reflections) {
                const childCategories = CategoryPlugin.getCategories(child);
                if (childCategories.size === 0) {
                    childCategories.add(CategoryPlugin.defaultCategory);
                }
                for (const childCat of childCategories) {
                    const category = categories.get(childCat);
                    if (category) {
                        category.children.push(child);
                    }
                    else {
                        const cat = new ReflectionCategory(childCat);
                        cat.children.push(child);
                        categories.set(childCat, cat);
                    }
                }
            }
            if (parent.comment) {
                for (const tag of parent.comment.blockTags) {
                    if (tag.tag === "@categoryDescription") {
                        const { header, body } = Comment.splitPartsToHeaderAndBody(tag.content);
                        const cat = categories.get(header);
                        if (cat) {
                            cat.description = body;
                        }
                        else {
                            this.application.logger.warn(this.application.i18n.comment_for_0_includes_categoryDescription_for_1_but_no_child_in_group(parent.getFriendlyFullName(), header));
                        }
                    }
                }
            }
            for (const cat of categories.values()) {
                this.sortFunction(cat.children);
            }
            return Array.from(categories.values());
        }
        /**
         * Callback used to sort categories by name.
         *
         * @param a The left reflection to sort.
         * @param b The right reflection to sort.
         * @returns The sorting weight.
         */
        static sortCatCallback(a, b) {
            let aWeight = CategoryPlugin.WEIGHTS.indexOf(a.title);
            let bWeight = CategoryPlugin.WEIGHTS.indexOf(b.title);
            if (aWeight === -1 || bWeight === -1) {
                let asteriskIndex = CategoryPlugin.WEIGHTS.indexOf("*");
                if (asteriskIndex === -1) {
                    asteriskIndex = CategoryPlugin.WEIGHTS.length;
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
        static getCategories(reflection) {
            const categories = new Set();
            function discoverCategories(comment) {
                if (!comment)
                    return;
                for (const tag of comment.blockTags) {
                    if (tag.tag === "@category") {
                        categories.add(Comment.combineDisplayParts(tag.content).trim());
                    }
                }
            }
            discoverCategories(reflection.comment);
            if (reflection.isDeclaration()) {
                for (const sig of reflection.getNonIndexSignatures()) {
                    discoverCategories(sig.comment);
                }
                if (reflection.type?.type === "reflection") {
                    discoverCategories(reflection.type.declaration.comment);
                    for (const sig of reflection.type.declaration.getNonIndexSignatures()) {
                        discoverCategories(sig.comment);
                    }
                }
            }
            if (reflection.isDocument() && "category" in reflection.frontmatter) {
                categories.add(String(reflection.frontmatter["category"]));
            }
            categories.delete("");
            return categories;
        }
    };
})();
export { CategoryPlugin };
