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
import { ConverterComponent } from "../components.js";
import { ConverterEvents } from "../converter-events.js";
import { Option } from "../../utils/index.js";
import { ContainerReflection, makeRecursiveVisitor, } from "../../models/index.js";
import { discoverAllReferenceTypes } from "../../utils/reflections.js";
import { ApplicationEvents } from "../../application-events.js";
/**
 * A plugin that resolves `{@link Foo}` tags.
 */
let LinkResolverPlugin = (() => {
    let _classSuper = ConverterComponent;
    let _validation_decorators;
    let _validation_initializers = [];
    let _validation_extraInitializers = [];
    return class LinkResolverPlugin extends _classSuper {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(_classSuper[Symbol.metadata] ?? null) : void 0;
            _validation_decorators = [Option("validation")];
            __esDecorate(this, null, _validation_decorators, { kind: "accessor", name: "validation", static: false, private: false, access: { has: obj => "validation" in obj, get: obj => obj.validation, set: (obj, value) => { obj.validation = value; } }, metadata: _metadata }, _validation_initializers, _validation_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        #validation_accessor_storage = __runInitializers(this, _validation_initializers, void 0);
        get validation() { return this.#validation_accessor_storage; }
        set validation(value) { this.#validation_accessor_storage = value; }
        constructor(owner) {
            super(owner);
            __runInitializers(this, _validation_extraInitializers);
            this.owner.on(ConverterEvents.RESOLVE_END, this.onResolve.bind(this), -300);
            this.application.on(ApplicationEvents.REVIVE, this.resolveLinks.bind(this), -300);
        }
        onResolve(context) {
            this.resolveLinks(context.project);
        }
        resolveLinks(project) {
            for (const id in project.reflections) {
                const reflection = project.reflections[id];
                if (reflection.comment) {
                    this.owner.resolveLinks(reflection.comment, reflection);
                }
                if (reflection.isDeclaration()) {
                    reflection.type?.visit(makeRecursiveVisitor({
                        union: (type) => {
                            type.elementSummaries = type.elementSummaries?.map((parts) => this.owner.resolveLinks(parts, reflection));
                        },
                    }));
                    if (reflection.readme) {
                        reflection.readme = this.owner.resolveLinks(reflection.readme, reflection);
                    }
                }
                if (reflection.isDocument()) {
                    reflection.content = this.owner.resolveLinks(reflection.content, reflection);
                }
                if (reflection.isParameter() &&
                    reflection.type?.type === "reference" &&
                    reflection.type.highlightedProperties) {
                    const resolved = new Map(Array.from(reflection.type.highlightedProperties, ([name, parts]) => {
                        return [
                            name,
                            this.owner.resolveLinks(parts, reflection),
                        ];
                    }));
                    reflection.type.highlightedProperties = resolved;
                }
                if (reflection instanceof ContainerReflection) {
                    if (reflection.groups) {
                        for (const group of reflection.groups) {
                            if (group.description) {
                                group.description = this.owner.resolveLinks(group.description, reflection);
                            }
                            if (group.categories) {
                                for (const cat of group.categories) {
                                    this.resolveCategoryLinks(cat, reflection);
                                }
                            }
                        }
                    }
                    if (reflection.categories) {
                        for (const cat of reflection.categories) {
                            this.resolveCategoryLinks(cat, reflection);
                        }
                    }
                }
            }
            if (project.readme) {
                project.readme = this.owner.resolveLinks(project.readme, project);
            }
            for (const { type, owner } of discoverAllReferenceTypes(project, false)) {
                if (!type.reflection) {
                    const resolveResult = this.owner.resolveExternalLink(type.toDeclarationReference(), owner, undefined, type.symbolId);
                    switch (typeof resolveResult) {
                        case "string":
                            type.externalUrl = resolveResult;
                            break;
                        case "object":
                            type.externalUrl = resolveResult.target;
                            break;
                    }
                }
            }
        }
        resolveCategoryLinks(category, owner) {
            if (category.description) {
                category.description = this.owner.resolveLinks(category.description, owner);
            }
        }
    };
})();
export { LinkResolverPlugin };
