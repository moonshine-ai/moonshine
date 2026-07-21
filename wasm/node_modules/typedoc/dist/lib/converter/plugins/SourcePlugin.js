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
import ts from "typescript";
import { DeclarationReflection, SignatureReflection, } from "../../models/reflections/index.js";
import { ConverterComponent } from "../components.js";
import { Option, normalizePath, getCommonDirectory, } from "../../utils/index.js";
import { isNamedNode } from "../utils/nodes.js";
import { relative } from "path";
import { SourceReference } from "../../models/index.js";
import { gitIsInstalled, RepositoryManager } from "../utils/repository.js";
import { ConverterEvents } from "../converter-events.js";
/**
 * A handler that attaches source file information to reflections.
 */
let SourcePlugin = (() => {
    let _classSuper = ConverterComponent;
    let _disableSources_decorators;
    let _disableSources_initializers = [];
    let _disableSources_extraInitializers = [];
    let _gitRevision_decorators;
    let _gitRevision_initializers = [];
    let _gitRevision_extraInitializers = [];
    let _gitRemote_decorators;
    let _gitRemote_initializers = [];
    let _gitRemote_extraInitializers = [];
    let _disableGit_decorators;
    let _disableGit_initializers = [];
    let _disableGit_extraInitializers = [];
    let _sourceLinkTemplate_decorators;
    let _sourceLinkTemplate_initializers = [];
    let _sourceLinkTemplate_extraInitializers = [];
    let _basePath_decorators;
    let _basePath_initializers = [];
    let _basePath_extraInitializers = [];
    return class SourcePlugin extends _classSuper {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(_classSuper[Symbol.metadata] ?? null) : void 0;
            _disableSources_decorators = [Option("disableSources")];
            _gitRevision_decorators = [Option("gitRevision")];
            _gitRemote_decorators = [Option("gitRemote")];
            _disableGit_decorators = [Option("disableGit")];
            _sourceLinkTemplate_decorators = [Option("sourceLinkTemplate")];
            _basePath_decorators = [Option("basePath")];
            __esDecorate(this, null, _disableSources_decorators, { kind: "accessor", name: "disableSources", static: false, private: false, access: { has: obj => "disableSources" in obj, get: obj => obj.disableSources, set: (obj, value) => { obj.disableSources = value; } }, metadata: _metadata }, _disableSources_initializers, _disableSources_extraInitializers);
            __esDecorate(this, null, _gitRevision_decorators, { kind: "accessor", name: "gitRevision", static: false, private: false, access: { has: obj => "gitRevision" in obj, get: obj => obj.gitRevision, set: (obj, value) => { obj.gitRevision = value; } }, metadata: _metadata }, _gitRevision_initializers, _gitRevision_extraInitializers);
            __esDecorate(this, null, _gitRemote_decorators, { kind: "accessor", name: "gitRemote", static: false, private: false, access: { has: obj => "gitRemote" in obj, get: obj => obj.gitRemote, set: (obj, value) => { obj.gitRemote = value; } }, metadata: _metadata }, _gitRemote_initializers, _gitRemote_extraInitializers);
            __esDecorate(this, null, _disableGit_decorators, { kind: "accessor", name: "disableGit", static: false, private: false, access: { has: obj => "disableGit" in obj, get: obj => obj.disableGit, set: (obj, value) => { obj.disableGit = value; } }, metadata: _metadata }, _disableGit_initializers, _disableGit_extraInitializers);
            __esDecorate(this, null, _sourceLinkTemplate_decorators, { kind: "accessor", name: "sourceLinkTemplate", static: false, private: false, access: { has: obj => "sourceLinkTemplate" in obj, get: obj => obj.sourceLinkTemplate, set: (obj, value) => { obj.sourceLinkTemplate = value; } }, metadata: _metadata }, _sourceLinkTemplate_initializers, _sourceLinkTemplate_extraInitializers);
            __esDecorate(this, null, _basePath_decorators, { kind: "accessor", name: "basePath", static: false, private: false, access: { has: obj => "basePath" in obj, get: obj => obj.basePath, set: (obj, value) => { obj.basePath = value; } }, metadata: _metadata }, _basePath_initializers, _basePath_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        #disableSources_accessor_storage = __runInitializers(this, _disableSources_initializers, void 0);
        get disableSources() { return this.#disableSources_accessor_storage; }
        set disableSources(value) { this.#disableSources_accessor_storage = value; }
        #gitRevision_accessor_storage = (__runInitializers(this, _disableSources_extraInitializers), __runInitializers(this, _gitRevision_initializers, void 0));
        get gitRevision() { return this.#gitRevision_accessor_storage; }
        set gitRevision(value) { this.#gitRevision_accessor_storage = value; }
        #gitRemote_accessor_storage = (__runInitializers(this, _gitRevision_extraInitializers), __runInitializers(this, _gitRemote_initializers, void 0));
        get gitRemote() { return this.#gitRemote_accessor_storage; }
        set gitRemote(value) { this.#gitRemote_accessor_storage = value; }
        #disableGit_accessor_storage = (__runInitializers(this, _gitRemote_extraInitializers), __runInitializers(this, _disableGit_initializers, void 0));
        get disableGit() { return this.#disableGit_accessor_storage; }
        set disableGit(value) { this.#disableGit_accessor_storage = value; }
        #sourceLinkTemplate_accessor_storage = (__runInitializers(this, _disableGit_extraInitializers), __runInitializers(this, _sourceLinkTemplate_initializers, void 0));
        get sourceLinkTemplate() { return this.#sourceLinkTemplate_accessor_storage; }
        set sourceLinkTemplate(value) { this.#sourceLinkTemplate_accessor_storage = value; }
        #basePath_accessor_storage = (__runInitializers(this, _sourceLinkTemplate_extraInitializers), __runInitializers(this, _basePath_initializers, void 0));
        get basePath() { return this.#basePath_accessor_storage; }
        set basePath(value) { this.#basePath_accessor_storage = value; }
        /**
         * All file names to find the base path from.
         */
        fileNames = (__runInitializers(this, _basePath_extraInitializers), new Set());
        repositories;
        constructor(owner) {
            super(owner);
            this.owner.on(ConverterEvents.END, this.onEnd.bind(this));
            this.owner.on(ConverterEvents.CREATE_DECLARATION, this.onDeclaration.bind(this));
            this.owner.on(ConverterEvents.CREATE_SIGNATURE, this.onSignature.bind(this));
            this.owner.on(ConverterEvents.RESOLVE_BEGIN, this.onBeginResolve.bind(this));
        }
        onEnd() {
            // Should probably clear repositories/ignoredPaths here, but these aren't likely to change between runs...
            this.fileNames.clear();
        }
        /**
         * Triggered when the converter has created a declaration reflection.
         *
         * Attach the current source file to the {@link DeclarationReflection.sources} array.
         *
         * @param _context  The context object describing the current state the converter is in.
         * @param reflection  The reflection that is currently processed.
         */
        onDeclaration(_context, reflection) {
            if (this.disableSources)
                return;
            const symbol = reflection.project.getSymbolFromReflection(reflection);
            for (const node of symbol?.declarations || []) {
                const sourceFile = node.getSourceFile();
                const fileName = normalizePath(sourceFile.fileName);
                this.fileNames.add(fileName);
                let position;
                if (ts.isSourceFile(node)) {
                    position = { character: 0, line: 0 };
                }
                else {
                    position = ts.getLineAndCharacterOfPosition(sourceFile, getLocationNode(node).getStart());
                }
                reflection.sources ||= [];
                reflection.sources.push(new SourceReference(fileName, position.line + 1, position.character));
            }
        }
        onSignature(_context, reflection, sig) {
            if (this.disableSources || !sig)
                return;
            const sourceFile = sig.getSourceFile();
            const fileName = normalizePath(sourceFile.fileName);
            this.fileNames.add(fileName);
            const position = ts.getLineAndCharacterOfPosition(sourceFile, getLocationNode(sig).getStart());
            reflection.sources ||= [];
            reflection.sources.push(new SourceReference(fileName, position.line + 1, position.character));
        }
        /**
         * Triggered when the converter begins resolving a project.
         *
         * @param context  The context object describing the current state the converter is in.
         */
        onBeginResolve(context) {
            if (this.disableSources)
                return;
            if (this.disableGit && !this.sourceLinkTemplate) {
                this.application.logger.error(context.i18n.disable_git_set_but_not_source_link_template());
                return;
            }
            if (this.disableGit &&
                this.sourceLinkTemplate.includes("{gitRevision}") &&
                !this.gitRevision) {
                this.application.logger.warn(context.i18n.disable_git_set_and_git_revision_used());
            }
            const basePath = this.basePath || getCommonDirectory([...this.fileNames]);
            this.repositories ||= new RepositoryManager(basePath, this.gitRevision, this.gitRemote, this.sourceLinkTemplate, this.disableGit, this.application.logger);
            for (const id in context.project.reflections) {
                const refl = context.project.reflections[id];
                if (!(refl instanceof DeclarationReflection ||
                    refl instanceof SignatureReflection)) {
                    continue;
                }
                if (replaceSourcesWithParentSources(refl)) {
                    refl.sources = refl.parent.sources;
                }
                for (const source of refl.sources || []) {
                    if (this.disableGit || gitIsInstalled()) {
                        const repo = this.repositories.getRepository(source.fullFileName);
                        source.url = repo?.getURL(source.fullFileName, source.line);
                    }
                    source.fileName = normalizePath(relative(basePath, source.fullFileName));
                }
            }
        }
    };
})();
export { SourcePlugin };
function getLocationNode(node) {
    if (isNamedNode(node))
        return node.name;
    return node;
}
function replaceSourcesWithParentSources(refl) {
    if (refl instanceof DeclarationReflection || !refl.sources) {
        return false;
    }
    const symbol = refl.project.getSymbolFromReflection(refl.parent);
    if (!symbol?.declarations) {
        return false;
    }
    for (const decl of symbol.declarations) {
        const file = decl.getSourceFile();
        const pos = file.getLineAndCharacterOfPosition(decl.pos);
        const end = file.getLineAndCharacterOfPosition(decl.end);
        if (refl.sources.some((src) => src.fullFileName === file.fileName &&
            pos.line <= src.line - 1 &&
            src.line - 1 <= end.line)) {
            return false;
        }
    }
    return true;
}
