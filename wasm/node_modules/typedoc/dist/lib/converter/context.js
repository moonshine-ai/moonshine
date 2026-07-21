import { ok as assert } from "assert";
import ts from "typescript";
import { ContainerReflection, DeclarationReflection, ReflectionKind, ReflectionFlag, } from "../models/index.js";
import { isNamedNode } from "./utils/nodes.js";
import { ConverterEvents } from "./converter-events.js";
import { resolveAliasedSymbol } from "./utils/symbols.js";
import { getComment, getFileComment, getJsDocComment, getNodeComment, getSignatureComment, } from "./comments/index.js";
import { getHumanName } from "../utils/tsutils.js";
/**
 * The context describes the current state the converter is in.
 */
export class Context {
    /**
     * The converter instance that has created the context.
     */
    converter;
    /**
     * The TypeChecker instance returned by the TypeScript compiler.
     */
    get checker() {
        return this.program.getTypeChecker();
    }
    /**
     * Translation interface for log messages.
     */
    get i18n() {
        return this.converter.application.i18n;
    }
    /**
     * The program currently being converted.
     * Accessing this property will throw if a source file is not currently being converted.
     */
    get program() {
        assert(this._program, "Tried to access Context.program when not converting a source file");
        return this._program;
    }
    _program;
    /**
     * All programs being converted.
     */
    programs;
    /**
     * The project that is currently processed.
     */
    project;
    /**
     * The scope or parent reflection that is currently processed.
     */
    scope;
    convertingTypeNode = false; // Inherited by withScope
    convertingClassOrInterface = false; // Not inherited
    shouldBeStatic = false; // Not inherited
    /**
     * Create a new Context instance.
     *
     * @param converter  The converter instance that has created the context.
     * @internal
     */
    constructor(converter, programs, project, scope = project) {
        this.converter = converter;
        this.programs = programs;
        this.project = project;
        this.scope = scope;
    }
    /** @internal */
    get logger() {
        return this.converter.application.logger;
    }
    /**
     * Return the type declaration of the given node.
     *
     * @param node  The TypeScript node whose type should be resolved.
     * @returns The type declaration of the given node.
     */
    getTypeAtLocation(node) {
        let nodeType;
        try {
            nodeType = this.checker.getTypeAtLocation(node);
        }
        catch {
            // ignore
        }
        if (!nodeType) {
            if (node.symbol) {
                nodeType = this.checker.getDeclaredTypeOfSymbol(node.symbol);
                // The TS types lie due to ts.SourceFile
            }
            else if (node.parent?.symbol) {
                nodeType = this.checker.getDeclaredTypeOfSymbol(node.parent.symbol);
                // The TS types lie due to ts.SourceFile
            }
            else if (node.parent?.parent?.symbol) {
                nodeType = this.checker.getDeclaredTypeOfSymbol(node.parent.parent.symbol);
            }
        }
        return nodeType;
    }
    getSymbolAtLocation(node) {
        let symbol = this.checker.getSymbolAtLocation(node);
        if (!symbol && isNamedNode(node)) {
            symbol = this.checker.getSymbolAtLocation(node.name);
        }
        return symbol;
    }
    expectSymbolAtLocation(node) {
        const symbol = this.getSymbolAtLocation(node);
        if (!symbol) {
            const { line } = ts.getLineAndCharacterOfPosition(node.getSourceFile(), node.pos);
            throw new Error(`Expected a symbol for node with kind ${ts.SyntaxKind[node.kind]} at ${node.getSourceFile().fileName}:${line + 1}`);
        }
        return symbol;
    }
    resolveAliasedSymbol(symbol) {
        return resolveAliasedSymbol(symbol, this.checker);
    }
    createDeclarationReflection(kind, symbol, exportSymbol, 
    // We need this because modules don't always have symbols.
    nameOverride) {
        const name = getHumanName(nameOverride ?? exportSymbol?.name ?? symbol?.name ?? "unknown");
        if (this.convertingClassOrInterface) {
            if (kind === ReflectionKind.Function) {
                kind = ReflectionKind.Method;
            }
            if (kind === ReflectionKind.Variable) {
                kind = ReflectionKind.Property;
            }
        }
        const reflection = new DeclarationReflection(name, kind, this.scope);
        this.postReflectionCreation(reflection, symbol, exportSymbol);
        return reflection;
    }
    postReflectionCreation(reflection, symbol, exportSymbol) {
        if (exportSymbol &&
            reflection.kind &
                (ReflectionKind.SomeModule | ReflectionKind.Reference)) {
            reflection.comment = this.getComment(exportSymbol, reflection.kind);
        }
        if (symbol && !reflection.comment) {
            reflection.comment = this.getComment(symbol, reflection.kind);
        }
        if (this.shouldBeStatic) {
            reflection.setFlag(ReflectionFlag.Static);
        }
        if (reflection instanceof DeclarationReflection) {
            reflection.escapedName = symbol?.escapedName;
            this.addChild(reflection);
        }
        if (symbol && this.converter.isExternal(symbol)) {
            reflection.setFlag(ReflectionFlag.External);
        }
        if (exportSymbol) {
            this.registerReflection(reflection, exportSymbol);
        }
        const path = reflection.kindOf(ReflectionKind.Namespace | ReflectionKind.Module)
            ? symbol?.declarations?.find(ts.isSourceFile)?.fileName
            : undefined;
        this.project.registerReflection(reflection, symbol, path);
    }
    finalizeDeclarationReflection(reflection) {
        this.converter.trigger(ConverterEvents.CREATE_DECLARATION, this, reflection);
        if (reflection.kindOf(ReflectionKind.MayContainDocuments)) {
            this.converter.processDocumentTags(reflection, reflection);
        }
    }
    addChild(reflection) {
        if (this.scope instanceof ContainerReflection) {
            this.scope.addChild(reflection);
        }
    }
    shouldIgnore(symbol) {
        return this.converter.shouldIgnore(symbol);
    }
    /**
     * Register a newly generated reflection. All created reflections should be
     * passed to this method to ensure that the project helper functions work correctly.
     *
     * @param reflection  The reflection that should be registered.
     * @param symbol  The symbol the given reflection was resolved from.
     */
    registerReflection(reflection, symbol) {
        this.project.registerReflection(reflection, symbol, void 0);
    }
    /** @internal */
    setActiveProgram(program) {
        this._program = program;
    }
    getComment(symbol, kind) {
        return getComment(symbol, kind, this.converter.config, this.logger, this.checker, this.project.files);
    }
    getNodeComment(node, moduleComment) {
        return getNodeComment(node, moduleComment, this.converter.config, this.logger, this.checker, this.project.files);
    }
    getFileComment(node) {
        return getFileComment(node, this.converter.config, this.logger, this.checker, this.project.files);
    }
    getJsDocComment(declaration) {
        return getJsDocComment(declaration, this.converter.config, this.logger, this.checker, this.project.files);
    }
    getSignatureComment(declaration) {
        return getSignatureComment(declaration, this.converter.config, this.logger, this.checker, this.project.files);
    }
    withScope(scope) {
        const context = new Context(this.converter, this.programs, this.project, scope);
        context.convertingTypeNode = this.convertingTypeNode;
        context.setActiveProgram(this._program);
        return context;
    }
}
