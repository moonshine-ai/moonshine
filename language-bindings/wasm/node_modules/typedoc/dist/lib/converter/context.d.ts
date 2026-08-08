import ts from "typescript";
import { type Reflection, type ProjectReflection, DeclarationReflection, type DocumentReflection, ReflectionKind } from "../models/index.js";
import type { Converter } from "./converter.js";
import type { TranslationProxy } from "../internationalization/internationalization.js";
/**
 * The context describes the current state the converter is in.
 */
export declare class Context {
    /**
     * The converter instance that has created the context.
     */
    readonly converter: Converter;
    /**
     * The TypeChecker instance returned by the TypeScript compiler.
     */
    get checker(): ts.TypeChecker;
    /**
     * Translation interface for log messages.
     */
    get i18n(): TranslationProxy;
    /**
     * The program currently being converted.
     * Accessing this property will throw if a source file is not currently being converted.
     */
    get program(): ts.Program;
    private _program?;
    /**
     * All programs being converted.
     */
    readonly programs: readonly ts.Program[];
    /**
     * The project that is currently processed.
     */
    readonly project: ProjectReflection;
    /**
     * The scope or parent reflection that is currently processed.
     */
    readonly scope: Reflection;
    convertingTypeNode: boolean;
    convertingClassOrInterface: boolean;
    shouldBeStatic: boolean;
    /**
     * Create a new Context instance.
     *
     * @param converter  The converter instance that has created the context.
     * @internal
     */
    constructor(converter: Converter, programs: readonly ts.Program[], project: ProjectReflection, scope?: Context["scope"]);
    /** @internal */
    get logger(): import("../utils/loggers.js").Logger;
    /**
     * Return the type declaration of the given node.
     *
     * @param node  The TypeScript node whose type should be resolved.
     * @returns The type declaration of the given node.
     */
    getTypeAtLocation(node: ts.Node): ts.Type | undefined;
    getSymbolAtLocation(node: ts.Node): ts.Symbol | undefined;
    expectSymbolAtLocation(node: ts.Node): ts.Symbol;
    resolveAliasedSymbol(symbol: ts.Symbol): ts.Symbol;
    createDeclarationReflection(kind: ReflectionKind, symbol: ts.Symbol | undefined, exportSymbol: ts.Symbol | undefined, nameOverride?: string): DeclarationReflection;
    postReflectionCreation(reflection: Reflection, symbol: ts.Symbol | undefined, exportSymbol: ts.Symbol | undefined): void;
    finalizeDeclarationReflection(reflection: DeclarationReflection): void;
    addChild(reflection: DeclarationReflection | DocumentReflection): void;
    shouldIgnore(symbol: ts.Symbol): boolean;
    /**
     * Register a newly generated reflection. All created reflections should be
     * passed to this method to ensure that the project helper functions work correctly.
     *
     * @param reflection  The reflection that should be registered.
     * @param symbol  The symbol the given reflection was resolved from.
     */
    registerReflection(reflection: Reflection, symbol: ts.Symbol | undefined): void;
    /** @internal */
    setActiveProgram(program: ts.Program | undefined): void;
    getComment(symbol: ts.Symbol, kind: ReflectionKind): import("../models/index.js").Comment | undefined;
    getNodeComment(node: ts.Node, moduleComment: boolean): import("../models/index.js").Comment | undefined;
    getFileComment(node: ts.SourceFile): import("../models/index.js").Comment | undefined;
    getJsDocComment(declaration: ts.JSDocPropertyLikeTag | ts.JSDocCallbackTag | ts.JSDocTypedefTag | ts.JSDocTemplateTag | ts.JSDocEnumTag): import("../models/index.js").Comment | undefined;
    getSignatureComment(declaration: ts.SignatureDeclaration | ts.JSDocSignature): import("../models/index.js").Comment | undefined;
    withScope(scope: Reflection): Context;
}
