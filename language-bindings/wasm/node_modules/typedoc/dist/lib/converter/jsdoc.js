// Converter functions for JSDoc defined types
// @typedef
// @callback
import { ok } from "assert";
import ts from "typescript";
import { DeclarationReflection, IntrinsicType, ReflectionKind, ReflectionType, SignatureReflection, } from "../models/index.js";
import { ReflectionSymbolId } from "../models/reflections/ReflectionSymbolId.js";
import { ConverterEvents } from "./converter-events.js";
import { convertParameterNodes, convertTemplateParameterNodes, } from "./factories/signature.js";
export function convertJsDocAlias(context, symbol, declaration, exportSymbol) {
    if (declaration.typeExpression &&
        ts.isJSDocTypeLiteral(declaration.typeExpression)) {
        convertJsDocInterface(context, declaration, symbol, exportSymbol);
        return;
    }
    // If the typedef tag is just referring to another type-space symbol, with no type parameters
    // or appropriate forwarding type parameters, then we treat it as a re-export instead of creating
    // a type alias with an import type.
    const aliasedSymbol = getTypedefReExportTarget(context, declaration);
    if (aliasedSymbol) {
        context.converter.convertSymbol(context, aliasedSymbol, exportSymbol ?? symbol);
        return;
    }
    const reflection = context.createDeclarationReflection(ReflectionKind.TypeAlias, symbol, exportSymbol);
    reflection.comment = context.getJsDocComment(declaration);
    reflection.type = context.converter.convertType(context.withScope(reflection), declaration.typeExpression?.type);
    convertTemplateParameters(context.withScope(reflection), declaration.parent);
    context.finalizeDeclarationReflection(reflection);
}
export function convertJsDocCallback(context, symbol, declaration, exportSymbol) {
    const alias = context.createDeclarationReflection(ReflectionKind.TypeAlias, symbol, exportSymbol);
    alias.comment = context.getJsDocComment(declaration);
    context.finalizeDeclarationReflection(alias);
    const ac = context.withScope(alias);
    alias.type = convertJsDocSignature(ac, declaration.typeExpression);
    convertTemplateParameters(ac, declaration.parent);
}
function convertJsDocInterface(context, declaration, symbol, exportSymbol) {
    const reflection = context.createDeclarationReflection(ReflectionKind.Interface, symbol, exportSymbol);
    reflection.comment = context.getJsDocComment(declaration);
    context.finalizeDeclarationReflection(reflection);
    const rc = context.withScope(reflection);
    const type = context.checker.getDeclaredTypeOfSymbol(symbol);
    for (const s of type.getProperties()) {
        context.converter.convertSymbol(rc, s);
    }
    convertTemplateParameters(rc, declaration.parent);
}
function convertJsDocSignature(context, node) {
    const symbol = context.getSymbolAtLocation(node) ?? node.symbol;
    const type = context.getTypeAtLocation(node);
    if (!symbol || !type) {
        return new IntrinsicType("Function");
    }
    const reflection = new DeclarationReflection("__type", ReflectionKind.TypeLiteral, context.scope);
    context.registerReflection(reflection, symbol);
    context.converter.trigger(ConverterEvents.CREATE_DECLARATION, context, reflection);
    const signature = new SignatureReflection("__type", ReflectionKind.CallSignature, reflection);
    context.project.registerSymbolId(signature, new ReflectionSymbolId(symbol, node));
    context.registerReflection(signature, void 0);
    const signatureCtx = context.withScope(signature);
    reflection.signatures = [signature];
    signature.type = context.converter.convertType(signatureCtx, node.type?.typeExpression?.type);
    signature.parameters = convertParameterNodes(signatureCtx, signature, node.parameters);
    signature.typeParameters = convertTemplateParameterNodes(context.withScope(reflection), node.typeParameters);
    return new ReflectionType(reflection);
}
function convertTemplateParameters(context, node) {
    ok(context.scope instanceof DeclarationReflection);
    context.scope.typeParameters = convertTemplateParameterNodes(context, node.tags?.filter(ts.isJSDocTemplateTag));
}
function getTypedefReExportTarget(context, declaration) {
    const typeExpression = declaration.typeExpression;
    if (!ts.isJSDocTypedefTag(declaration) ||
        !typeExpression ||
        ts.isJSDocTypeLiteral(typeExpression) ||
        !ts.isImportTypeNode(typeExpression.type) ||
        !typeExpression.type.qualifier ||
        !ts.isIdentifier(typeExpression.type.qualifier)) {
        return;
    }
    const targetSymbol = context.expectSymbolAtLocation(typeExpression.type.qualifier);
    const decl = targetSymbol.declarations?.[0];
    if (!decl ||
        !(ts.isTypeAliasDeclaration(decl) ||
            ts.isInterfaceDeclaration(decl) ||
            ts.isJSDocTypedefTag(decl) ||
            ts.isJSDocCallbackTag(decl))) {
        return;
    }
    const targetParams = ts.getEffectiveTypeParameterDeclarations(decl);
    const localParams = ts.getEffectiveTypeParameterDeclarations(declaration);
    const localArgs = typeExpression.type.typeArguments || [];
    // If we have type parameters, ensure they are forwarding parameters with no transformations.
    // This doesn't check constraints since they aren't checked in JSDoc types.
    if (targetParams.length !== localParams.length ||
        localArgs.some((arg, i) => !ts.isTypeReferenceNode(arg) ||
            !ts.isIdentifier(arg.typeName) ||
            arg.typeArguments ||
            localParams[i]?.name.text !== arg.typeName.text)) {
        return;
    }
    return targetSymbol;
}
