import ts from "typescript";
import { ParameterReflection, type Reflection, ReflectionKind, SignatureReflection, TypeParameterReflection } from "../../models/index.js";
import type { Context } from "../context.js";
export declare function createSignature(context: Context, kind: ReflectionKind.CallSignature | ReflectionKind.ConstructorSignature | ReflectionKind.GetSignature | ReflectionKind.SetSignature, signature: ts.Signature, symbol: ts.Symbol | undefined, declaration?: ts.SignatureDeclaration | ts.JSDocSignature): void;
/**
 * Special cased constructor factory for functions tagged with `@class`
 */
export declare function createConstructSignatureWithType(context: Context, signature: ts.Signature, classType: Reflection): void;
export declare function convertParameterNodes(context: Context, sigRef: SignatureReflection, parameters: readonly (ts.JSDocParameterTag | ts.ParameterDeclaration)[]): ParameterReflection[];
export declare function convertTypeParameterNodes(context: Context, parameters: readonly ts.TypeParameterDeclaration[] | undefined): TypeParameterReflection[] | undefined;
export declare function createTypeParamReflection(param: ts.TypeParameterDeclaration, context: Context): TypeParameterReflection;
export declare function convertTemplateParameterNodes(context: Context, nodes: readonly ts.JSDocTemplateTag[] | undefined): TypeParameterReflection[] | undefined;
