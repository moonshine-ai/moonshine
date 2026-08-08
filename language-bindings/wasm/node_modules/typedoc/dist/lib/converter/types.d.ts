import ts from "typescript";
import { type SomeType } from "../models/index.js";
import type { Context } from "./context.js";
export interface TypeConverter<TNode extends ts.TypeNode = ts.TypeNode, TType extends ts.Type = ts.Type> {
    kind: TNode["kind"][];
    convert(context: Context, node: TNode): SomeType;
    convertType(context: Context, type: TType, serializedNode: TNode, originalNode: ts.TypeNode | undefined): SomeType;
}
export declare function loadConverters(): void;
export declare function convertType(context: Context, typeOrNode: ts.Type | ts.TypeNode | undefined, maybeNode?: ts.TypeNode): SomeType;
