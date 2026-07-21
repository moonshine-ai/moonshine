import ts from "typescript";
import type { Context } from "./context.js";
export declare function convertJsDocAlias(context: Context, symbol: ts.Symbol, declaration: ts.JSDocTypedefTag | ts.JSDocEnumTag, exportSymbol?: ts.Symbol): void;
export declare function convertJsDocCallback(context: Context, symbol: ts.Symbol, declaration: ts.JSDocCallbackTag, exportSymbol?: ts.Symbol): void;
