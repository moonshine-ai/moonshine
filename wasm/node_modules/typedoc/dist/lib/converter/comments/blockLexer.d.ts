import ts from "typescript";
import { type Token } from "./lexer.js";
export declare function lexBlockComment(file: string, pos?: number, end?: number, jsDoc?: ts.JSDoc | undefined, checker?: ts.TypeChecker | undefined): Generator<Token, undefined, undefined>;
