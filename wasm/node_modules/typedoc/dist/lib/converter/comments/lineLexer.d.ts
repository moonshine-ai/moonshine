import type ts from "typescript";
import { type Token } from "./lexer.js";
export declare function lexLineComments(file: string, ranges: ts.CommentRange[]): Generator<Token, undefined, undefined>;
