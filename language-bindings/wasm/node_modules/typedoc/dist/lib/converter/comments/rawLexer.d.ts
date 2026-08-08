import { type Token } from "./lexer.js";
/**
 * Note: This lexer intentionally *only* recognizes inline tags and code blocks.
 * This is because it is intended for use on markdown documents, and we shouldn't
 * take some stray `@user` mention within a "Thanks" section of someone's changelog
 * as starting a block!
 */
export declare function lexCommentString(file: string): Generator<Token, undefined, undefined>;
