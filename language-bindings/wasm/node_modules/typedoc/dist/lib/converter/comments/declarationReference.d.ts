/**
 * Parser for declaration references, see the [TSDoc grammar](https://github.com/microsoft/tsdoc/blob/main/tsdoc/src/beta/DeclarationReference.grammarkdown)
 * for reference. TypeDoc **does not** support the full grammar today. This is intentional, since the TSDoc
 * specified grammar allows the user to construct nonsensical declaration references such as `abc![def!ghi]`
 *
 * @module
 */
export declare const MeaningKeywords: readonly ["class", "interface", "type", "enum", "namespace", "function", "var", "constructor", "member", "event", "call", "new", "index", "complex", "getter", "setter"];
export type MeaningKeyword = (typeof MeaningKeywords)[number];
export interface DeclarationReference {
    resolutionStart: "global" | "local";
    moduleSource?: string;
    symbolReference?: SymbolReference;
}
export interface Meaning {
    keyword?: MeaningKeyword;
    label?: string;
    index?: number;
}
export declare function meaningToString(meaning: Meaning): string;
export interface SymbolReference {
    path?: ComponentPath[];
    meaning?: Meaning;
}
export interface ComponentPath {
    /**
     * How to resolve the `path`
     * - `.` - Navigate via `exports` of symbol
     * - `#` - Navigate via `members` of symbol
     * - `~` - Navigate via `locals` of symbol (note: TypeDoc does not support
     *   locals, see the declaration reference docs)
     */
    navigation: "." | "#" | "~";
    path: string;
}
export declare function parseString(source: string, pos: number, end: number): [string, number] | undefined;
export declare function parseModuleSource(source: string, pos: number, end: number): [string, number] | undefined;
export declare function parseSymbolReference(source: string, pos: number, end: number): [SymbolReference, number] | undefined;
export declare function parseComponent(source: string, pos: number, end: number): [string, number] | undefined;
export declare function parseComponentPath(source: string, pos: number, end: number): readonly [ComponentPath[], number] | undefined;
export declare function parseMeaning(source: string, pos: number, end: number): [Meaning, number] | undefined;
export declare function parseDeclarationReference(source: string, pos: number, end: number): [DeclarationReference, number] | undefined;
