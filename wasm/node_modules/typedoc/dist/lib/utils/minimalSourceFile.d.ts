import type { LineAndCharacter, SourceFileLike } from "typescript";
export declare class MinimalSourceFile implements SourceFileLike {
    readonly fileName: string;
    readonly text: string;
    constructor(text: string, fileName: string);
    getLineAndCharacterOfPosition(pos: number): LineAndCharacter;
}
