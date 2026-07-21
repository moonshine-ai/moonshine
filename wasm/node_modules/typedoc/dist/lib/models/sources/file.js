/**
 * Represents references of reflections to their defining source files.
 *
 * @see {@link DeclarationReflection.sources}
 */
export class SourceReference {
    /**
     * The filename of the source file.
     */
    fileName;
    /**
     * The absolute filename of the source file.
     * @internal
     */
    fullFileName;
    /**
     * The one based number of the line that emitted the declaration.
     */
    line;
    /**
     * The index of the character that emitted the declaration.
     */
    character;
    /**
     * URL for displaying the source file.
     */
    url;
    constructor(fileName, line, character) {
        this.fileName = fileName;
        this.fullFileName = fileName;
        this.line = line;
        this.character = character;
    }
    equals(other) {
        return (this.fullFileName == other.fullFileName &&
            this.line === other.line &&
            this.character === other.character);
    }
    toObject() {
        return {
            fileName: this.fileName,
            line: this.line,
            character: this.character,
            url: this.url,
        };
    }
    fromObject(_de, obj) {
        this.url = obj.url;
    }
}
