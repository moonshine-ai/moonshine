import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * A handler that attaches source file information to reflections.
 */
export declare class SourcePlugin extends ConverterComponent {
    accessor disableSources: boolean;
    accessor gitRevision: string;
    accessor gitRemote: string;
    accessor disableGit: boolean;
    accessor sourceLinkTemplate: string;
    accessor basePath: string;
    /**
     * All file names to find the base path from.
     */
    private fileNames;
    private repositories?;
    constructor(owner: Converter);
    private onEnd;
    /**
     * Triggered when the converter has created a declaration reflection.
     *
     * Attach the current source file to the {@link DeclarationReflection.sources} array.
     *
     * @param _context  The context object describing the current state the converter is in.
     * @param reflection  The reflection that is currently processed.
     */
    private onDeclaration;
    private onSignature;
    /**
     * Triggered when the converter begins resolving a project.
     *
     * @param context  The context object describing the current state the converter is in.
     */
    private onBeginResolve;
}
