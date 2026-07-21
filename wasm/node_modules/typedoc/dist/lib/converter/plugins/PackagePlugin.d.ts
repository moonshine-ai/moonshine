import { ConverterComponent } from "../components.js";
import { EntryPointStrategy } from "../../utils/index.js";
import type { Converter } from "../converter.js";
/**
 * A handler that tries to find the package.json and readme.md files of the
 * current project.
 */
export declare class PackagePlugin extends ConverterComponent {
    accessor readme: string;
    accessor entryPointStrategy: EntryPointStrategy;
    accessor entryPoints: string[];
    accessor includeVersion: boolean;
    /**
     * The file name of the found readme.md file.
     */
    private readmeFile?;
    /**
     * Contents of the readme.md file discovered, if any
     */
    private readmeContents?;
    /**
     * Contents of package.json for the active project
     */
    private packageJson?;
    constructor(owner: Converter);
    private onRevive;
    private onBegin;
    private onBeginResolve;
    private addEntries;
}
