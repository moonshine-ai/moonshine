import ts from "typescript";
import type { Logger } from "./loggers.js";
import type { Options } from "./options/index.js";
/**
 * Defines how entry points are interpreted.
 * @enum
 */
export declare const EntryPointStrategy: {
    /**
     * The default behavior in v0.22+, expects all provided entry points as being part of a single program.
     * Any directories included in the entry point list will result in `dir/index.([cm][tj]s|[tj]sx?)` being used.
     */
    readonly Resolve: "resolve";
    /**
     * The default behavior in v0.21 and earlier. Behaves like the resolve behavior, but will recursively
     * expand directories into an entry point for each file within the directory.
     */
    readonly Expand: "expand";
    /**
     * Run TypeDoc in each directory passed as an entry point. Once all directories have been converted,
     * use the merge option to produce final output.
     */
    readonly Packages: "packages";
    /**
     * Merges multiple previously generated output from TypeDoc's --json output together into a single project.
     */
    readonly Merge: "merge";
};
export type EntryPointStrategy = (typeof EntryPointStrategy)[keyof typeof EntryPointStrategy];
export interface DocumentationEntryPoint {
    displayName: string;
    program: ts.Program;
    sourceFile: ts.SourceFile;
}
export interface DocumentEntryPoint {
    displayName: string;
    path: string;
}
export declare function inferEntryPoints(logger: Logger, options: Options): DocumentationEntryPoint[];
export declare function getEntryPoints(logger: Logger, options: Options): DocumentationEntryPoint[] | undefined;
/**
 * Document entry points are markdown documents that the user has requested we include in the project with
 * an option rather than a `@document` tag.
 *
 * @returns A list of `.md` files to include in the documentation as documents.
 */
export declare function getDocumentEntryPoints(logger: Logger, options: Options): DocumentEntryPoint[];
export declare function getWatchEntryPoints(logger: Logger, options: Options, program: ts.Program): DocumentationEntryPoint[] | undefined;
export declare function getPackageDirectories(logger: Logger, options: Options, packageGlobPaths: string[]): string[];
export declare function getExpandedEntryPointsForPaths(logger: Logger, inputFiles: string[], options: Options, programs?: ts.Program[]): DocumentationEntryPoint[];
