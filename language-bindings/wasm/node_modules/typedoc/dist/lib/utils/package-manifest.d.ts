import type { Logger } from "./loggers.js";
import type { Minimatch } from "minimatch";
/**
 * Loads a package.json and validates that it is a JSON Object
 */
export declare function loadPackageManifest(logger: Logger, packageJsonPath: string): Record<string, unknown> | undefined;
/**
 * Given a list of (potentially wildcard containing) package paths,
 * return all the actual package folders found.
 */
export declare function expandPackages(logger: Logger, packageJsonDir: string, workspaces: string[], exclude: Minimatch[]): string[];
