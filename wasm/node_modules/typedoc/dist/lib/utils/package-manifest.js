// Utilities to support the inspection of node package "manifests"
import { dirname, resolve } from "path";
import { readFile, glob } from "./fs.js";
import { matchesAny, nicePath } from "./paths.js";
/**
 * Helper for the TS type system to understand hasOwnProperty
 * and narrow a type appropriately.
 * @param obj the receiver of the hasOwnProperty method call
 * @param prop the property to test for
 */
function hasOwnProperty(obj, prop) {
    return Object.prototype.hasOwnProperty.call(obj, prop);
}
/**
 * Loads a package.json and validates that it is a JSON Object
 */
export function loadPackageManifest(logger, packageJsonPath) {
    const packageJson = JSON.parse(readFile(packageJsonPath));
    if (typeof packageJson !== "object" || !packageJson) {
        logger.error(logger.i18n.file_0_not_an_object(nicePath(packageJsonPath)));
        return undefined;
    }
    return packageJson;
}
/**
 * Load the paths to packages specified in a Yarn workspace package JSON
 * Returns undefined if packageJSON does not define a Yarn workspace
 * @param packageJSON the package json object
 */
function getPackagePaths(packageJSON) {
    if (Array.isArray(packageJSON["workspaces"]) &&
        packageJSON["workspaces"].every((i) => typeof i === "string")) {
        return packageJSON["workspaces"];
    }
    if (typeof packageJSON["workspaces"] === "object" &&
        packageJSON["workspaces"] != null) {
        const workspaces = packageJSON["workspaces"];
        if (hasOwnProperty(workspaces, "packages") &&
            Array.isArray(workspaces["packages"]) &&
            workspaces["packages"].every((i) => typeof i === "string")) {
            return workspaces["packages"];
        }
    }
    return undefined;
}
/**
 * Given a list of (potentially wildcard containing) package paths,
 * return all the actual package folders found.
 */
export function expandPackages(logger, packageJsonDir, workspaces, exclude) {
    // Technically npm and Yarn workspaces don't support recursive nesting,
    // however we support the passing of paths to either packages or
    // to the root of a workspace tree in our params and so we could here
    // be dealing with either a root or a leaf. So let's do this recursively,
    // as it actually is simpler from an implementation perspective anyway.
    return workspaces.flatMap((workspace) => {
        const expandedPackageJsonPaths = glob(resolve(packageJsonDir, workspace, "package.json"), resolve(packageJsonDir));
        if (expandedPackageJsonPaths.length === 0) {
            logger.warn(logger.i18n.entry_point_0_did_not_match_any_packages(nicePath(workspace)));
        }
        else if (expandedPackageJsonPaths.length !== 1) {
            logger.verbose(`Expanded ${nicePath(workspace)} to:\n\t${expandedPackageJsonPaths
                .map(nicePath)
                .join("\n\t")}`);
        }
        return expandedPackageJsonPaths.flatMap((packageJsonPath) => {
            if (matchesAny(exclude, dirname(packageJsonPath))) {
                return [];
            }
            const packageJson = loadPackageManifest(logger, packageJsonPath);
            if (packageJson === undefined) {
                return [];
            }
            const packagePaths = getPackagePaths(packageJson);
            if (packagePaths === undefined) {
                // Assume this is a single package repo
                return [dirname(packageJsonPath)];
            }
            // This is a workspace root package, recurse
            return expandPackages(logger, dirname(packageJsonPath), packagePaths, exclude);
        });
    });
}
