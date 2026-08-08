import { join, relative, resolve } from "path";
import ts from "typescript";
import * as FS from "fs";
import { expandPackages } from "./package-manifest.js";
import { createMinimatch, matchesAny, nicePath, normalizePath, } from "./paths.js";
import { deriveRootDir, discoverPackageJson, getCommonDirectory, glob, inferPackageEntryPointPaths, isDir, } from "./fs.js";
import { assertNever } from "./general.js";
/**
 * Defines how entry points are interpreted.
 * @enum
 */
export const EntryPointStrategy = {
    /**
     * The default behavior in v0.22+, expects all provided entry points as being part of a single program.
     * Any directories included in the entry point list will result in `dir/index.([cm][tj]s|[tj]sx?)` being used.
     */
    Resolve: "resolve",
    /**
     * The default behavior in v0.21 and earlier. Behaves like the resolve behavior, but will recursively
     * expand directories into an entry point for each file within the directory.
     */
    Expand: "expand",
    /**
     * Run TypeDoc in each directory passed as an entry point. Once all directories have been converted,
     * use the merge option to produce final output.
     */
    Packages: "packages",
    /**
     * Merges multiple previously generated output from TypeDoc's --json output together into a single project.
     */
    Merge: "merge",
};
export function inferEntryPoints(logger, options) {
    const packageJson = discoverPackageJson(options.packageDir ?? process.cwd());
    if (!packageJson) {
        logger.warn(logger.i18n.no_entry_points_provided());
        return [];
    }
    const pathEntries = inferPackageEntryPointPaths(packageJson.file);
    const entryPoints = [];
    const programs = getEntryPrograms(pathEntries.map((p) => p[1]), logger, options);
    // See also: addInferredDeclarationMapPaths in ReflectionSymbolId
    const jsToTsSource = new Map();
    for (const program of programs) {
        const opts = program.getCompilerOptions();
        const rootDir = opts.rootDir || getCommonDirectory(program.getRootFileNames());
        const outDir = opts.outDir || rootDir;
        for (const tsFile of program.getRootFileNames()) {
            const jsFile = normalizePath(resolve(outDir, relative(rootDir, tsFile)).replace(/\.([cm]?)[tj]sx?$/, ".$1js"));
            jsToTsSource.set(jsFile, tsFile);
        }
    }
    for (const [name, path] of pathEntries) {
        // Strip leading ./ from the display name
        const displayName = name.replace(/^\.\/?/, "");
        const targetPath = jsToTsSource.get(path) || path;
        const program = programs.find((p) => p.getSourceFile(targetPath));
        if (program) {
            entryPoints.push({
                displayName,
                program,
                sourceFile: program.getSourceFile(targetPath),
            });
        }
        else if (/\.[cm]?js$/.test(path)) {
            logger.warn(logger.i18n.failed_to_resolve_0_to_ts_path(nicePath(path)));
        }
    }
    if (entryPoints.length === 0) {
        logger.warn(logger.i18n.no_entry_points_provided());
        return [];
    }
    return entryPoints;
}
export function getEntryPoints(logger, options) {
    if (!options.isSet("entryPoints")) {
        logger.warn(logger.i18n.no_entry_points_provided());
        return [];
    }
    const entryPoints = options.getValue("entryPoints");
    const exclude = options.getValue("exclude");
    // May be set explicitly to be an empty array to only include a readme for a package
    // See #2264
    if (entryPoints.length === 0) {
        return [];
    }
    let result;
    const strategy = options.getValue("entryPointStrategy");
    switch (strategy) {
        case EntryPointStrategy.Resolve:
            result = getEntryPointsForPaths(logger, expandGlobs(entryPoints, exclude, logger), options);
            break;
        case EntryPointStrategy.Expand:
            result = getExpandedEntryPointsForPaths(logger, expandGlobs(entryPoints, exclude, logger), options);
            break;
        case EntryPointStrategy.Merge:
        case EntryPointStrategy.Packages:
            // Doesn't really have entry points in the traditional way of how TypeDoc has dealt with them.
            return [];
        default:
            assertNever(strategy);
    }
    if (result.length === 0) {
        logger.error(logger.i18n.unable_to_find_any_entry_points());
        return;
    }
    return result;
}
/**
 * Document entry points are markdown documents that the user has requested we include in the project with
 * an option rather than a `@document` tag.
 *
 * @returns A list of `.md` files to include in the documentation as documents.
 */
export function getDocumentEntryPoints(logger, options) {
    const docGlobs = options.getValue("projectDocuments");
    if (docGlobs.length === 0) {
        return [];
    }
    const docPaths = expandGlobs(docGlobs, [], logger);
    // We might want to expand this in the future, there are quite a lot of extensions
    // that have at some point or another been used for markdown: https://superuser.com/a/285878
    const supportedFileRegex = /\.(md|markdown)$/;
    const expanded = expandInputFiles(logger, docPaths, options, supportedFileRegex);
    const baseDir = options.getValue("basePath") || deriveRootDir(expanded);
    return expanded.map((path) => {
        return {
            displayName: relative(baseDir, path).replace(/\.[^.]+$/, ""),
            path,
        };
    });
}
export function getWatchEntryPoints(logger, options, program) {
    let result;
    const entryPoints = options.getValue("entryPoints");
    const exclude = options.getValue("exclude");
    const strategy = options.getValue("entryPointStrategy");
    switch (strategy) {
        case EntryPointStrategy.Resolve:
            result = getEntryPointsForPaths(logger, expandGlobs(entryPoints, exclude, logger), options, [program]);
            break;
        case EntryPointStrategy.Expand:
            result = getExpandedEntryPointsForPaths(logger, expandGlobs(entryPoints, exclude, logger), options, [program]);
            break;
        case EntryPointStrategy.Packages:
            logger.error(logger.i18n.watch_does_not_support_packages_mode());
            break;
        case EntryPointStrategy.Merge:
            logger.error(logger.i18n.watch_does_not_support_merge_mode());
            break;
        default:
            assertNever(strategy);
    }
    if (result && result.length === 0) {
        logger.error(logger.i18n.unable_to_find_any_entry_points());
        return;
    }
    return result;
}
export function getPackageDirectories(logger, options, packageGlobPaths) {
    const exclude = createMinimatch(options.getValue("exclude"));
    const rootDir = deriveRootDir(packageGlobPaths);
    // packages arguments are workspace tree roots, or glob patterns
    // This expands them to leave only leaf packages
    return expandPackages(logger, rootDir, packageGlobPaths, exclude);
}
function getModuleName(fileName, baseDir) {
    return normalizePath(relative(baseDir, fileName)).replace(/(\/index)?(\.d)?\.([cm][tj]s|[tj]sx?)$/, "");
}
/**
 * Converts a list of file-oriented paths in to DocumentationEntryPoints for conversion.
 * This is in contrast with the package-oriented `getEntryPointsForPackages`
 */
function getEntryPointsForPaths(logger, inputFiles, options, programs = getEntryPrograms(inputFiles, logger, options)) {
    const baseDir = options.getValue("basePath") || deriveRootDir(inputFiles);
    const entryPoints = [];
    let expandSuggestion = true;
    entryLoop: for (const fileOrDir of inputFiles.map(normalizePath)) {
        const toCheck = [fileOrDir];
        if (!/\.([cm][tj]s|[tj]sx?)$/.test(fileOrDir)) {
            toCheck.push(`${fileOrDir}/index.ts`, `${fileOrDir}/index.cts`, `${fileOrDir}/index.mts`, `${fileOrDir}/index.tsx`, `${fileOrDir}/index.js`, `${fileOrDir}/index.cjs`, `${fileOrDir}/index.mjs`, `${fileOrDir}/index.jsx`);
        }
        for (const program of programs) {
            for (const check of toCheck) {
                const sourceFile = program.getSourceFile(check);
                if (sourceFile) {
                    entryPoints.push({
                        displayName: getModuleName(resolve(check), baseDir),
                        sourceFile,
                        program,
                    });
                    continue entryLoop;
                }
            }
        }
        logger.warn(logger.i18n.entry_point_0_not_in_program(nicePath(fileOrDir)));
        if (expandSuggestion && isDir(fileOrDir)) {
            expandSuggestion = false;
            logger.info(logger.i18n.use_expand_or_glob_for_files_in_dir());
        }
    }
    return entryPoints;
}
export function getExpandedEntryPointsForPaths(logger, inputFiles, options, programs = getEntryPrograms(inputFiles, logger, options)) {
    const compilerOptions = options.getCompilerOptions();
    const supportedFileRegex = compilerOptions.allowJs || compilerOptions.checkJs
        ? /\.([cm][tj]s|[tj]sx?)$/
        : /\.([cm]ts|tsx?)$/;
    return getEntryPointsForPaths(logger, expandInputFiles(logger, inputFiles, options, supportedFileRegex), options, programs);
}
function expandGlobs(inputFiles, exclude, logger) {
    const excludePatterns = createMinimatch(exclude);
    const base = deriveRootDir(inputFiles);
    const result = inputFiles.flatMap((entry) => {
        const result = glob(entry, base, {
            includeDirectories: true,
            followSymlinks: true,
        });
        const filtered = result.filter((file) => file === entry || !matchesAny(excludePatterns, file));
        if (result.length === 0) {
            logger.warn(logger.i18n.glob_0_did_not_match_any_files(nicePath(entry)));
        }
        else if (filtered.length === 0) {
            logger.warn(logger.i18n.entry_point_0_did_not_match_any_files_after_exclude(nicePath(entry)));
        }
        else if (filtered.length !== 1) {
            logger.verbose(`Expanded ${nicePath(entry)} to:\n\t${filtered
                .map(nicePath)
                .join("\n\t")}`);
        }
        return filtered;
    });
    return result;
}
function getEntryPrograms(inputFiles, logger, options) {
    const noTsConfigFound = options.getFileNames().length === 0 &&
        options.getProjectReferences().length === 0;
    const rootProgram = noTsConfigFound
        ? ts.createProgram({
            rootNames: inputFiles,
            options: options.getCompilerOptions(),
        })
        : ts.createProgram({
            rootNames: options.getFileNames(),
            options: options.getCompilerOptions(),
            projectReferences: options.getProjectReferences(),
        });
    const programs = [rootProgram];
    // This might be a solution style tsconfig, in which case we need to add a program for each
    // reference so that the converter can look through each of these.
    if (rootProgram.getRootFileNames().length === 0) {
        logger.verbose("tsconfig appears to be a solution style tsconfig - creating programs for references");
        const resolvedReferences = rootProgram.getResolvedProjectReferences();
        for (const ref of resolvedReferences ?? []) {
            if (!ref)
                continue; // This indicates bad configuration... will be reported later.
            programs.push(ts.createProgram({
                options: options.fixCompilerOptions(ref.commandLine.options),
                rootNames: ref.commandLine.fileNames,
                projectReferences: ref.commandLine.projectReferences,
            }));
        }
    }
    return programs;
}
/**
 * Expand a list of input files.
 *
 * Searches for directories in the input files list and replaces them with a
 * listing of all TypeScript files within them. One may use the ```--exclude``` option
 * to filter out files with a pattern.
 *
 * @param inputFiles  The list of files that should be expanded.
 * @returns  The list of input files with expanded directories.
 */
function expandInputFiles(logger, entryPoints, options, supportedFile) {
    const files = [];
    const exclude = createMinimatch(options.getValue("exclude"));
    function add(file, entryPoint) {
        let stats;
        try {
            stats = FS.statSync(file);
        }
        catch {
            // No permission or a symbolic link, do not resolve.
            return;
        }
        const fileIsDir = stats.isDirectory();
        if (fileIsDir && !file.endsWith("/")) {
            file = `${file}/`;
        }
        if (fileIsDir) {
            FS.readdirSync(file).forEach((next) => {
                add(join(file, next), false);
            });
        }
        else if (supportedFile.test(file)) {
            if (!entryPoint && matchesAny(exclude, file)) {
                return;
            }
            files.push(normalizePath(file));
        }
    }
    entryPoints.forEach((file) => {
        const resolved = resolve(file);
        if (!FS.existsSync(resolved)) {
            logger.warn(logger.i18n.entry_point_0_did_not_exist(file));
            return;
        }
        add(resolved, true);
    });
    return files;
}
