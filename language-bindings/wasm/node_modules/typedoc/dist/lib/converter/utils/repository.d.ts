import { type Logger } from "../../utils/index.js";
export declare function gitIsInstalled(): boolean;
export interface Repository {
    readonly path: string;
    getURL(fileName: string, line: number): string | undefined;
}
export declare class AssumedRepository implements Repository {
    readonly path: string;
    readonly gitRevision: string;
    readonly sourceLinkTemplate: string;
    constructor(path: string, gitRevision: string, sourceLinkTemplate: string);
    getURL(fileName: string, line: number): string | undefined;
}
/**
 * Stores data of a repository.
 */
export declare class GitRepository implements Repository {
    /**
     * The path of this repository on disk.
     */
    path: string;
    /**
     * All files tracked by the repository.
     */
    files: Set<string>;
    urlTemplate: string;
    gitRevision: string;
    /**
     * Create a new Repository instance.
     *
     * @param path  The root path of the repository.
     */
    constructor(path: string, gitRevision: string, urlTemplate: string);
    /**
     * Get the URL of the given file on GitHub or Bitbucket.
     *
     * @param fileName  The file whose URL should be determined.
     * @returns A URL pointing to the web preview of the given file or undefined.
     */
    getURL(fileName: string, line: number): string | undefined;
    /**
     * Try to create a new repository instance.
     *
     * Checks whether the given path is the root of a valid repository and if so
     * creates a new instance of {@link GitRepository}.
     *
     * @param path  The potential repository root.
     * @returns A new instance of {@link GitRepository} or undefined.
     */
    static tryCreateRepository(path: string, sourceLinkTemplate: string, gitRevision: string, gitRemote: string, logger: Logger): GitRepository | undefined;
}
/**
 * Responsible for keeping track of 0-N repositories which exist on a machine.
 * This used to be inlined in SourcePlugin, moved out for easy unit testing.
 *
 * Git repositories can be nested. Files should be resolved to a repo as shown
 * below:
 * ```text
 * /project
 * /project/.git (A)
 * /project/file.js (A)
 * /project/folder/file.js (A)
 * /project/sub/.git (B)
 * /project/sub/file.js (B)
 * ```
 *
 * In order words, it is not safe to assume that just because a file is within
 * the `/project` directory, that it belongs to repo `A`. As calling git is
 * expensive (~20-300ms depending on the machine, antivirus, etc.) we check for
 * `.git` folders manually, and only call git if one is found.
 *
 * Symlinked files have the potential to further complicate this. If TypeScript's
 * `preserveSymlinks` option is on, then this may be passed the path to a symlinked
 * file. Unlike TypeScript, we will resolve the path, as the repo link should really
 * point to the actual file.
 */
export declare class RepositoryManager {
    private basePath;
    private gitRevision;
    private gitRemote;
    private sourceLinkTemplate;
    private disableGit;
    private logger;
    private cache;
    private assumedRepo;
    constructor(basePath: string, gitRevision: string, gitRemote: string, sourceLinkTemplate: string, disableGit: boolean, logger: Logger);
    /**
     * Check whether the given file is placed inside a repository.
     *
     * @param fileName  The name of the file a repository should be looked for.
     * @returns The found repository info or undefined.
     */
    getRepository(fileName: string): Repository | undefined;
    private getRepositoryFolder;
}
export declare function guessSourceUrlTemplate(remotes: string[]): string | undefined;
