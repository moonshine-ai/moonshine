var __esDecorate = (this && this.__esDecorate) || function (ctor, descriptorIn, decorators, contextIn, initializers, extraInitializers) {
    function accept(f) { if (f !== void 0 && typeof f !== "function") throw new TypeError("Function expected"); return f; }
    var kind = contextIn.kind, key = kind === "getter" ? "get" : kind === "setter" ? "set" : "value";
    var target = !descriptorIn && ctor ? contextIn["static"] ? ctor : ctor.prototype : null;
    var descriptor = descriptorIn || (target ? Object.getOwnPropertyDescriptor(target, contextIn.name) : {});
    var _, done = false;
    for (var i = decorators.length - 1; i >= 0; i--) {
        var context = {};
        for (var p in contextIn) context[p] = p === "access" ? {} : contextIn[p];
        for (var p in contextIn.access) context.access[p] = contextIn.access[p];
        context.addInitializer = function (f) { if (done) throw new TypeError("Cannot add initializers after decoration has completed"); extraInitializers.push(accept(f || null)); };
        var result = (0, decorators[i])(kind === "accessor" ? { get: descriptor.get, set: descriptor.set } : descriptor[key], context);
        if (kind === "accessor") {
            if (result === void 0) continue;
            if (result === null || typeof result !== "object") throw new TypeError("Object expected");
            if (_ = accept(result.get)) descriptor.get = _;
            if (_ = accept(result.set)) descriptor.set = _;
            if (_ = accept(result.init)) initializers.unshift(_);
        }
        else if (_ = accept(result)) {
            if (kind === "field") initializers.unshift(_);
            else descriptor[key] = _;
        }
    }
    if (target) Object.defineProperty(target, contextIn.name, descriptor);
    done = true;
};
var __runInitializers = (this && this.__runInitializers) || function (thisArg, initializers, value) {
    var useValue = arguments.length > 2;
    for (var i = 0; i < initializers.length; i++) {
        value = useValue ? initializers[i].call(thisArg, value) : initializers[i].call(thisArg);
    }
    return useValue ? value : void 0;
};
import { spawnSync } from "child_process";
import { normalizePath } from "../../utils/index.js";
import { NonEnumerable } from "../../utils/general.js";
import { dirname, join } from "path";
import { existsSync } from "fs";
const TEN_MEGABYTES = 1024 * 10000;
function git(...args) {
    return spawnSync("git", args, {
        encoding: "utf-8",
        windowsHide: true,
        maxBuffer: TEN_MEGABYTES,
    });
}
let haveGit;
export function gitIsInstalled() {
    haveGit ??= git("--version").status === 0;
    return haveGit;
}
export class AssumedRepository {
    path;
    gitRevision;
    sourceLinkTemplate;
    constructor(path, gitRevision, sourceLinkTemplate) {
        this.path = path;
        this.gitRevision = gitRevision;
        this.sourceLinkTemplate = sourceLinkTemplate;
    }
    getURL(fileName, line) {
        const replacements = {
            gitRevision: this.gitRevision,
            "gitRevision:short": this.gitRevision.substring(0, 8),
            path: fileName.substring(this.path.length + 1),
            line,
        };
        return this.sourceLinkTemplate.replace(/\{(gitRevision|gitRevision:short|path|line)\}/g, (_, key) => replacements[key]);
    }
}
/**
 * Stores data of a repository.
 */
let GitRepository = (() => {
    let _files_decorators;
    let _files_initializers = [];
    let _files_extraInitializers = [];
    return class GitRepository {
        static {
            const _metadata = typeof Symbol === "function" && Symbol.metadata ? Object.create(null) : void 0;
            _files_decorators = [NonEnumerable];
            __esDecorate(null, null, _files_decorators, { kind: "field", name: "files", static: false, private: false, access: { has: obj => "files" in obj, get: obj => obj.files, set: (obj, value) => { obj.files = value; } }, metadata: _metadata }, _files_initializers, _files_extraInitializers);
            if (_metadata) Object.defineProperty(this, Symbol.metadata, { enumerable: true, configurable: true, writable: true, value: _metadata });
        }
        /**
         * The path of this repository on disk.
         */
        path;
        /**
         * All files tracked by the repository.
         */
        files = __runInitializers(this, _files_initializers, new Set());
        urlTemplate = __runInitializers(this, _files_extraInitializers);
        gitRevision;
        /**
         * Create a new Repository instance.
         *
         * @param path  The root path of the repository.
         */
        constructor(path, gitRevision, urlTemplate) {
            this.path = path;
            this.gitRevision = gitRevision;
            this.urlTemplate = urlTemplate;
            const out = git("-C", path, "ls-files", "-z");
            if (out.status === 0) {
                out.stdout.split("\0").forEach((file) => {
                    if (file !== "") {
                        this.files.add(normalizePath(path + "/" + file));
                    }
                });
            }
        }
        /**
         * Get the URL of the given file on GitHub or Bitbucket.
         *
         * @param fileName  The file whose URL should be determined.
         * @returns A URL pointing to the web preview of the given file or undefined.
         */
        getURL(fileName, line) {
            if (!this.files.has(fileName)) {
                return;
            }
            const replacements = {
                gitRevision: this.gitRevision,
                "gitRevision:short": this.gitRevision.substring(0, 8),
                path: fileName.substring(this.path.length + 1),
                line,
            };
            return this.urlTemplate.replace(/\{(gitRevision|gitRevision:short|path|line)\}/g, (_, key) => replacements[key]);
        }
        /**
         * Try to create a new repository instance.
         *
         * Checks whether the given path is the root of a valid repository and if so
         * creates a new instance of {@link GitRepository}.
         *
         * @param path  The potential repository root.
         * @returns A new instance of {@link GitRepository} or undefined.
         */
        static tryCreateRepository(path, sourceLinkTemplate, gitRevision, gitRemote, logger) {
            gitRevision ||= git("-C", path, "rev-parse", "HEAD").stdout.trim();
            if (!gitRevision)
                return; // Will only happen in a repo with no commits.
            let urlTemplate;
            if (sourceLinkTemplate) {
                urlTemplate = sourceLinkTemplate;
            }
            else {
                const remotesOut = git("-C", path, "remote", "get-url", gitRemote);
                if (remotesOut.status === 0) {
                    urlTemplate = guessSourceUrlTemplate(remotesOut.stdout.split("\n"));
                }
                else {
                    logger.warn(logger.i18n.git_remote_0_not_valid(gitRemote));
                }
            }
            if (!urlTemplate)
                return;
            return new GitRepository(normalizePath(path), gitRevision, urlTemplate);
        }
    };
})();
export { GitRepository };
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
export class RepositoryManager {
    basePath;
    gitRevision;
    gitRemote;
    sourceLinkTemplate;
    disableGit;
    logger;
    cache = new Map();
    assumedRepo;
    constructor(basePath, gitRevision, gitRemote, sourceLinkTemplate, disableGit, logger) {
        this.basePath = basePath;
        this.gitRevision = gitRevision;
        this.gitRemote = gitRemote;
        this.sourceLinkTemplate = sourceLinkTemplate;
        this.disableGit = disableGit;
        this.logger = logger;
        this.assumedRepo = new AssumedRepository(this.basePath, this.gitRevision, this.sourceLinkTemplate);
    }
    /**
     * Check whether the given file is placed inside a repository.
     *
     * @param fileName  The name of the file a repository should be looked for.
     * @returns The found repository info or undefined.
     */
    getRepository(fileName) {
        if (this.disableGit) {
            return this.assumedRepo;
        }
        return this.getRepositoryFolder(normalizePath(dirname(fileName)));
    }
    getRepositoryFolder(dir) {
        if (this.cache.has(dir)) {
            return this.cache.get(dir);
        }
        if (existsSync(join(dir, ".git"))) {
            // This might just be a git repo, or we might be in some self-recursive symlink
            // loop, and the repo is actually somewhere else. Ask Git where the repo actually is.
            const repo = git("-C", dir, "rev-parse", "--show-toplevel");
            if (repo.status === 0) {
                const repoDir = repo.stdout.replace("\n", "");
                // This check is only necessary if we're in a symlink loop, otherwise
                // it will always be true.
                if (!this.cache.has(repoDir)) {
                    this.cache.set(repoDir, GitRepository.tryCreateRepository(repoDir, this.sourceLinkTemplate, this.gitRevision, this.gitRemote, this.logger));
                }
                this.cache.set(dir, this.cache.get(repoDir));
            }
            else {
                // Not a git repo, probably corrupt.
                this.cache.set(dir, undefined);
            }
        }
        else {
            // We may be at the root of the file system, in which case there is no repo.
            this.cache.set(dir, undefined);
            this.cache.set(dir, this.getRepositoryFolder(dirname(dir)));
        }
        return this.cache.get(dir);
    }
}
// Should have three capturing groups:
// 1. hostname
// 2. user
// 3. project
const repoExpressions = [
    /(github(?!.us)(?:\.[a-z]+)*\.[a-z]{2,})[:/]([^/]+)\/(.*)/,
    /(\w+\.githubprivate.com)[:/]([^/]+)\/(.*)/, // GitHub enterprise
    /(\w+\.ghe.com)[:/]([^/]+)\/(.*)/, // GitHub enterprise
    /(\w+\.github.us)[:/]([^/]+)\/(.*)/, // GitHub enterprise
    /(bitbucket.org)[:/]([^/]+)\/(.*)/,
    /(gitlab.com)[:/]([^/]+)\/(.*)/,
];
export function guessSourceUrlTemplate(remotes) {
    let hostname = "";
    let user = "";
    let project = "";
    outer: for (const repoLink of remotes) {
        for (const regex of repoExpressions) {
            const match = regex.exec(repoLink);
            if (match) {
                hostname = match[1];
                user = match[2];
                project = match[3];
                break outer;
            }
        }
    }
    if (!hostname)
        return;
    if (project.endsWith(".git")) {
        project = project.slice(0, -4);
    }
    let sourcePath = "blob";
    let anchorPrefix = "L";
    if (hostname.includes("gitlab")) {
        sourcePath = "-/blob";
    }
    else if (hostname.includes("bitbucket")) {
        sourcePath = "src";
        anchorPrefix = "lines-";
    }
    return `https://${hostname}/${user}/${project}/${sourcePath}/{gitRevision}/{path}#${anchorPrefix}{line}`;
}
