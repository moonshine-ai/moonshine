import { existsSync } from "fs";
import { isAbsolute, join, relative, resolve } from "path";
import ts from "typescript";
import { findPackageForPath, getCommonDirectory, readFile, } from "../../utils/fs.js";
import { normalizePath } from "../../utils/paths.js";
import { getQualifiedName } from "../../utils/tsutils.js";
import { optional, validate } from "../../utils/validation.js";
import { splitUnquotedString } from "./utils.js";
let transientCount = 0;
const transientIds = new WeakMap();
/**
 * This exists so that TypeDoc can store a unique identifier for a `ts.Symbol` without
 * keeping a reference to the `ts.Symbol` itself. This identifier should be stable across
 * runs so long as the symbol is exported from the same file.
 */
export class ReflectionSymbolId {
    fileName;
    qualifiedName;
    /**
     * Note: This is **not** serialized. It exists for sorting by declaration order, but
     * should not be needed when deserializing from JSON.
     * Will be set to `Infinity` if the ID was deserialized from JSON.
     */
    pos;
    /**
     * Note: This is **not** serialized. It exists to support detection of the differences between
     * symbols which share declarations, but are instantiated with different type parameters.
     * This will be `NaN` if the symbol reference is not transient.
     * Note: This can only be non-NaN if {@link pos} is finite.
     */
    transientId;
    constructor(symbol, declaration) {
        if ("name" in symbol) {
            declaration ??= symbol.declarations?.[0];
            this.fileName = normalizePath(declaration?.getSourceFile().fileName ?? "");
            if (symbol.declarations?.some(ts.isSourceFile)) {
                this.qualifiedName = "";
            }
            else {
                this.qualifiedName = getQualifiedName(symbol, symbol.name);
            }
            this.pos = declaration?.getStart() ?? Infinity;
            if (symbol.flags & ts.SymbolFlags.Transient) {
                this.transientId = transientIds.get(symbol) ?? ++transientCount;
                transientIds.set(symbol, this.transientId);
            }
            else {
                this.transientId = NaN;
            }
        }
        else {
            this.fileName = symbol.sourceFileName;
            this.qualifiedName = symbol.qualifiedName;
            this.pos = Infinity;
            this.transientId = NaN;
        }
    }
    getStableKey() {
        if (Number.isFinite(this.pos)) {
            return `${this.fileName}\0${this.qualifiedName}\0${this.pos}\0${this.transientId}`;
        }
        else {
            return `${this.fileName}\0${this.qualifiedName}`;
        }
    }
    toDeclarationReference() {
        return {
            resolutionStart: "global",
            moduleSource: findPackageForPath(this.fileName),
            symbolReference: {
                path: splitUnquotedString(this.qualifiedName, ".").map((path) => ({
                    navigation: ".",
                    path,
                })),
            },
        };
    }
    toObject(serializer) {
        const sourceFileName = isAbsolute(this.fileName)
            ? normalizePath(relative(serializer.projectRoot, resolveDeclarationMaps(this.fileName)))
            : this.fileName;
        return {
            sourceFileName,
            qualifiedName: this.qualifiedName,
        };
    }
}
const declarationMapCache = new Map();
function resolveDeclarationMaps(file) {
    if (!/\.d\.[cm]?ts$/.test(file))
        return file;
    if (declarationMapCache.has(file))
        return declarationMapCache.get(file);
    const mapFile = file + ".map";
    if (!existsSync(mapFile))
        return file;
    let sourceMap;
    try {
        sourceMap = JSON.parse(readFile(mapFile));
    }
    catch {
        return file;
    }
    if (validate({
        file: String,
        sourceRoot: optional(String),
        sources: [Array, String],
    }, sourceMap)) {
        // There's a pretty large assumption in here that we only have
        // 1 source file per js file. This is a pretty standard typescript approach,
        // but people might do interesting things with transpilation that could break this.
        let source = sourceMap.sources[0];
        // If we have a sourceRoot, trim any leading slash from the source, and join them
        // Similar to how it's done at https://github.com/mozilla/source-map/blob/58819f09018d56ef84dc41ba9c93f554e0645169/lib/util.js#L412
        if (sourceMap.sourceRoot !== undefined) {
            source = source.replace(/^\//, "");
            source = join(sourceMap.sourceRoot, source);
        }
        const result = resolve(mapFile, "..", source);
        declarationMapCache.set(file, result);
        return result;
    }
    return file;
}
// See also: inferEntryPoints in entry-point.ts
export function addInferredDeclarationMapPaths(opts, files) {
    const rootDir = opts.rootDir || getCommonDirectory(files);
    const declDir = opts.declarationDir || opts.outDir || rootDir;
    for (const file of files) {
        const mapFile = normalizePath(resolve(declDir, relative(rootDir, file)).replace(/\.([cm]?[tj]s)x?$/, ".d.$1"));
        declarationMapCache.set(mapFile, file);
    }
}
