/**
 * Loads and caches the Emscripten module produced by the core build
 * (`moonshine.mjs` + `moonshine.wasm`). Everything else in the binding goes
 * through the singleton returned by {@link loadMoonshineModule}.
 */
import { toMoonshineError } from './errors.js';
let cached;
/**
 * Resolves the Emscripten factory. By default it dynamically imports the
 * sibling `./moonshine.mjs` emitted by the build; callers can inject their own
 * via {@link LoadModuleOptions.factory} for non-standard bundling.
 */
async function resolveFactory(options) {
    if (options.factory)
        return options.factory;
    // The generated ES module lives next to this file after bundling.
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore - generated at build time, no types.
    const mod = await import('./moonshine.mjs');
    return (mod.default ?? mod);
}
/**
 * Loads (and memoizes) the Moonshine WASM module. Safe to call repeatedly; the
 * heavy compile happens once.
 */
export function loadMoonshineModule(options = {}) {
    if (!cached) {
        cached = (async () => {
            try {
                const factory = await resolveFactory(options);
                const moduleArgs = {};
                if (options.locateFile)
                    moduleArgs.locateFile = options.locateFile;
                return await factory(moduleArgs);
            }
            catch (err) {
                cached = undefined; // allow retry on failure
                throw toMoonshineError(err);
            }
        })();
    }
    return cached;
}
/** Clears the cached module (mainly for tests). */
export function resetMoonshineModule() {
    cached = undefined;
}
//# sourceMappingURL=module.js.map