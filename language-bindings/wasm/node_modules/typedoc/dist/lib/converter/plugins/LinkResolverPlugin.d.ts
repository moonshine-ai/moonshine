import { ConverterComponent } from "../components.js";
import type { Context, Converter } from "../../converter/index.js";
import { type ValidationOptions } from "../../utils/index.js";
import { type ProjectReflection } from "../../models/index.js";
/**
 * A plugin that resolves `{@link Foo}` tags.
 */
export declare class LinkResolverPlugin extends ConverterComponent {
    accessor validation: ValidationOptions;
    constructor(owner: Converter);
    onResolve(context: Context): void;
    resolveLinks(project: ProjectReflection): void;
    private resolveCategoryLinks;
}
