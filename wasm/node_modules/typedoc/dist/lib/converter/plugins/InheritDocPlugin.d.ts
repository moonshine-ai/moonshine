import { ConverterComponent } from "../components.js";
import { type ValidationOptions } from "../../utils/index.js";
import type { Converter } from "../converter.js";
/**
 * A plugin that handles `@inheritDoc` tags by copying documentation from another API item.
 * It is NOT responsible for handling bare JSDoc style `@inheritDoc` tags which do not specify
 * a target to inherit from. Those are handled by the ImplementsPlugin class.
 *
 * What gets copied:
 * - short text
 * - text
 * - `@remarks` block
 * - `@params` block
 * - `@typeParam` block
 * - `@return` block
 */
export declare class InheritDocPlugin extends ConverterComponent {
    accessor validation: ValidationOptions;
    private dependencies;
    /**
     * Create a new InheritDocPlugin instance.
     */
    constructor(owner: Converter);
    /**
     * Traverse through reflection descendant to check for `inheritDoc` tag.
     * If encountered, the parameter of the tag is used to determine a source reflection
     * that will provide actual comment.
     */
    private processInheritDoc;
    private copyComment;
    private createCircularDependencyWarnings;
}
