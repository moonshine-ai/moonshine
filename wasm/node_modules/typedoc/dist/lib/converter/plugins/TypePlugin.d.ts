import { DeclarationReflection } from "../../models/reflections/index.js";
import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * Responsible for adding `implementedBy` / `implementedFrom`
 */
export declare class TypePlugin extends ConverterComponent {
    reflections: Set<DeclarationReflection>;
    constructor(owner: Converter);
    private onRevive;
    private onResolve;
    private resolve;
    private postpone;
    private onResolveEnd;
    private finishResolve;
}
