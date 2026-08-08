import type { OptionsReader, Options } from "../index.js";
import type { Logger } from "../../loggers.js";
/**
 * Obtains option values from command-line arguments
 */
export declare class ArgumentsReader implements OptionsReader {
    readonly name = "arguments";
    readonly order: number;
    readonly supportsPackages = false;
    private args;
    constructor(priority: number, args?: string[]);
    read(container: Options, logger: Logger): void;
}
