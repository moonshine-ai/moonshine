import ts from "typescript";
import type { MinimalSourceFile } from "./minimalSourceFile.js";
import type { TranslatedString, TranslationProxy } from "../internationalization/internationalization.js";
import type { IfInternal } from "./index.js";
/**
 * List of known log levels. Used to specify the urgency of a log message.
 */
export declare enum LogLevel {
    Verbose = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    None = 4
}
/**
 * A logger that will not produce any output.
 *
 * This logger also serves as the base class of other loggers as it implements
 * all the required utility functions.
 */
export declare class Logger {
    /**
     * Translation utility for internationalization.
     * @privateRemarks
     * This is fully initialized by the application during bootstrapping.
     */
    i18n: TranslationProxy;
    /**
     * How many error messages have been logged?
     */
    errorCount: number;
    /**
     * How many warning messages have been logged?
     */
    warningCount: number;
    private seenErrors;
    private seenWarnings;
    /**
     * The minimum logging level to print.
     */
    level: LogLevel;
    /**
     * Has an error been raised through the log method?
     */
    hasErrors(): boolean;
    /**
     * Has a warning been raised through the log method?
     */
    hasWarnings(): boolean;
    /**
     * Reset the error counter.
     */
    resetErrors(): void;
    /**
     * Reset the warning counter.
     */
    resetWarnings(): void;
    /**
     * Log the given verbose message.
     *
     * @param text  The message that should be logged.
     */
    verbose(text: string): void;
    /** Log the given info message. */
    info(text: IfInternal<TranslatedString, string>): void;
    /**
     * Log the given warning.
     *
     * @param text  The warning that should be logged.
     */
    warn(text: IfInternal<TranslatedString, string>, node?: ts.Node): void;
    warn(text: IfInternal<TranslatedString, string>, pos: number, file: MinimalSourceFile): void;
    /**
     * Log the given error.
     *
     * @param text  The error that should be logged.
     */
    error(text: IfInternal<TranslatedString, string>, node?: ts.Node): void;
    error(text: IfInternal<TranslatedString, string>, pos: number, file: MinimalSourceFile): void;
    /**
     * Print a log message.
     *
     * @param _message The message itself.
     * @param level The urgency of the log message.
     */
    log(_message: string, level: LogLevel): void;
    /**
     * Print the given TypeScript log messages.
     *
     * @param diagnostics  The TypeScript messages that should be logged.
     */
    diagnostics(diagnostics: ReadonlyArray<ts.Diagnostic>): void;
    /**
     * Print the given TypeScript log message.
     *
     * @param diagnostic  The TypeScript message that should be logged.
     */
    diagnostic(diagnostic: ts.Diagnostic): void;
    protected addContext(message: string, _level: LogLevel, ..._args: [ts.Node?] | [number, MinimalSourceFile]): string;
}
/**
 * A logger that outputs all messages to the console.
 */
export declare class ConsoleLogger extends Logger {
    /**
     * Print a log message.
     *
     * @param message  The message itself.
     * @param level  The urgency of the log message.
     */
    log(message: string, level: Exclude<LogLevel, LogLevel.None>): void;
    protected addContext(message: string, level: Exclude<LogLevel, LogLevel.None>, ...args: [ts.Node?] | [number, MinimalSourceFile]): string;
}
