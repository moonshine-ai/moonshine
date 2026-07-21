import type { ProjectReflection } from "../models/index.js";
import type { Logger } from "../utils/index.js";
export declare function validateExports(project: ProjectReflection, logger: Logger, intentionallyNotExported: readonly string[]): void;
