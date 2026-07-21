import { type ProjectReflection, type ReferenceType, Reflection } from "../models/index.js";
export declare function discoverAllReferenceTypes(project: ProjectReflection, forExportValidation: boolean): {
    type: ReferenceType;
    owner: Reflection;
}[];
