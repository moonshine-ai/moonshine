export { Options, Option } from "./options.js";
export type { OptionsReader } from "./options.js";
export { ArgumentsReader, PackageJsonReader, TypeDocReader, TSConfigReader, } from "./readers/index.js";
export { CommentStyle, EmitStrategy, ParameterType, ParameterHint, } from "./declaration.js";
export type { TypeDocOptions, TypeDocOptionMap, ValidationOptions, KeyToDeclaration, DeclarationOption, DeclarationOptionBase, StringDeclarationOption, NumberDeclarationOption, BooleanDeclarationOption, ArrayDeclarationOption, MixedDeclarationOption, ObjectDeclarationOption, MapDeclarationOption, FlagsDeclarationOption, DeclarationOptionToOptionType, TypeDocOptionValues, ParameterTypeToOptionTypeMap, ManuallyValidatedOption, JsDocCompatibility, OutputSpecification, } from "./declaration.js";
export * as OptionDefaults from "./defaults.js";
