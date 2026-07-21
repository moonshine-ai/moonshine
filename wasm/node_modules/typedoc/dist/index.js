/**
 * @module TypeDoc API
 *
 * In addition to the members documented here, TypeDoc exports a `typedoc/debug`
 * entry point which exports some functions which may be useful during plugin
 * development or debugging. Exports from that entry point are **not stable**
 * and may change or be removed at any time.
 */
export { Application } from "./lib/application.js";
export { EventDispatcher } from "./lib/utils/events.js";
export { resetReflectionID } from "./lib/models/reflections/abstract.js";
/**
 * All symbols documented under the Models namespace are also available in the root import.
 *
 * @categoryDescription Types
 * Describes a TypeScript type.
 *
 * @categoryDescription Reflections
 * Describes a documentation entry. The root entry is a {@link ProjectReflection}
 * and contains {@link DeclarationReflection} instances.
 *
 * @summary
 * TypeDoc converts source code into these object types.
 */
export * as Models from "./lib/models/index.js";
/**
 * All symbols documented under the Configuration namespace are also available in the root import.
 * @summary
 * Controls how TypeDoc reads option files and what options are available.
 */
export * as Configuration from "./lib/utils/options/index.js";
export * from "./lib/models/index.js";
export { Converter, Context, } from "./lib/converter/index.js";
export { Renderer, DefaultTheme, DefaultThemeRenderContext, Slugger, UrlMapping, Theme, PageEvent, RendererEvent, MarkdownEvent, IndexEvent, } from "./lib/output/index.js";
export { Outputs } from "./lib/output/output.js";
export { ArgumentsReader, Option, CommentStyle, JSX, LogLevel, Logger, Options, OptionDefaults, PackageJsonReader, ParameterHint, ParameterType, TSConfigReader, TypeDocReader, EntryPointStrategy, EventHooks, MinimalSourceFile, normalizePath, } from "./lib/utils/index.js";
export { JSONOutput, Serializer, Deserializer, SerializeEvent, } from "./lib/serialization/index.js";
export * as Internationalization from "./lib/internationalization/index.js";
import TypeScript from "typescript";
export { TypeScript };
