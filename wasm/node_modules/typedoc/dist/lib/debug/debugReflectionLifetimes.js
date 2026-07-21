import { ConverterEvents } from "../converter/converter-events.js";
export function debugReflectionLifetimes(app) {
    app.converter.on(ConverterEvents.CREATE_PROJECT, logCreate);
    app.converter.on(ConverterEvents.CREATE_SIGNATURE, logCreate);
    app.converter.on(ConverterEvents.CREATE_TYPE_PARAMETER, logCreate);
    app.converter.on(ConverterEvents.CREATE_DECLARATION, logCreate);
    app.converter.on(ConverterEvents.CREATE_DOCUMENT, logCreate);
    app.converter.on(ConverterEvents.CREATE_PARAMETER, logCreate);
    app.converter.on(ConverterEvents.CREATE_PROJECT, (_context, project) => {
        const oldRemove = project["_removeReflection"];
        project["_removeReflection"] = function (reflection) {
            console.log("Remove", reflection.id, reflection.getFullName());
            return oldRemove.call(this, reflection);
        };
    });
}
function logCreate(_context, refl) {
    console.log("Create", refl.variant, refl.id, refl.getFullName());
}
