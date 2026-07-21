/* eslint-disable no-console */
import { join } from "node:path";
import { Reflection, ReflectionKind, } from "../models/index.js";
const serializer = {
    priority: 0,
    supports(x) {
        return x instanceof Reflection;
    },
    toObject(item, obj) {
        obj.url = item.url;
        obj.hasOwnDocument = item.hasOwnDocument;
        // obj.anchor = item.anchor;
        delete obj.sources;
        delete obj.groups;
        delete obj.categories;
        delete obj.readme;
        delete obj.content;
        obj.kind = ReflectionKind[obj.kind];
        delete obj.flags;
        delete obj.defaultValue;
        delete obj.symbolIdMap;
        delete obj.files;
        delete obj.packageName;
        delete obj.variant;
        delete obj.extendedTypes;
        delete obj.inheritedFrom;
        if (!["reflection", "reference"].includes(obj.type?.type)) {
            delete obj.type;
        }
        if (obj.comment) {
            obj.comment.summary = obj.comment.summary.filter((part) => part.kind === "inline-tag");
            obj.comment.blockTags = obj.comment.blockTags?.filter((tag) => {
                tag.content = tag.content.filter((part) => part.kind === "inline-tag");
                return tag.content.length;
            });
            if (!obj.comment.summary.length &&
                !obj.comment.blockTags?.length &&
                !obj.comment.modifierTags) {
                delete obj.comment;
            }
        }
        return obj;
    },
};
export function debugRendererUrls(app, { json = false, logs = false } = { logs: true }) {
    app.renderer.postRenderAsyncJobs.push(async (evt) => {
        if (json) {
            app.serializer.addSerializer(serializer);
            await app.generateJson(evt.project, join(evt.outputDirectory, "url_debug.json"));
            app.serializer.removeSerializer(serializer);
        }
        if (logs) {
            for (const id in evt.project.reflections) {
                const refl = evt.project.reflections[id];
                console.log(refl.id, refl.getFullName(), refl.url, refl.hasOwnDocument);
            }
        }
    });
}
