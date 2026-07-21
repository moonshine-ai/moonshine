/**
 * Model-architecture enums and helpers, mirroring the constants in
 * `core/moonshine-c-api.h` and the `model_arch_to_string` /
 * `string_to_model_arch` helpers in the Python binding.
 */
/** Speech-to-text model architectures. Values match `MOONSHINE_MODEL_ARCH_*`. */
export var ModelArch;
(function (ModelArch) {
    ModelArch[ModelArch["Tiny"] = 0] = "Tiny";
    ModelArch[ModelArch["Base"] = 1] = "Base";
    ModelArch[ModelArch["TinyStreaming"] = 2] = "TinyStreaming";
    ModelArch[ModelArch["BaseStreaming"] = 3] = "BaseStreaming";
    ModelArch[ModelArch["SmallStreaming"] = 4] = "SmallStreaming";
    ModelArch[ModelArch["MediumStreaming"] = 5] = "MediumStreaming";
})(ModelArch || (ModelArch = {}));
const MODEL_ARCH_NAMES = {
    [ModelArch.Tiny]: 'tiny',
    [ModelArch.Base]: 'base',
    [ModelArch.TinyStreaming]: 'tiny_streaming',
    [ModelArch.BaseStreaming]: 'base_streaming',
    [ModelArch.SmallStreaming]: 'small_streaming',
    [ModelArch.MediumStreaming]: 'medium_streaming',
};
export function modelArchToString(arch) {
    return MODEL_ARCH_NAMES[arch] ?? `unknown(${arch})`;
}
export function stringToModelArch(name) {
    const entry = Object.entries(MODEL_ARCH_NAMES).find(([, value]) => value === name);
    if (!entry) {
        throw new Error(`Unknown model arch: ${name}`);
    }
    return Number(entry[0]);
}
/** Embedding-model architectures for intent recognition. */
export var EmbeddingModelArch;
(function (EmbeddingModelArch) {
    EmbeddingModelArch[EmbeddingModelArch["Gemma300M"] = 0] = "Gemma300M";
})(EmbeddingModelArch || (EmbeddingModelArch = {}));
/** Transcribe flags. Values match `MOONSHINE_FLAG_*`. */
export var TranscribeFlags;
(function (TranscribeFlags) {
    TranscribeFlags[TranscribeFlags["None"] = 0] = "None";
    TranscribeFlags[TranscribeFlags["ForceUpdate"] = 1] = "ForceUpdate";
    TranscribeFlags[TranscribeFlags["SpellingMode"] = 2] = "SpellingMode";
})(TranscribeFlags || (TranscribeFlags = {}));
//# sourceMappingURL=enums.js.map