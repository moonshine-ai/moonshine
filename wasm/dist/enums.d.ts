/**
 * Model-architecture enums and helpers, mirroring the constants in
 * `core/moonshine-c-api.h` and the `model_arch_to_string` /
 * `string_to_model_arch` helpers in the Python binding.
 */
/** Speech-to-text model architectures. Values match `MOONSHINE_MODEL_ARCH_*`. */
export declare enum ModelArch {
    Tiny = 0,
    Base = 1,
    TinyStreaming = 2,
    BaseStreaming = 3,
    SmallStreaming = 4,
    MediumStreaming = 5
}
export declare function modelArchToString(arch: ModelArch): string;
export declare function stringToModelArch(name: string): ModelArch;
/** Embedding-model architectures for intent recognition. */
export declare enum EmbeddingModelArch {
    Gemma300M = 0
}
/** Transcribe flags. Values match `MOONSHINE_FLAG_*`. */
export declare enum TranscribeFlags {
    None = 0,
    ForceUpdate = 1,
    SpellingMode = 2
}
//# sourceMappingURL=enums.d.ts.map