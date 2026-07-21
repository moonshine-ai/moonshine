/**
 * Model-architecture enums and helpers, mirroring the constants in
 * `core/moonshine-c-api.h` and the `model_arch_to_string` /
 * `string_to_model_arch` helpers in the Python binding.
 */

/** Speech-to-text model architectures. Values match `MOONSHINE_MODEL_ARCH_*`. */
export enum ModelArch {
  Tiny = 0,
  Base = 1,
  TinyStreaming = 2,
  BaseStreaming = 3,
  SmallStreaming = 4,
  MediumStreaming = 5,
}

const MODEL_ARCH_NAMES: Record<ModelArch, string> = {
  [ModelArch.Tiny]: 'tiny',
  [ModelArch.Base]: 'base',
  [ModelArch.TinyStreaming]: 'tiny_streaming',
  [ModelArch.BaseStreaming]: 'base_streaming',
  [ModelArch.SmallStreaming]: 'small_streaming',
  [ModelArch.MediumStreaming]: 'medium_streaming',
};

export function modelArchToString(arch: ModelArch): string {
  return MODEL_ARCH_NAMES[arch] ?? `unknown(${arch})`;
}

export function stringToModelArch(name: string): ModelArch {
  const entry = Object.entries(MODEL_ARCH_NAMES).find(
    ([, value]) => value === name,
  );
  if (!entry) {
    throw new Error(`Unknown model arch: ${name}`);
  }
  return Number(entry[0]) as ModelArch;
}

/** Embedding-model architectures for intent recognition. */
export enum EmbeddingModelArch {
  Gemma300M = 0,
}

/** Transcribe flags. Values match `MOONSHINE_FLAG_*`. */
export enum TranscribeFlags {
  None = 0,
  ForceUpdate = 1 << 0,
  SpellingMode = 1 << 1,
}
