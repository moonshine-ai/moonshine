import os
import sys

from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.download import download_model, get_cache_dir
from moonshine_voice.utils import get_assets_path, load_wav_file

MODEL_INFO = {
    "ar": {
        "english_name": "Arabic",
        "models": [
            {"model_name": "base-ar", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-ar/quantized/base-ar"}
        ]
    },
    "es": {
        "english_name": "Spanish",
        "models": [
            {"model_name": "base-es", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-es/quantized/base-es"}
        ]
    },
    "en": {
        "english_name": "English",
        "models": [
            {"model_name": "base-en", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-en/quantized/base-en"},
            {"model_name": "tiny-en", "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-en/quantized/tiny-en"}
        ]
    },
    "ja": {
        "english_name": "Japanese",
        "models": [
            {"model_name": "base-ja", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-ja/quantized/base-ja"},
            {"model_name": "tiny-ja", "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja"}
        ]
    },
    "ko": {
        "english_name": "Korean",
        "models": [
            {"model_name": "base-ko", "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko"}
        ]
    },
    "vi": {
        "english_name": "Vietnamese",
        "models": [
            {"model_name": "base-vi", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-vi/quantized/base-vi"}
        ]
    },
    "uk": {
        "english_name": "Ukrainian",
        "models": [
            {"model_name": "base-uk", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-uk/quantized/base-uk"}
        ]
    },
    "zh": {
        "english_name": "Chinese",
        "models": [
            {"model_name": "base-zh", "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-zh/quantized/base-zh"}
        ]
    },
}


def find_model_info(language: str = "en", model_arch: ModelArch = None) -> dict:
    if language in MODEL_INFO.keys():
        language_key = language
    else:
        for key, info in MODEL_INFO.items():
            if language.lower() == info["english_name"].lower():
                language_key = key
                break
        if language_key is None:
            raise ValueError(
                f"Language not found: {language}. Supported languages: {supported_languages_friendly()}")

    model_info = MODEL_INFO[language_key]
    available_models = model_info["models"]
    if model_arch is None:
        return available_models[0]
    for model in available_models:
        if model["model_arch"] == model_arch:
            return model
    raise ValueError(
        f"Model not found for language: {language} and model arch: {model_arch}. Available models: {available_models}")


def supported_languages_friendly() -> str:
    return ", ".join([f"{key} ({info['english_name']})" for key, info in MODEL_INFO.items()])


def supported_languages() -> list[str]:
    return list(MODEL_INFO.keys())


def get_components_for_model_info(model_info: dict) -> list[str]:
    return [
        "encoder_model.ort",
        "decoder_model_merged.ort",
        "tokenizer.bin"
    ]


def download_model_from_info(model_info: dict) -> tuple[str, ModelArch]:
    cache_dir = get_cache_dir()
    model_download_url = model_info["download_url"]
    model_folder_name = model_download_url.replace("https://", "")
    root_model_path = os.path.join(cache_dir, model_folder_name)
    components = get_components_for_model_info(model_info)
    for component in components:
        component_download_url = f"{model_download_url}/{component}"
        component_path = os.path.join(root_model_path, component)
        download_model(component_download_url, component_path)
    return str(root_model_path), model_info["model_arch"]


def get_model_for_language(wanted_language: str = "en", wanted_model_arch: ModelArch = None) -> tuple[str, ModelArch]:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    if wanted_language != "en":
        print("Using a model released under the non-commercial Moonshine Community License. See https://www.moonshine.ai/license for details.", file=sys.stderr)
    return download_model_from_info(model_info)


def log_model_info(wanted_language: str = "en", wanted_model_arch: ModelArch = None) -> None:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    model_root_path, model_arch = download_model_from_info(model_info)
    print(f"Downloaded model path: {model_root_path}")
    print(f"Model arch: {model_arch}")
    print(f"Model download url: {model_info['download_url']}")
    print(f"Model components: {get_components_for_model_info(model_info)}")


if __name__ == "__main__":
    model_path, model_arch = get_model_for_language("en", ModelArch.BASE)

    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
    two_cities_wav_path = os.path.join(get_assets_path(), "two_cities.wav")
    audio_data, sample_rate = load_wav_file(two_cities_wav_path)
    transcript = transcriber.transcribe_without_streaming(
        audio_data, sample_rate=sample_rate
    )
    for line in transcript.lines:
        print(
            f"[{line.start_time:.2f}s - {line.start_time + line.duration:.2f}s] {line.text}"
        )

    for language in supported_languages():
        model_path, model_arch = get_model_for_language(language)
        # Triggers a download of the model.
        log_model_info(language, model_arch)
        # Make sure we can load the model.
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
