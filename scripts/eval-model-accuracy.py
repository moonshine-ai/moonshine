#
# On a Mac you'll need to set up ffmpeg using:
# brew install ffmpeg@8
# export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@8/lib:$DYLD_LIBRARY_PATH"
#

from moonshine_voice import (
    get_model_for_language,
    Transcriber,
    string_to_model_arch,
    model_arch_to_string,
    ModelArch,
)

import argparse
import os
import numpy as np
from datasets import load_dataset
from jiwer import wer, cer
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
import pandas as pd
import json
from whisper.normalizers import EnglishTextNormalizer

language_info = {
    "ar": {
        "english_name": "Arabic",
    },
    "es": {
        "english_name": "Spanish",
    },
    "ja": {
        "english_name": "Japanese",
    },
    "ko": {
        "english_name": "Korean",
    },
    "en": {
        "english_name": "English",
    },
    "uk": {
        "english_name": "Ukrainian",
    },
    "vi": {
        "english_name": "Vietnamese",
    },
    "zh": {
        "english_name": "Mandarin",
    },
}

english_text_normalizer = EnglishTextNormalizer()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--languages",
    type=str,
    default="ar_eg,en_us,es_419,ja_jp,ko_kr,uk_ua,vi_vn,cmn_hans_cn",
)
parser.add_argument("--model-archs", type=str, default="base")
parser.add_argument("--model-paths", type=str, default=None)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

languages = args.languages.split(",")

model_arch_strings = args.model_archs.split(",")
model_archs = [
    string_to_model_arch(model_arch_string) for model_arch_string in model_arch_strings
]

if args.model_paths is not None:
    model_paths = args.model_paths.split(":")
    if len(model_paths) != len(model_archs):
        raise ValueError(
            "Number of model paths must match number of model architectures"
        )
    model_paths = [model_path.strip() for model_path in model_paths]
else:
    model_paths = None

results = {}
results["Language"] = []
for model_arch in model_archs:
    results[f"{model_arch_to_string(model_arch).capitalize()} WER"] = []
    results[f"{model_arch_to_string(model_arch).capitalize()} CER"] = []

results["Fleurs Code"] = []

detailed_results = {
    "ground_truth": [],
    "transcription": [],
    "wer": [],
    "cer": [],
    "language": [],
    "model_arch": [],
}

for fleurs_language in languages:
    if fleurs_language == "cmn_hans_cn":
        language_code = "zh"
        country_code = "cn"
    else:
        language_code, country_code = fleurs_language.split("_")

    english_name = language_info[language_code]["english_name"]
    results["Language"].append(english_name)
    results["Fleurs Code"].append(fleurs_language)

    if language_code == "ko":
        current_model_archs = [ModelArch.TINY]
        model_arch_string = "Base"
    else:
        current_model_archs = model_archs
        model_arch_string = model_arch_to_string(model_arch).capitalize()

    for model_arch in current_model_archs:
        print(
            f"Evaluating {model_arch_to_string(model_arch)} model for {english_name} on FLEURS dataset"
        )

        if model_paths is not None:
            path = model_paths[model_archs.index(model_arch)]
            arch = model_arch
        else:
            path, arch = get_model_for_language(language_code, model_arch)
        # English and Spanish use the tokenizer more efficiently, 
        # so we can use a lower max tokens per second to avoid hallucinations.
        if language_code == "en" or language_code == "es":
            max_tokens_per_second = 6.5
        else:
            max_tokens_per_second = 13.0
        # Disable the VAD since these are already pre-segmented.
        options = {
            "vad_threshold": 0.0,
            "max_tokens_per_second": max_tokens_per_second,
        }
        transcriber = Transcriber(path, arch, options=options)

        fleurs_dataset = load_dataset(
            "google/fleurs", fleurs_language, trust_remote_code=True
        )

        test_dataset = fleurs_dataset["test"]

        wer_total = 0
        cer_total = 0
        character_count = 0

        for sample in tqdm(test_dataset):
            audio = sample["audio"]["array"].astype(np.float32)
            ground_truth = sample["transcription"]
            current_character_count = len(ground_truth)
            character_count += current_character_count
            audio_duration = audio.shape[0] / 16000.0
            transcript = transcriber.transcribe_without_streaming(audio, 16000)
            first_line = transcript.lines[0]
            transcription = first_line.text
            normalized_ground_truth = english_text_normalizer(ground_truth)
            normalized_transcription = english_text_normalizer(transcription)
            current_wer = wer(normalized_ground_truth, normalized_transcription)
            current_cer = cer(normalized_ground_truth, normalized_transcription)
            wer_total += current_wer * current_character_count
            cer_total += current_cer * current_character_count

            if args.verbose:
                print(f"Ground truth: {normalized_ground_truth}")
                print(f"Transcription: {normalized_transcription}")
                print(f"WER: {current_wer:.2%}, CER: {current_cer:.2%}")
                print("--------------------------------")

            detailed_results["ground_truth"].append(normalized_ground_truth)
            detailed_results["transcription"].append(normalized_transcription)
            detailed_results["wer"].append(current_wer)
            detailed_results["cer"].append(current_cer)
            detailed_results["model_arch"].append(model_arch_to_string(model_arch))

        wer_result = wer_total / character_count
        cer_result = cer_total / character_count

        print(f"WER: {wer_result:.2%}, CER: {cer_result:.2%}")

        results[f"{model_arch_string} WER"].append(f"{wer_result:.2%}")
        results[f"{model_arch_string} CER"].append(f"{cer_result:.2%}")

dataframe = pd.DataFrame(results)
print(dataframe)
dataframe.to_excel("moonshine-eval-results.xlsx", index=False)
print("Results saved to moonshine-eval-results.xlsx")

detailed_df = pd.DataFrame(detailed_results)

detailed_writer = pd.ExcelWriter(
    "moonshine-eval-detailed-results.xlsx", engine="xlsxwriter"
)
detailed_df.to_excel(detailed_writer, sheet_name="Moonshine Eval")
detailed_workbook = detailed_writer.book
detailed_worksheet = detailed_writer.sheets["Moonshine Eval"]

wrap_text_format = detailed_workbook.add_format({"text_wrap": True})
percent_format = detailed_workbook.add_format({"num_format": "0%"})

detailed_worksheet.set_column(1, 2, 40, wrap_text_format)
detailed_worksheet.set_column(3, 4, None, percent_format)

detailed_writer.close()

print("Detailed results saved to moonshine-eval-detailed-results.xlsx")
