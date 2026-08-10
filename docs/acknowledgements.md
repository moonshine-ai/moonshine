# Acknowledgements

We're grateful to:

- Lambda and Stephen Balaban for supporting our model training through [their foundational model grants](https://lambda.ai/research).
- The ONNX Runtime community for building [a fast, cross-platform inference engine](https://github.com/microsoft/onnxruntime).
- [Alexander Veysov](https://github.com/snakers4) for the great [Silero Voice Activity Detector](https://github.com/snakers4/silero-vad).
- [Viktor Kirilov](https://github.com/onqtam) for [his fantastic DocTest C++ testing framework](https://github.com/doctest/doctest).
- [Nemanja Trifunovic](https://github.com/nemtrif) for [his very helpful UTF8 CPP library](https://github.com/nemtrif/utfcpp).
- The [Pyannote team](https://www.pyannote.ai/) for making available their speaker embedding model.
- The [espeak-ng community](https://github.com/espeak-ng/espeak-ng/), for all of their inspiring work tackling the endless complexities of translating the written word into speech.
- The [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) and [eSpeak NG](https://github.com/espeak-ng/espeak-ng) for English G2P lexicon and pronunciation filtering ([`core/moonshine-tts/data/en_us`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/en_us)).
- [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) for multilingual IPA lexicon data used across many locales ([`core/moonshine-tts/data`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data)).
- [WikiPron](https://github.com/CUNY-CL/wikipron) (CUNY-CL) for Italian, Russian, and European Portuguese pronunciations.
- [Koichi Yasuoka](https://huggingface.co/KoichiYasuoka) for the Hugging Face models [chinese-roberta-base-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos), [roberta-small-japanese-char-luw-upos](https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos), and [roberta-base-korean-morph-upos](https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos).
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [onnx-community/Kokoro-82M-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-ONNX) for Kokoro TTS weights and ONNX ([`core/moonshine-tts/data/kokoro`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/kokoro)).
- [PiperTTS](https://huggingface.co/rhasspy/piper-voices) for their excellent lightweight TTS models.
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) from [MyShell](https://myshell.ai) as reference for Korean Piper voice training ([`core/moonshine-tts/data/ko`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/ko)).
- [English Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Copyrights) and [hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords) for Hindi lexicon material ([`core/moonshine-tts/data/hi`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/hi)).
- [hbenbel/French-Dictionary](https://github.com/hbenbel/French-Dictionary) for related French liaison lexicon work ([`core/moonshine-tts/data/fr`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/fr)).
- [AbderrahmanSkiredj1/arabertv02_tashkeel_fadel](https://huggingface.co/AbderrahmanSkiredj1/arabertv02_tashkeel_fadel) for Arabic diacritization and [CAMeL Tools](https://camel-tools.readthedocs.io/) for optional Arabic MSA lexicon builds ([`core/moonshine-tts/data/ar_msa`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/ar_msa)).
- [ZipVoice](https://github.com/k2-fsa/ZipVoice) for their high-quality text to speech and voice cloning.
- The team behind the [VCTK dataset](https://datashare.ed.ac.uk/collections/8f1b06bc-ec26-4b8d-ac4e-acb14537d811/search) at the University of Edinburgh for generously providing a rich source of voice styles.
