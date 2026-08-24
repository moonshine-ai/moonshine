# License

This code, apart from the source in `core/third-party`, is licensed under the MIT License, see LICENSE in this repository.

Moonshine models are released under the MIT License by default, in every language and at every size. This includes all streaming speech-to-text models and all English-language models.

The only speech-to-text models that are **not** MIT are the legacy non-streaming models for languages other than English, which remain under the [Moonshine Community License](https://moonshine.ai), a non-commercial license. That list is exhaustive:

| Language | Non-commercial models |
| --- | --- |
| Arabic | Base, Tiny |
| Japanese | Base, Tiny |
| Korean | Base, Tiny |
| Mandarin | Base, Tiny |
| Spanish | Base |
| Ukrainian | Base, Tiny |
| Vietnamese | Base, Tiny |

Any speech-to-text model not named in that table is MIT, including every streaming model and the English Tiny and Base models. These legacy models stay available by naming the architecture, for the languages that now default to streaming.

The code in `core/third-party` is licensed according to the terms of the open source projects it originates from, with details in a LICENSE file in each subfolder. 

The Eigen library is compiled with only the MPL-2.0 subset, all files with other licenses are removed.

The text to speech and grapheme to phoneme models and data files are licensed under the terms listed in their readmes and their source repositories. Per-language details and regeneration notes live under [`core/moonshine-tts/data/`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/README.md).
