# English (US) — `en_us`

## Contents

- **`dict_filtered_heteronyms.tsv`** — CMUdict-derived lexicon with extra pronunciations pruned using corpus + eSpeak alignment (see script below). Words that still have multiple CMU readings are resolved at runtime by **sorting alternatives and taking the first** (no heteronym ONNX).
- **`g2p-config.json`** — `uses_dictionary` and **`uses_oov_model`** (must be `true` for the bundled layout; the C++ factory requires OOV ONNX when this flag is on).
- **`oov/`** — `model.onnx` + `onnx-config.json` for greedy character→phoneme decoding for out-of-vocabulary words.

## Provenance

| Asset | Source |
|--------|--------|
| Base pronunciations | [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) (`cmudict.dict`), converted to IPA via repo `cmudict_ipa` — see `scripts/download_cmudict_to_tsv.py`. |
| Filtered TSV | Repo pipeline: `scripts/filter_dict_by_espeak_coverage.py` prunes rare readings using eSpeak NG + corpus text. |
| OOV ONNX | Small transformer decoder **trained in this repository** (`train_oov.py`, etc.); checkpoints live under `models/en_us/oov/` (e.g. `checkpoint.pt` + JSON vocabs). |
| ONNX export | `scripts/export_models_to_onnx.py` (or the OOV-only path your pipeline uses) merges vocabs and indices into `onnx-config.json` next to `model.onnx`. |

## Recreating (high level)

1. **Raw CMU → TSV**

   ```bash
   python scripts/download_cmudict_to_tsv.py
   ```

   Writes `data/en_us/dict.tsv` by default.

2. **Prune rare readings** (requires `espeak-phonemizer` / system eSpeak NG and a sentence corpus such as `data/en_us/input_text.txt` or your own `--input-text`):

   ```bash
   python scripts/filter_dict_by_espeak_coverage.py \
     --dict-path data/en_us/dict.tsv \
     --input-text data/en_us/input_text.txt \
     --out data/en_us/dict_filtered_heteronyms.tsv
   ```

   Install the result as `models/en_us/dict_filtered_heteronyms.tsv` (or the path your `model_root` uses).

3. **Train** the OOV model (see project docs / `train_oov.py`, dataset scripts under `scripts/`).

4. **Export ONNX** (PyTorch + `onnx` installed). Use the repo’s English OOV export path (e.g. `scripts/export_models_to_onnx.py` when present in your tree) so **`oov/model.onnx`** and **`oov/onnx-config.json`** are produced, then copy them into `data/en_us/oov/`.

   Keep `models/en_us/g2p-config.json` aligned with what you ship (`uses_oov_model: true`, no heteronym bundle). Re-exporting from the same checkpoint reproduces `model.onnx` and `onnx-config.json` with a fixed exporter (see `data/README.md` → *Regeneration verification*).

5. Copy **`dict_filtered_heteronyms.tsv`**, **`g2p-config.json`**, and the **`oov/`** directory into `data/en_us/` if you maintain this layout.
