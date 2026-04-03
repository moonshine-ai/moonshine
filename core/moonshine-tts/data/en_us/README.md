# English (US) — `en_us`

## Contents

- **`dict_filtered_heteronyms.tsv`** — CMUdict-derived lexicon with heteronym rows pruned using corpus + eSpeak alignment (see script below).
- **`g2p-config.json`** — flags which optional ONNX bundles exist (`uses_heteronym_model`, `uses_oov_model`).
- **`heteronym/`** — `model.onnx` + `onnx-config.json` (merged vocabs + `homograph_index` payload) + **`homograph_index.json`** for ordering CMU alternatives.
- **`oov/`** — `model.onnx` + `onnx-config.json` for greedy character→phoneme decoding for unknown words.

## Provenance

| Asset | Source |
|--------|--------|
| Base pronunciations | [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) (`cmudict.dict`), converted to IPA via repo `cmudict_ipa` — see `scripts/download_cmudict_to_tsv.py`. |
| Filtered TSV | Repo pipeline: `scripts/filter_dict_by_espeak_coverage.py` prunes rare heteronym readings using eSpeak NG + corpus text. |
| Heteronym / OOV ONNX | Small transformer decoders **trained in this repository** (`train_heteronym.py`, `train_oov.py`); checkpoints live under `models/en_us/heteronym/` and `models/en_us/oov/` (e.g. `checkpoint.pt` + JSON vocabs). |
| ONNX export | `scripts/export_models_to_onnx.py` merges vocabs and indices into `onnx-config.json` next to `model.onnx`. |

## Recreating (high level)

1. **Raw CMU → TSV**

   ```bash
   python scripts/download_cmudict_to_tsv.py
   ```

   Writes `data/en_us/dict.tsv` by default.

2. **Prune heteronyms** (requires `espeak-phonemizer` / system eSpeak NG and a sentence corpus such as `data/en_us/input_text.txt` or your own `--input-text`):

   ```bash
   python scripts/filter_dict_by_espeak_coverage.py \
     --dict-path data/en_us/dict.tsv \
     --input-text data/en_us/input_text.txt \
     --out data/en_us/dict_filtered_heteronyms.tsv
   ```

   Install the result as `models/en_us/dict_filtered_heteronyms.tsv` (or the path your `model_root` uses).

3. **Train** heteronym and OOV models (see project docs / `train_heteronym.py`, `train_oov.py`, dataset scripts under `scripts/`).

4. **Export ONNX** (from repo root, with PyTorch + `onnx` installed):

   ```bash
   python scripts/export_models_to_onnx.py --model-root models --language en_us --only both \
     --heteronym-checkpoint models/en_us/heteronym/checkpoint.pt \
     --oov-checkpoint models/en_us/oov/checkpoint.pt
   ```

   Keep `models/en_us/g2p-config.json` present so `--only` defaults match your intent (if it is missing, the exporter falls back to CLI-only flags). Re-exporting from the same checkpoints reproduces `model.onnx` and `onnx-config.json` byte-for-byte with the fixed exporter (see `data/README.md` → *Regeneration verification*).

5. Copy `dict_filtered_heteronyms.tsv`, `g2p-config.json`, and the `heteronym/` and `oov/` directories into `data/en_us/` if you maintain this layout.
