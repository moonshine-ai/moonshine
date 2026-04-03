# C++ runtime data (per-language bundles)

This tree mirrors assets the C++ `moonshine-tts` (speak) and `moonshine-tts-g2p` (G2P-only) CLIs expect under a single **`--model-root`** (see each `README.md` for paths). When this project is embedded as the `moonshine-tts` submodule, canonical Python assets usually live under the **parent** repo’s `data/` and `models/`; this directory is the curated copy shipped for self-contained C++ builds.

| Folder | Role |
|--------|------|
| [ar_msa](ar_msa/README.md) | Arabic MSA: tashkīl ONNX + optional lexicon |
| [de](de/README.md) | German IPA lexicon |
| [en_us](en_us/README.md) | English CMU-style lexicon + heteronym/OOV ONNX |
| [fr](fr/README.md) | French lexicon + liaison POS CSVs |
| [hi](hi/README.md) | Hindi Devanagari lexicon (Wiktionary + frequency merge) |
| [it](it/README.md) | Italian IPA lexicon |
| [ja](ja/README.md) | Japanese lexicon + char-LUW UPOS ONNX |
| [ko](ko/README.md) | Korean IPA lexicon |
| [kokoro](kokoro/README.md) | Kokoro-82M ONNX TTS + `.kokorovoice` bundles (`moonshine-tts` speak CLI) |
| [nl](nl/README.md) | Dutch IPA lexicon |
| [pt_br](pt_br/README.md) | Brazilian Portuguese IPA lexicon |
| [pt_pt](pt_pt/README.md) | European Portuguese IPA lexicon |
| [ru](ru/README.md) | Russian IPA lexicon |
| [vi](vi/README.md) | Vietnamese IPA lexicon |
| [zh_hans](zh_hans/README.md) | Simplified Chinese lexicon + RoBERTa UPOS ONNX |

All commands below assume the **repository root** as the current working directory unless noted.

## Regeneration verification (2026-03-30)

Commands below were run from a clean temp output directory and compared to the parent monorepo’s `data/` / `models/` and this `moonshine-tts/data/` tree unless noted.

| Recipe | Byte-identical to tree? | Notes |
|--------|-------------------------|--------|
| `download_multilingual_ipa_lexicons.py` for `de`, `fr`, `it`, `ja`, `ko`, `nl`, `pt_br`, `pt_pt`, `ru`, `vi`, `zh_hans` | **Yes** | All eleven `dict.tsv` files matched the parent repo’s `data/<lang>/dict.tsv` and this tree’s `data/<lang>/dict.tsv`. |
| `download_cmudict_to_tsv.py` → `data/en_us/dict.tsv` | **Yes** | Restored after run; output matched prior file. |
| `export_models_to_onnx.py` (heteronym + OOV) | **Yes** (after bugfix) | `model.onnx` and `onnx-config.json` match `models/en_us/{heteronym,oov}/` when re-exported from the same checkpoints. A script bug that wrote `onnx_export.onnx_path` as `onnx-config.json` was fixed in `scripts/export_models_to_onnx.py` (must pass the `model.onnx` path into `_build_config_onnx`, not the JSON path). Copy `g2p-config.json` into the temp `model_root` if you rely on `--only config` defaults. |
| `export_arabic_msa_diacritizer_onnx.py` | **No (failed)** | With `torch` 2.10 + current `transformers`, export raises `ValueError` on `attention_mask` shape inside BERT. The checked-in Arabic ONNX was produced with an older stack; see [ar_msa/README.md](ar_msa/README.md). |
| `export_chinese_roberta_upos_onnx.py` | **Partial** | `meta.json` and `vocab.txt` match; `tokenizer_config.json` differs by an empty `extra_special_tokens` key (tokenizer version). `model.onnx` **differs** (same HF weights, different ONNX graph / int8 shrink / opset path under torch 2.10). |
| `export_japanese_ud_onnx.py` | **Partial** | Same pattern as Chinese: `meta.json` / `vocab.txt` match; `tokenizer_config.json` minor JSON diff. Checked-in bundle uses **external** weights (`model.onnx` + `model.onnx.data`); a fresh export produced a single larger `model.onnx` (no `.data` split). |
| `export_korean_ud_onnx.py` | **Partial** | `meta.json` matches; `model.onnx` differs in size/hash (int8 shrink / exporter). |
| `filter_dict_by_espeak_coverage.py` → `dict_filtered_heteronyms.tsv` | Not rerun | Needs eSpeak NG + corpus; recipe is environment-specific. |
| `build_ar_msa_lexicon_from_camel_tools.py` | Not rerun | Requires Camel Tools + MLE DB install. |
| French `*.csv` POS lists | N/A | No automated download script in-repo. |

**Takeaway:** Lexicon recipes are deterministic against current upstream URLs. Transformer ONNX exports are **not** guaranteed to be byte-stable across PyTorch / `transformers` / export-backend versions; treat `meta.json` + tokenizer assets + parity tests as the contract when bytes drift.
