# Simplified Chinese — `zh_hans`

## Contents

- **`dict.tsv`** — word/phrase → Mandarin IPA; used with POS-aware disambiguation for common polyphones.
- **`roberta_chinese_base_upos_onnx/`** — RoBERTa token-classification ONNX: WordPiece + UPOS tags for word segmentation and tag features in C++ `ChineseTokPosOnnx` / `ChineseOnnxG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — `data/zh_hans.txt` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |
| ONNX bundle | [KoichiYasuoka/chinese-roberta-base-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos) on Hugging Face. |

## Recreating

### Lexicon

```bash
python scripts/download_multilingual_ipa_lexicons.py --only zh_hans
```

### ONNX

1. Install: `pip install torch onnx transformers numpy` (see `scripts/export_chinese_roberta_upos_onnx.py` for any version notes).
2. From repo root:

   ```bash
   python scripts/export_chinese_roberta_upos_onnx.py
   ```

   Default output: `data/zh_hans/roberta_chinese_base_upos_onnx/`.

3. C++ expects at least: `model.onnx`, `vocab.txt`, `tokenizer_config.json`, `meta.json` (additional tokenizer files from export are optional for the C++ loader).

Copy `dict.tsv` and the ONNX directory into `data/zh_hans/` as needed.

**Byte stability:** A fresh export may not match the committed `model.onnx` byte-for-byte (PyTorch ONNX backend, int8 shrinking, opset). `meta.json` and `vocab.txt` have been observed to match; `tokenizer_config.json` may pick up small JSON differences across `transformers` versions.
