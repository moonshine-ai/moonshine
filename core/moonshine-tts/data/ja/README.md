# Japanese — `ja`

`moonshine_tts` and ``MoonshineTTS`` (with ``use_bundled_cpp_g2p_data``) use ``MoonshineG2POptions::model_root = builtin_cpp_data_root()`` (i.e. this ``data`` tree), so no ``--model-root`` flag is required for CLI TTS.

## Contents

- **`dict.tsv`** — surface → IPA lexicon (ipa-dict), used after token-level analysis.
- **`roberta_japanese_char_luw_upos_onnx/`** — RoBERTa **char-level LUW + UPOS** ONNX used by C++ `JapaneseTokPosOnnx` / `JapaneseOnnxG2p` for segmentation and tags (not a direct “IPA model”).

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — `data/ja.txt` (MIT). Via `scripts/download_multilingual_ipa_lexicons.py`. |
| ONNX bundle | [KoichiYasuoka/roberta-small-japanese-char-luw-upos](https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos) on Hugging Face. |

## Recreating

### Lexicon

```bash
python scripts/download_multilingual_ipa_lexicons.py --only ja
```

### ONNX (includes external weights if the exporter splits them)

1. Install: `pip install torch onnx transformers` (and any extras required by `scripts/export_japanese_ud_onnx.py` on your machine).
2. From repo root:

   ```bash
   python scripts/export_japanese_ud_onnx.py
   ```

   Default output directory: `data/ja/roberta_japanese_char_luw_upos_onnx/`.

3. C++ runtime expects at least: `model.onnx`, `vocab.txt`, `tokenizer_config.json`, `meta.json`. If ONNX Runtime loads external data, keep **`model.onnx.data`** next to `model.onnx`.

Copy the `dict.tsv` and the whole ONNX directory into `data/ja/` as needed.

**Byte stability:** Like Chinese, `model.onnx` is not guaranteed to be byte-identical across exporter versions. The checked-in Japanese bundle may use **external** data (`model.onnx` + `model.onnx.data`); a new export might emit a single larger `model.onnx` instead—both layouts can work with ONNX Runtime if paths stay consistent.
