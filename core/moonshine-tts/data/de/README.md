# German — `de`

## Contents

- **`dict.tsv`** — word → IPA (one pronunciation per line, tab-separated), used by C++ `GermanRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — file `data/de.txt` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py` (see script header table). |

The same lexicon may also appear under `models/de/dict.tsv` in this repo (identical content used when `model_root` layout places `de` under `models/`).

## Recreating

From the repository root:

```bash
python scripts/download_multilingual_ipa_lexicons.py --only de
```

This writes `data/de/dict.tsv` and `data/de/source.txt`. Copy `dict.tsv` into `data/de/` if you keep a separate C++ data tree.

Optional: inspect `data/de/source.txt` for the exact upstream URL recorded at download time.
