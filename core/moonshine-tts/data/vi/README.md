# Vietnamese — `vi`

## Contents

- **`dict.tsv`** — word → IPA (Northern orthography `vi_N` in ipa-dict) for C++ `VietnameseRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — Vietnamese dataset (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only vi
```

Writes `data/vi/dict.tsv` and `data/vi/source.txt`. Copy `dict.tsv` into `data/vi/` if needed.
