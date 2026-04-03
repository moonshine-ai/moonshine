# Dutch — `nl`

## Contents

- **`dict.tsv`** — word → IPA for C++ `DutchRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — Dutch dataset in the ipa-dict tree (MIT). **License note:** ipa-dict’s README marks Dutch (INT) data as **CC BY**—see upstream repo for exact terms. Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only nl
```

Writes `data/nl/dict.tsv` and `data/nl/source.txt`. Copy `dict.tsv` into `data/nl/` if needed.
