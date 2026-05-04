# Portuguese (Brazil) — `pt_br`

## Contents

- **`dict.tsv`** — word → IPA for C++ `PortugueseRuleG2p` (Brazilian variant).

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — Brazilian Portuguese file in `data/` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only pt_br
```

Writes `data/pt_br/dict.tsv` and `data/pt_br/source.txt`. Copy `dict.tsv` into `data/pt_br/` if needed.
