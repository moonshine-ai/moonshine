# Portuguese (Portugal) — `pt_pt`

## Contents

- **`dict.tsv`** — word → IPA for C++ `PortugueseRuleG2p` (European variant).

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [CUNY-CL/wikipron](https://github.com/CUNY-CL/wikipron) scrape TSV `por_latn_po_broad` (Apache-2.0), via `scripts/download_multilingual_ipa_lexicons.py`. |

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only pt_pt
```

Writes `data/pt_pt/dict.tsv` and `data/pt_pt/source.txt`. Copy `dict.tsv` into `data/pt_pt/` if needed.
