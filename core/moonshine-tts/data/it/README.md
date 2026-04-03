# Italian — `it`

## Contents

- **`dict.tsv`** — word → IPA for C++ `ItalianRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [CUNY-CL/wikipron](https://github.com/CUNY-CL/wikipron) scrape TSV `ita_latn_broad` (Apache-2.0), normalized to repo TSV by `scripts/download_multilingual_ipa_lexicons.py`. |

*ipa-dict does not ship Italian; this repo fills Italian from WikiPron (see table in that script’s docstring).*

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only it
```

Writes `data/it/dict.tsv` and `data/it/source.txt`. Copy `dict.tsv` into `data/it/` if needed.
