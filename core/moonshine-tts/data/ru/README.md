# Russian — `ru`

## Contents

- **`dict.tsv`** — word → IPA for C++ `RussianRuleG2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [CUNY-CL/wikipron](https://github.com/CUNY-CL/wikipron) — `rus_cyrl_narrow` scrape (Apache-2.0). WikiPron labels this as “narrow” relative to Wiktionary; the repo uses it as the Russian lexicon source. Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only ru
```

Writes `data/ru/dict.tsv` and `data/ru/source.txt`. Copy `dict.tsv` into `data/ru/` if needed.
