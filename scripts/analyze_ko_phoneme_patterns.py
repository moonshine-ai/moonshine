#!/usr/bin/env python3
"""
Analyze systematic phoneme pattern differences between Moonshine and Piper Korean G2P.

Reads the per-word intelligibility report JSON (run with --per-word first) and extracts
recurring phoneme substitution patterns to guide rule improvements.

Usage:
    # First generate per-word data:
    python scripts/tts_g2p_intelligibility.py \
        --wiki-text scripts/data/wiki-text/ko_conversational.txt \
        --no-cache --languages ko --max-lines-per-lang 20 --per-word

    # Then analyze:
    python scripts/analyze_ko_phoneme_patterns.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPORT = Path("scripts/data/tts_g2p_intelligibility_report.json")


def levenshtein_alignment(s: str, t: str):
    """Return (distance, list of (op, s_char, t_char)) edit operations."""
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if s[i - 1] == t[j - 1] else 1):
            if s[i - 1] == t[j - 1]:
                ops.append(("match", s[i - 1], t[j - 1]))
            else:
                ops.append(("sub", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", s[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", t[j - 1]))
            j -= 1
    ops.reverse()
    return dp[n][m], ops


def extract_substitution_patterns(ops, context_size=2):
    """Extract substitution/insertion/deletion patterns with surrounding context."""
    patterns = []
    for idx, (op, sc, tc) in enumerate(ops):
        if op == "match":
            continue
        # Get context
        left = "".join(o[1] or o[2] for o in ops[max(0, idx - context_size):idx])
        right = "".join(o[1] or o[2] for o in ops[idx + 1:idx + 1 + context_size])
        patterns.append({
            "op": op,
            "moon": sc,
            "piper": tc,
            "left_ctx": left,
            "right_ctx": right,
        })
    return patterns


def classify_phoneme(ch):
    """Classify a single IPA character into a broad category."""
    vowels = set("aeiouɐɛɯʌɔwjɪ")
    stress = set("ˈˌ")
    stops = set("kptbdɡq")
    if ch in stress:
        return "stress"
    if ch in vowels:
        return "vowel"
    if ch in stops:
        return "stop"
    if ch in "-":
        return "boundary"
    return "consonant"


def main():
    if not _REPORT.is_file():
        print(f"Report not found: {_REPORT}", file=sys.stderr)
        print("Run: python scripts/tts_g2p_intelligibility.py --wiki-text scripts/data/wiki-text/ko_conversational.txt --no-cache --languages ko --max-lines-per-lang 20 --per-word", file=sys.stderr)
        return 1

    data = json.loads(_REPORT.read_text(encoding="utf-8"))

    # Find Korean data
    ko = None
    for key in ["ko", "ko_kr"]:
        if key in data.get("languages", {}):
            ko = data["languages"][key]
            break
    if ko is None:
        print("No Korean data in report", file=sys.stderr)
        return 1

    lines = ko.get("lines", [])
    if not lines:
        print("No line data in report", file=sys.stderr)
        return 1

    print(f"Analyzing {len(lines)} entries...\n")

    # Collect all substitution patterns
    all_subs = Counter()       # (moon_char, piper_char) -> count
    all_deletions = Counter()  # moon_char -> count (present in moon, absent in piper)
    all_insertions = Counter() # piper_char -> count (absent in moon, present in piper)
    all_patterns = []
    total_lev = 0
    total_entries = 0
    entries_with_piper = 0

    # Per-word tracking
    high_lev_words = []

    for entry in lines:
        moon = entry.get("moonshine_g2p_ipa", "")
        piper = entry.get("upstream_piper_package_phonemes", "")
        gt = entry.get("ground_truth", "")
        lev = entry.get("phoneme_levenshtein_vs_piper")

        if not moon or not piper:
            continue
        entries_with_piper += 1

        if lev is None:
            lev, _ = levenshtein_alignment(moon, piper)

        total_lev += lev
        total_entries += 1

        dist, ops = levenshtein_alignment(moon, piper)
        patterns = extract_substitution_patterns(ops)
        all_patterns.extend(patterns)

        for p in patterns:
            if p["op"] == "sub":
                all_subs[(p["moon"], p["piper"])] += 1
            elif p["op"] == "del":
                all_deletions[p["moon"]] += 1
            elif p["op"] == "ins":
                all_insertions[p["piper"]] += 1

        if lev >= 3:
            high_lev_words.append((gt, moon, piper, lev))

    print(f"Entries with Piper phonemes: {entries_with_piper}")
    print(f"Total Levenshtein distance: {total_lev}")
    print(f"Average Levenshtein per word: {total_lev / total_entries:.2f}" if total_entries else "N/A")
    print()

    # Top substitutions
    print("=" * 70)
    print("TOP SUBSTITUTIONS (Moonshine char -> Piper char)")
    print("=" * 70)
    for (mc, pc), count in all_subs.most_common(30):
        cat_m = classify_phoneme(mc)
        cat_p = classify_phoneme(pc)
        print(f"  {mc!r:6s} -> {pc!r:6s}  count={count:4d}  ({cat_m} -> {cat_p})")

    print()
    print("=" * 70)
    print("TOP DELETIONS (present in Moonshine, missing in Piper)")
    print("=" * 70)
    for ch, count in all_deletions.most_common(20):
        print(f"  {ch!r:6s}  count={count:4d}  ({classify_phoneme(ch)})")

    print()
    print("=" * 70)
    print("TOP INSERTIONS (missing in Moonshine, present in Piper)")
    print("=" * 70)
    for ch, count in all_insertions.most_common(20):
        print(f"  {ch!r:6s}  count={count:4d}  ({classify_phoneme(ch)})")

    # Contextual patterns for top substitutions
    print()
    print("=" * 70)
    print("CONTEXTUAL PATTERNS (top substitutions with surrounding context)")
    print("=" * 70)
    ctx_patterns = defaultdict(list)
    for p in all_patterns:
        if p["op"] == "sub":
            key = (p["moon"], p["piper"])
            ctx_patterns[key].append(f"...{p['left_ctx']}[{p['moon']}->{p['piper']}]{p['right_ctx']}...")

    for (mc, pc), count in all_subs.most_common(10):
        examples = ctx_patterns[(mc, pc)][:5]
        print(f"\n  {mc!r} -> {pc!r} (count={count}):")
        for ex in examples:
            print(f"    {ex}")

    # High-Levenshtein words
    print()
    print("=" * 70)
    print("HIGHEST LEVENSHTEIN WORDS (distance >= 3)")
    print("=" * 70)
    high_lev_words.sort(key=lambda x: -x[3])
    for gt, moon, piper, lev in high_lev_words[:20]:
        print(f"  [{gt}] lev={lev}")
        print(f"    moon:  {moon}")
        print(f"    piper: {piper}")
        print()

    # Summary of actionable categories
    print("=" * 70)
    print("ACTIONABLE SUMMARY")
    print("=" * 70)

    # Group substitutions by category
    stress_edits = sum(c for (m, p), c in all_subs.items() if classify_phoneme(m) == "stress" or classify_phoneme(p) == "stress")
    stop_edits = sum(c for (m, p), c in all_subs.items() if classify_phoneme(m) == "stop" or classify_phoneme(p) == "stop")
    vowel_edits = sum(c for (m, p), c in all_subs.items() if classify_phoneme(m) == "vowel" and classify_phoneme(p) == "vowel")
    boundary_edits = sum(c for (m, p), c in all_subs.items() if classify_phoneme(m) == "boundary" or classify_phoneme(p) == "boundary")

    total_sub_edits = sum(all_subs.values())
    total_del_edits = sum(all_deletions.values())
    total_ins_edits = sum(all_insertions.values())
    total_all = total_sub_edits + total_del_edits + total_ins_edits

    print(f"\n  Total edit operations: {total_all}")
    print(f"    Substitutions: {total_sub_edits} ({100*total_sub_edits/total_all:.0f}%)")
    print(f"    Deletions:     {total_del_edits} ({100*total_del_edits/total_all:.0f}%)")
    print(f"    Insertions:    {total_ins_edits} ({100*total_ins_edits/total_all:.0f}%)")
    print(f"\n  Substitution breakdown:")
    print(f"    Stress-related:   {stress_edits}")
    print(f"    Stop consonants:  {stop_edits}")
    print(f"    Vowel changes:    {vowel_edits}")
    print(f"    Boundary markers: {boundary_edits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
