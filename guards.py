#!/usr/bin/env python3
"""Deterministic quality guards for the matrix, with per-example paired deltas.

Compares every condition against the bf16/full-KV baseline on the same prompts. These checks
need no paid API, so they can run before any detector budget is spent.

Per pair: length ratio, CJK leakage, repeated 4-gram rate, number/entity preservation, and a
cheap character-level similarity. Reports distributions, not just means, because the question
is whether a configuration loses something *reliably*.

  python3 guards.py --outdir /root/exp/out --baseline bf16-full
"""
import argparse
import json
import os
import re
import statistics
import sys

CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
              (0x20000, 0x2FA1F), (0x2E80, 0x2EFF), (0x2F00, 0x2FDF),
              (0x3000, 0x303F), (0xFF00, 0xFFEF))


def has_cjk(text):
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in CJK_RANGES)


def repeated_ngram_rate(text, n=4):
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def numbers(text):
    return sorted(re.findall(r"\d+(?:[.,]\d+)?", text))


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def jaccard(a, b):
    sa = set(re.findall(r"[a-z']+", a.lower()))
    sb = set(re.findall(r"[a-z']+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def summarise(values):
    if not values:
        return None
    values = sorted(values)
    return {"min": round(values[0], 4),
            "median": round(values[len(values) // 2], 4),
            "max": round(values[-1], 4),
            "mean": round(sum(values) / len(values), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/root/exp/out")
    ap.add_argument("--baseline", default="bf16-full")
    args = ap.parse_args()

    base_path = os.path.join(args.outdir, args.baseline + ".jsonl")
    if not os.path.exists(base_path):
        print("missing baseline", base_path)
        return 1
    base = load(base_path)
    conditions = sorted(f[:-6] for f in os.listdir(args.outdir) if f.endswith(".jsonl"))

    report = {}
    for cond in conditions:
        path = os.path.join(args.outdir, cond + ".jsonl")
        data = load(path)
        common = sorted(set(base) & set(data))
        if not common:
            continue
        length_ratio, cjk_count, rep_delta, num_lost, similarity = [], 0, [], 0, []
        for idx in common:
            b_text = base[idx]["output"]
            c_text = data[idx]["output"]
            length_ratio.append(len(c_text) / max(1, len(b_text)))
            if has_cjk(c_text):
                cjk_count += 1
            rep_delta.append(repeated_ngram_rate(c_text) - repeated_ngram_rate(b_text))
            b_nums, c_nums = numbers(b_text), numbers(c_text)
            num_lost += len([n for n in b_nums if n not in c_nums])
            similarity.append(jaccard(b_text, c_text))
        entry = {
            "pairs": len(common),
            "length_ratio": summarise(length_ratio),
            "cjk_outputs": cjk_count,
            "repeated_4gram_delta": summarise(rep_delta),
            "numbers_missing_vs_baseline": num_lost,
            "word_jaccard_vs_baseline": summarise(similarity),
            "outputs": len(data),
        }
        report[cond] = entry
        print("== {} ({} pairs)".format(cond, len(common)))
        print("   length ratio   :", entry["length_ratio"])
        print("   cjk outputs    :", cjk_count)
        print("   repeat 4-gram  :", entry["repeated_4gram_delta"])
        print("   numbers lost   :", num_lost)
        print("   word jaccard   :", entry["word_jaccard_vs_baseline"])

    out_path = os.path.join(args.outdir, "guards.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
