#!/usr/bin/env python3
"""Detector proxy: score every condition's outputs with an open AI-text detector.

The paid detectors (GPTZero, ZeroGPT, judges) need budget approval, so this uses a public
RoBERTa AI-text detector as a stand-in. It is not GPTZero and its absolute numbers are not
comparable to production pass rates -- but it answers the question the matrix exists to answer:
does a configuration produce text that is measurably *more detectable* than the bf16 baseline?

Reports, per condition, the mean fake-probability and the paired per-example delta against the
baseline, because the question is whether a config loses reliably, not whether a mean moves.

  python3 detector_proxy.py --outdir /root/exp/out --baseline bf16-full
"""
import argparse
import json
import os
import statistics
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "openai-community/roberta-base-openai-detector"


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r["id"]: r for r in rows}


def score_batch(model, tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1)
    # Use the model's own label mapping rather than assuming an index: this checkpoint reports
    # {0: 'Fake', 1: 'Real'}, which is the opposite of the obvious guess.
    fake_idx = model.config.label2id.get("Fake", 0)
    return probs[:, fake_idx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/root/exp/out")
    ap.add_argument("--baseline", default="bf16-full")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    print("detector:", MODEL, "on", device, "labels:", model.config.id2label)

    base = load(os.path.join(args.outdir, args.baseline + ".jsonl"))
    report = {}
    for name in sorted(f[:-6] for f in os.listdir(args.outdir) if f.endswith(".jsonl")):
        data = load(os.path.join(args.outdir, name + ".jsonl"))
        common = sorted(set(base) & set(data))
        texts = [data[i]["output"] for i in common]
        scores = []
        for start in range(0, len(texts), args.batch_size):
            scores.extend(score_batch(model, tokenizer, texts[start:start + args.batch_size], device))
        base_scores = []
        base_texts = [base[i]["output"] for i in common]
        for start in range(0, len(base_texts), args.batch_size):
            base_scores.extend(score_batch(model, tokenizer, base_texts[start:start + args.batch_size], device))

        deltas = [s - b for s, b in zip(scores, base_scores)]
        report[name] = {
            "pairs": len(common),
            "fake_prob_mean": round(statistics.mean(scores), 4),
            "fake_prob_median": round(statistics.median(scores), 4),
            "fake_prob_ge_0_5": sum(1 for s in scores if s >= 0.5),
            "paired_delta_mean": round(statistics.mean(deltas), 4),
            "paired_delta_median": round(statistics.median(deltas), 4),
            "deltas_worse": sum(1 for d in deltas if d > 0.05),
            "deltas_better": sum(1 for d in deltas if d < -0.05),
        }
        print("== {} ({} pairs)".format(name, len(common)))
        print("   fake prob mean/median : {} / {}   (>=0.5: {})".format(
            report[name]["fake_prob_mean"], report[name]["fake_prob_median"],
            report[name]["fake_prob_ge_0_5"]))
        print("   paired delta vs base  : mean {} median {}  | worse {} better {}".format(
            report[name]["paired_delta_mean"], report[name]["paired_delta_median"],
            report[name]["deltas_worse"], report[name]["deltas_better"]))

    out = os.path.join(args.outdir, "detector_proxy.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
