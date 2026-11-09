#!/usr/bin/env python3
"""Per-example paired deltas for the detector and judge scores.

score_real.py records each example's score; this computes, for every condition against the
bf16 baseline, the mean/median delta and how many examples were measurably worse or better.
That is the question the matrix exists to answer: does a configuration lose something reliably,
not whether an average drifts.

  python3 paired_deltas.py --outdir /root/exp/out_ban --baseline bf16-full
"""
import argparse
import glob
import json
import os
import statistics

FIELDS = {
    "detector": ("per_example_detector", "higher_is_worse"),
    "fidelity": ("per_example_fidelity", "higher_is_better"),
    "quality": ("per_example_quality", "higher_is_better"),
}
THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/root/exp/out_ban")
    ap.add_argument("--baseline", default="bf16-full")
    ap.add_argument("--report", default=None, help="write JSON here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.outdir, "score_real_*.json")))
    if not files:
        print("no score_real_*.json in", args.outdir)
        return 1

    report = {}
    for path in files:
        with open(path) as fh:
            scores = json.load(fh)
        base = scores.get(args.baseline, {})
        label = os.path.basename(path)[:-5]
        report[label] = {}
        print("== {}".format(label))
        for field, (key, direction) in FIELDS.items():
            base_map = base.get(key) or {}
            if not base_map:
                continue
            for cond, entry in sorted(scores.items()):
                if cond == args.baseline:
                    continue
                cond_map = entry.get(key) or {}
                common = sorted(set(base_map) & set(cond_map), key=lambda x: int(x))
                if not common:
                    continue
                deltas = [cond_map[i] - base_map[i] for i in common]
                if direction == "higher_is_worse":
                    worse = sum(1 for d in deltas if d > THRESHOLD)
                    better = sum(1 for d in deltas if d < -THRESHOLD)
                else:
                    worse = sum(1 for d in deltas if d < -THRESHOLD)
                    better = sum(1 for d in deltas if d > THRESHOLD)
                summary = {
                    "pairs": len(common),
                    "delta_mean": round(statistics.mean(deltas), 4),
                    "delta_median": round(statistics.median(deltas), 4),
                    "worse": worse,
                    "better": better,
                    "unchanged": len(deltas) - worse - better,
                    "baseline_mean": round(statistics.mean(base_map[i] for i in common), 4),
                    "condition_mean": round(statistics.mean(cond_map[i] for i in common), 4),
                }
                report[label].setdefault(field, {})[cond] = summary
                print("  {:<9} {:<12} pairs={:<3} base={:<7} cond={:<7} delta_mean={:<8} median={:<8} worse={:<3} better={:<3}".format(
                    field, cond, summary["pairs"], summary["baseline_mean"], summary["condition_mean"],
                    summary["delta_mean"], summary["delta_median"], worse, better))

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
