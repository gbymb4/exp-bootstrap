#!/usr/bin/env python3
"""Score every captured condition with the *real* detectors and the existing LLM-judge pipeline.

Two halves, runnable independently:

  --detector gptzero|zerogpt   calls the commercial AI-text detectors
  --judge anthropic            calls claude-sonnet-5 with qa-scoring's fidelity + quality rubrics

Results are written per condition with the per-example scores, so paired deltas against the
bf16 baseline can be computed afterwards (see the summary this prints).

  python3 score_real.py --outdir /root/exp/out --baseline bf16-full \
      --detector gptzero --judge anthropic
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

RUBRIC_DIR = "/root/exp/rubrics"


# ----------------------------------------------------------------------------- detectors

GPTZERO_VERSION = os.environ.get("GPTZERO_VERSION", "2026-11-04-base")


def gptzero_score(text, key, timeout=120):
    resp = requests.post(
        "https://api.gptzero.me/v2/predict/text",
        headers={"x-api-key": key, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json={"document": text, "version": GPTZERO_VERSION, "multilingual": False},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return None, "http {}".format(resp.status_code), resp.text[:200]
    data = resp.json()
    docs = data.get("documents") or []
    if not docs:
        return None, "no documents", json.dumps(data)[:200]
    probs = docs[0].get("class_probabilities") or {}
    ai = probs.get("ai")
    if ai is None:
        ai = docs[0].get("average_generated_prob")
    return ai, "ok", ""


def zerogpt_score(text, key, timeout=120):
    resp = requests.post(
        "https://api.zerogpt.com/api/detect/detectText",
        headers={"ApiKey": key, "Content-Type": "application/json"},
        json={"input_text": text},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return None, "http {}".format(resp.status_code), resp.text[:200]
    data = resp.json()
    payload = data.get("data") or {}
    score = payload.get("score")
    if score is None:
        score = payload.get("fake")
    if score is not None:
        score = float(score) / 100.0 if float(score) > 1 else float(score)
    return score, "ok", ""


DETECTORS = {"gptzero": gptzero_score, "zerogpt": zerogpt_score}


# ----------------------------------------------------------------------------- judge

def read_rubric(name):
    with open(os.path.join(RUBRIC_DIR, name)) as fh:
        return fh.read()


def read_schema(name):
    with open(os.path.join(RUBRIC_DIR, name)) as fh:
        raw = json.load(fh)
    return strip_constraints(raw.get("schema", raw))


def strip_constraints(value):
    if isinstance(value, list):
        return [strip_constraints(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {k: strip_constraints(v) for k, v in value.items()
            if k not in ("minimum", "maximum", "multipleOf")}


def judge_pair(original, rewritten, key, model, timeout=180):
    fidelity_prompt = read_rubric("llm_judge_fidelity_v2.txt")
    quality_prompt = read_rubric("llm_judge_quality v2.txt")
    fidelity_schema = read_schema("llm_judge_fidelity_schema.json")
    quality_schema = read_schema("llm_judge_quality_schema_v2.json")

    def call(system, user, schema):
        body = {
            "model": model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json=body, timeout=timeout)
        if r.status_code != 200:
            return None, "http {} {}".format(r.status_code, r.text[:160])
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        try:
            return json.loads(text), "ok"
        except ValueError:
            return None, "unparseable: " + text[:160]

    fidelity, err1 = call(fidelity_prompt,
                          "Original Text:\n\n{}\n\nRewritten Text:\n\n{}".format(original, rewritten),
                          fidelity_schema)
    quality, err2 = call(quality_prompt, rewritten, quality_schema)
    return {"fidelity": fidelity, "quality": quality}, "; ".join(e for e in (err1, err2) if e)


# ----------------------------------------------------------------------------- main

def load(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows[rec["id"]] = rec
    return rows


def summarise(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    vals = sorted(vals)
    return {"min": round(vals[0], 4), "median": round(vals[len(vals) // 2], 4),
            "max": round(vals[-1], 4), "mean": round(sum(vals) / len(vals), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/root/exp/out")
    ap.add_argument("--baseline", default="bf16-full")
    ap.add_argument("--detector", choices=list(DETECTORS), default=None)
    ap.add_argument("--judge", choices=["anthropic"], default=None)
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3, help="pause between detector calls")
    args = ap.parse_args()

    det_key = os.environ.get(args.detector.upper() + "_API_KEY") if args.detector else None
    judge_key = os.environ.get("ANTHROPIC_API_KEY") if args.judge else None
    if args.detector and not det_key:
        print("missing", args.detector.upper() + "_API_KEY")
        return 1
    if args.judge and not judge_key:
        print("missing ANTHROPIC_API_KEY")
        return 1

    conditions = sorted(f[:-6] for f in os.listdir(args.outdir) if f.endswith(".jsonl"))
    base = load(os.path.join(args.outdir, args.baseline + ".jsonl"))
    report = {}

    for cond in conditions:
        data = load(os.path.join(args.outdir, cond + ".jsonl"))
        common = sorted(set(base) & set(data))
        if args.limit:
            common = common[:args.limit]
        det_scores, judge_fid, judge_qual, errors = [], [], [], 0
        det_by_id, fid_by_id, qual_by_id = {}, {}, {}

        for idx in common:
            rec = data[idx]
            if args.detector:
                score, status, detail = DETECTORS[args.detector](rec["output"], det_key)
                if score is None:
                    errors += 1
                    print("  detector error id={} {} {}".format(idx, status, detail))
                else:
                    det_scores.append(score)
                    det_by_id[str(idx)] = score
                time.sleep(args.sleep)
            if args.judge:
                verdict, err = judge_pair(base[idx]["prompt"], rec["output"], judge_key, args.judge_model)
                if verdict and verdict.get("fidelity"):
                    judge_fid.append(verdict["fidelity"].get("overall_score"))
                    fid_by_id[str(idx)] = verdict["fidelity"].get("overall_score")
                if verdict and verdict.get("quality"):
                    judge_qual.append(verdict["quality"].get("overall_score"))
                    qual_by_id[str(idx)] = verdict["quality"].get("overall_score")
                if err and err != "; ":
                    errors += 1
                    print("  judge error id={} {}".format(idx, err))

        entry = {
            "pairs": len(common),
            "detector": args.detector,
            "detector_scores": summarise(det_scores),
            "detector_errors": errors if args.detector else 0,
            "judge_fidelity": summarise(judge_fid),
            "judge_quality": summarise(judge_qual),
            # Per-example scores: the point of the exercise is paired deltas against the
            # baseline, which needs each example's score rather than a summary.
            "per_example_detector": det_by_id,
            "per_example_fidelity": fid_by_id,
            "per_example_quality": qual_by_id,
        }
        report[cond] = entry
        print("== {} ({} pairs)".format(cond, len(common)))
        if args.detector:
            print("   detector AI-probability:", entry["detector_scores"])
        if args.judge:
            print("   judge fidelity  :", entry["judge_fidelity"])
            print("   judge quality   :", entry["judge_quality"])

    name = "score_real_{}{}.json".format(args.detector or "none", "_judge" if args.judge else "")
    out_path = os.path.join(args.outdir, name)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
