#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible endpoint: TTFT, decode throughput, wall throughput.

Streaming is used so time-to-first-token is genuine rather than the header time the gateway
reports. No external detectors or judges are called — this measures local inference only,
which is the part of the comparison that does not cost money.

  python bench/bench.py --base https://host --key KEY --conc 1 --max-tokens 200 --reps 3

--base may be a gateway (GATEWAY_API_KEY) or a worker (INTERNAL_KEY); both speak the same
API. Optional --metrics-url scrapes vLLM /metrics for KV-cache pressure when reachable.
"""
import argparse
import json
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("requests is not installed; pip install --target <dir> requests")
    sys.exit(2)

PROMPTS = [
    "Rewrite this to sound more human while keeping the meaning: The implementation of the new policy was met with considerable resistance from the stakeholders.",
    "Rewrite naturally: In conclusion, it is evident that the utilization of renewable energy sources constitutes a pivotal component of sustainable development.",
    "Rewrite this paragraph naturally, preserving length and structure: The rapid expansion of urban areas has led to increased demand for transportation infrastructure. Municipal authorities have responded by allocating additional funding toward public transit projects, though critics argue that these measures remain insufficient given projected population growth.",
    "Rewrite more naturally: The committee deliberated extensively before arriving at a decision that balanced competing interests.",
    "Rewrite this to read like a person wrote it: Furthermore, the analysis demonstrates a statistically significant correlation between the variables under consideration.",
    "Rewrite naturally: It is important to note that the aforementioned considerations should be taken into account when formulating future strategies.",
    "Rewrite this sentence more organically: The organization has implemented a comprehensive framework for evaluating employee performance on an annual basis.",
    "Rewrite naturally, keeping all facts: The study enrolled 240 participants between the ages of 18 and 65, and follow-up assessments were conducted at six and twelve months.",
    "Rewrite more naturally: Notwithstanding these limitations, the findings offer valuable insights into the underlying mechanisms.",
    "Rewrite this to sound less automated: The proposed methodology leverages advanced computational techniques to facilitate the optimization of resource allocation.",
]


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct / 100.0))
    return ordered[idx]


def one_request(base, key, model, prompt, max_tokens, temperature, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    started = time.perf_counter()
    ttft = None
    chunks = 0
    with requests.post(base.rstrip("/") + "/v1/chat/completions", json=body,
                       headers={"authorization": "Bearer " + key}, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            return {"status": resp.status_code, "body": resp.text[:200]}
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        data = json.loads(payload)
                    except ValueError:
                        continue
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        chunks += 1
    total = time.perf_counter() - started
    decode_ms = None if ttft is None else (total - ttft) * 1000.0
    tok_per_s = None
    if decode_ms and chunks > 1:
        tok_per_s = (chunks - 1) / (decode_ms / 1000.0)
    return {"status": 200, "ttft_ms": None if ttft is None else ttft * 1000.0,
            "total_ms": total * 1000.0, "chunks": chunks, "tok_per_s": tok_per_s}


def scrape_metrics(url):
    if not url:
        return {}
    try:
        text = requests.get(url, timeout=10).text
    except Exception:
        return {}
    wanted = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in ("vllm:kv_cache_usage_perc", "vllm:num_requests_running",
                     "vllm:num_requests_waiting", "vllm:gpu_cache_usage_perc"):
            if line.startswith(name + "{"):
                wanted[name] = line.rsplit(" ", 1)[-1]
    return wanted


def main():
    ap = argparse.ArgumentParser(description="Local inference benchmark (no paid services).")
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="gzrl3")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--metrics-url", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    before = scrape_metrics(args.metrics_url)
    results = []
    wall_start = time.perf_counter()
    for rep in range(args.reps):
        batch = []
        threads = []
        import concurrent.futures as futures
        with futures.ThreadPoolExecutor(max_workers=args.conc) as pool:
            for i in range(args.conc):
                prompt = PROMPTS[(rep * args.conc + i) % len(PROMPTS)]
                threads.append(pool.submit(one_request, args.base, args.key, args.model,
                                           prompt, args.max_tokens, args.temperature, 300))
            for t in threads:
                batch.append(t.result())
        results.extend(batch)
    wall = time.perf_counter() - wall_start
    after = scrape_metrics(args.metrics_url)

    ok = [r for r in results if r.get("status") == 200]
    failed = [r for r in results if r.get("status") != 200]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    total = [r["total_ms"] for r in ok if r.get("total_ms")]
    tps = [r["tok_per_s"] for r in ok if r.get("tok_per_s")]

    print("label={} base={} conc={} reps={} max_tokens={}".format(
        args.label, args.base, args.conc, args.reps, args.max_tokens))
    print("  requests: {} ok={} failed={}".format(len(results), len(ok), len(failed)))
    if failed:
        print("  failures: {}".format([(f.get("status"), f.get("body", "")[:80]) for f in failed[:3]]))
    if ttft:
        print("  ttft ms:  min={:.0f} p50={:.0f} p95={:.0f} max={:.0f}".format(
            min(ttft), percentile(ttft, 50), percentile(ttft, 95), max(ttft)))
    if total:
        print("  total ms: min={:.0f} p50={:.0f} p95={:.0f} max={:.0f}".format(
            min(total), percentile(total, 50), percentile(total, 95), max(total)))
    if tps:
        print("  decode tok/s per request: min={:.1f} p50={:.1f} p95={:.1f} max={:.1f}".format(
            min(tps), percentile(tps, 50), percentile(tps, 95), max(tps)))
    print("  wall {:.0f}ms, throughput {:.2f} req/s".format(wall * 1000, len(ok) / wall))
    if before or after:
        print("  metrics before: {}".format(before))
        print("  metrics after:  {}".format(after))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"label": args.label, "conc": args.conc, "reps": args.reps,
                       "requests": len(results), "ok": len(ok), "failed": len(failed),
                       "ttft_ms": ttft, "total_ms": total, "tok_per_s": tps,
                       "wall_s": wall, "metrics_before": before, "metrics_after": after}, fh, indent=2)
        print("  wrote {}".format(args.json_out))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
