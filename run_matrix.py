#!/usr/bin/env python3
"""Run the four-condition matrix: {bf16, 4-bit} weights x {full, FP8} KV cache.

For each condition it starts vLLM, waits for readiness, generates one output per evaluation
prompt (deterministic: temperature 0), and saves the outputs to /root/exp/out/<condition>.jsonl
so that scoring can be done once across all conditions.

  python3 run_matrix.py --prompts /root/exp/prompts/eval-100.jsonl --max-tokens 512
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

BASE = "/root/exp/models/qwen3-4b-instruct-2507"
QUANT = "/root/exp/quant4"
OUTDIR = "/root/exp/out"
KEY = "benchtmp032"
PORT = 8000

CONDITIONS = [
    ("bf16-full", BASE, None),
    ("bf16-fp8kv", BASE, "fp8"),
    ("4bit-full", QUANT, None),
    ("4bit-fp8kv", QUANT, "fp8"),
]


def gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
    except Exception:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def kill_gpu():
    for pid in gpu_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)


def wait_ready(timeout=420):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://127.0.0.1:{}/v1/models".format(PORT),
                             headers={"authorization": "Bearer " + KEY}, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_server(model, kv_dtype, log_path):
    if os.path.isdir(model):
        pass
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", "base",
            "--max-model-len", "12288", "--max-num-seqs", "12",
            "--gpu-memory-utilization", "0.90",
            "--host", "127.0.0.1", "--port", str(PORT), "--api-key", KEY]
    if kv_dtype:
        args += ["--kv-cache-dtype", kv_dtype]
    if model == QUANT:
        args += ["--quantization", "compressed-tensors"]
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc


def generate(prompt, max_tokens):
    body = {"model": "base", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    started = time.time()
    resp = requests.post("http://127.0.0.1:{}/v1/chat/completions".format(PORT),
                         headers={"authorization": "Bearer " + KEY}, json=body, timeout=600)
    elapsed = time.time() - started
    if resp.status_code != 200:
        return None, elapsed, resp.status_code
    data = resp.json()
    choice = data["choices"][0]
    text = choice["message"]["content"]
    usage = data.get("usage", {})
    return text, elapsed, usage.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/root/exp/prompts/eval-100.jsonl")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--conditions", default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prompts = []
    with open(args.prompts) as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["text"])
    if args.limit:
        prompts = prompts[:args.limit]
    print("prompts:", len(prompts))

    wanted = set(args.conditions.split(",")) if args.conditions else None
    os.makedirs(args.outdir, exist_ok=True)

    for name, model, kv in CONDITIONS:
        if wanted and name not in wanted:
            continue
        if name.startswith("4bit") and not os.path.isdir(QUANT):
            print("SKIP {}: no quantized checkpoint".format(name))
            continue
        print("== condition", name)
        kill_gpu()
        log_path = "/root/exp/vllm-{}.log".format(name)
        start_server(model, kv, log_path)
        if not wait_ready():
            print("FAILED to start", name, "see", log_path)
            kill_gpu()
            continue

        records = []
        total_tokens = 0
        failures = 0
        for i, prompt in enumerate(prompts):
            text, elapsed, tokens = generate(prompt, args.max_tokens)
            if text is None:
                failures += 1
                print("  prompt", i, "failed")
                continue
            total_tokens += tokens if isinstance(tokens, int) else 0
            records.append({"id": i, "prompt": prompt, "output": text,
                            "elapsed_s": round(elapsed, 3), "tokens": tokens})
            if (i + 1) % 20 == 0:
                print("  {}/{}".format(i + 1, len(prompts)))

        path = os.path.join(args.outdir, name + ".jsonl")
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print("  saved {} records -> {} (failures {})".format(len(records), path, failures))
        kill_gpu()

    print("matrix complete")


if __name__ == "__main__":
    main()
