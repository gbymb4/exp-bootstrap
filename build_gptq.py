#!/usr/bin/env python3
"""Build the GPTQ 4-bit checkpoint with llmcompressor.

W4A16, group 128, lm_head left in higher precision. Calibration uses prompts that are disjoint
from the evaluation set (see make_prompts.py).

  python3 build_gptq.py --base /root/exp/models/qwen3-4b-instruct-2507 \
      --calib /root/exp/prompts/calib-128.jsonl --out /root/exp/quant4
"""
import argparse
import json
import sys


def load_texts(path, limit):
    texts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            value = row.get("text") or row.get("input_text")
            if isinstance(value, str) and len(value) > 100:
                texts.append(value)
            if len(texts) >= limit:
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    texts = load_texts(args.calib, args.limit)
    print("calibration samples:", len(texts))
    if not args.apply:
        print("dry-run: would quantize {} into {}".format(args.base, args.out))
        return 0

    import datasets
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    dataset = datasets.Dataset.from_dict({"text": texts})
    recipe = GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
    oneshot(model=model, dataset=dataset, recipe=recipe,
            num_calibration_samples=len(texts), max_seq_length=args.max_seq_length)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)
    print("saved compressed checkpoint to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
