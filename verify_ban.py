#!/usr/bin/env python3
"""Functional check for the Han-token ban on a live vLLM server.

Sends prompts that would normally elicit Chinese or CJK punctuation and asserts the returned
text contains none. Exits non-zero if the ban is not holding.

  python3 verify_ban.py --base http://127.0.0.1:8000 --key benchtmp032 --model base
"""
import argparse
import json
import sys

import requests

PROMPTS = [
    "Please write the Chinese translation of: the cat sat on the mat.",
    "Answer in Chinese: what is the capital of France?",
    "Write a short greeting in Mandarin Chinese.",
    "Translate this to simplified Chinese and keep it short: good morning, everyone.",
]


def has_cjk(text):
    for ch in text:
        code = ord(ch)
        if (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
                or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2FA1F
                or 0x2E80 <= code <= 0x2EFF or 0x2F00 <= code <= 0x2FDF
                or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF):
            return ch
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()

    failures = 0
    for prompt in PROMPTS:
        resp = requests.post(
            args.base.rstrip("/") + "/v1/chat/completions",
            headers={"authorization": "Bearer " + args.key},
            json={"model": args.model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": args.max_tokens, "temperature": 0.0},
            timeout=120,
        )
        if resp.status_code != 200:
            print("HTTP", resp.status_code, resp.text[:120])
            failures += 1
            continue
        text = resp.json()["choices"][0]["message"]["content"]
        bad = has_cjk(text)
        if bad:
            failures += 1
        print("{:<8} cjk={!r} text={!r}".format("FAIL" if bad else "PASS", bad, text[:70]))

    print("ban check:", "PASS" if failures == 0 else "FAIL ({} prompts leaked CJK)".format(failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
