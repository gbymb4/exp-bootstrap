#!/usr/bin/env python3
"""Build calibration and evaluation prompt sets from a public text dataset.

The proprietary corpora live on a volume that is currently unreachable, so this uses public
essay-style English text (WikiText) as a stand-in. Two disjoint sets are written:
calib-128.jsonl (quantization calibration) and eval-100.jsonl (quality comparison prompts).
Disjointness matters: calibrating on the same prompts you later measure would flatter the
result.
"""
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "Salesforce/wikitext"
FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"


def main():
    path = hf_hub_download(repo_id=REPO, filename=FILE, repo_type="dataset")
    table = pq.read_table(path, columns=["text"])
    lines = table.column("text").to_pylist()

    buf = ""
    paras = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            if buf:
                paras.append(buf)
                buf = ""
            continue
        buf += (" " if buf else "") + line
        if len(buf) > 700:
            paras.append(buf)
            buf = ""
    if buf:
        paras.append(buf)

    usable = [p for p in paras if 400 <= len(p) <= 4000]
    print("candidate paragraphs:", len(usable))

    os.makedirs("/root/exp/prompts", exist_ok=True)
    for name, start, count in (("calib-128", 0, 128), ("eval-100", 128, 100)):
        chunk = usable[start:start + count]
        with open("/root/exp/prompts/{}.jsonl".format(name), "w") as fh:
            for text in chunk:
                fh.write(json.dumps({"text": text}) + "\n")
        lengths = sorted(len(t) for t in chunk)
        print("{}: {} prompts, chars min={} median={} max={}".format(
            name, len(chunk), lengths[0], lengths[len(lengths) // 2], lengths[-1]))


if __name__ == "__main__":
    main()
