#!/bin/bash
# Bootstrap an experiment pod: fetch the model files it needs and make the Han-token ban
# importable. Run once per pod; safe to re-run.
set -uo pipefail
GIST=https://gist.githubusercontent.com/gbymb4/16e7ecbccfc7f19ac85b219ec98796ac/raw
mkdir -p /root/exp/models /root/exp/code

echo "== python"
python3 -V

echo "== fetch helper files"
curl -sfL "$GIST/ban_chinese.py" -o /root/exp/code/ban_chinese.py && echo "ban_chinese ok"
curl -sfL "$GIST/bench.py" -o /root/exp/bench.py && echo "bench ok"

echo "== model files (base 4B + draft 0.6B)"
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir="/root/exp/models/qwen3-4b-instruct-2507")
print("base done")
snapshot_download("Qwen/Qwen3-0.6B", local_dir="/root/exp/models/qwen3-0.6b")
print("draft done")
PY

echo "== make ban importable"
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
cp /root/exp/code/ban_chinese.py "$SP/ban_chinese.py"
python3 -c "import ban_chinese; print('ban import ok')"

echo "== sizes"
du -sh /root/exp/models/* 2>/dev/null
df -h / | tail -1
echo "== bootstrap done"
