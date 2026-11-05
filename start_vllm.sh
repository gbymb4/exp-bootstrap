#!/bin/bash
# Start vLLM on the experiment pod. Configuration comes from environment variables so the
# same script serves every variant (baseline, FP8 KV, spec decode, ...).
#   MODEL  EXTRA  MAXLEN  SEQS  BAN
# BAN=1 (default) attaches the Han-token logits processor; set BAN=0 on vLLM versions whose
# LogitsProcessor base class no longer matches ban_chinese.py.
set -uo pipefail
MODEL=${MODEL:-/root/exp/models/qwen3-4b-instruct-2507}
EXTRA=${EXTRA:-}
MAXLEN=${MAXLEN:-12288}
SEQS=${SEQS:-12}
BAN=${BAN:-1}

BANARG=""
if [ "$BAN" = "1" ]; then
  BANARG="--logits-processors ban_chinese:BanChineseLogitsProcessor"
fi

pkill -f 'vllm.entrypoints.openai' 2>/dev/null
sleep 3
cd /root/exp
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name base \
  $BANARG \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$SEQS" \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 --api-key benchtmp032 \
  $EXTRA > /root/exp/vllm.log 2>&1 &
echo "started BAN=$BAN EXTRA='$EXTRA' MAXLEN=$MAXLEN SEQS=$SEQS"
