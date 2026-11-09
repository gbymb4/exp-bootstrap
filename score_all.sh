#!/bin/bash
# Full scoring pass over the ban-on matrix outputs: both commercial detectors, then the
# Anthropic LLM judge with qa-scoring's rubrics.
set -a
. /root/exp/bashrc_api_keys.env
set +a
cd /root/exp

echo "gptzero start $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log
python3 score_real.py --outdir /root/exp/out_ban --baseline bf16-full --detector gptzero \
  > /root/exp/gptzero_full.log 2>&1
echo "gptzero done $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log

echo "zerogpt start $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log
python3 score_real.py --outdir /root/exp/out_ban --baseline bf16-full --detector zerogpt \
  > /root/exp/zerogpt_full.log 2>&1
echo "zerogpt done $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log

echo "judge start $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log
python3 score_real.py --outdir /root/exp/out_ban --baseline bf16-full --judge anthropic \
  > /root/exp/judge_full.log 2>&1
echo "judge done $(date -u +%H:%M:%S)" >> /root/exp/scoring_times.log

echo SCORING_DONE >> /root/exp/overnight.log
