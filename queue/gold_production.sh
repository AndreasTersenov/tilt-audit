#!/bin/bash
# Gold-standard production: gate on the T-L1 ladder chain (PID $1) exiting,
# then 16 confirmatory configs + T-L3 second seed, sequential on GPU 1.
until ! kill -0 $1 2>/dev/null; do sleep 5; done
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /mnt/home/tersenov/software/tilt-audit
for n in 64 32; do
  for t in mid strong; do
    for y in 0 1 2 3; do
      .venv/bin/python scripts/run_gold.py --n $n --tilt $t --yseed $y \
        >> logs/gold_production.log 2>&1
    done
  done
done
# T-L3: independent seed on the flagship config
.venv/bin/python scripts/run_gold.py --n 64 --tilt mid --yseed 0 --seed 1 \
  --tag tl3 >> logs/gold_production.log 2>&1
echo "GOLD PRODUCTION DONE $(date -u +%H:%M)" >> logs/gold_production.log
