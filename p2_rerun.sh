#!/bin/bash
# p2 重跑：3 target-mode × (epochs3 / epochs2)，每个 train + eval-only
# 修正：venv python、--model-path（默认是 Mac 路径）、HF_HUB_OFFLINE、pipefail
set -o pipefail
cd /root/modernbert-rlcd-ablation
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p runs/logs

declare -A RC
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee runs/logs/${tag}.log
  RC[$tag]=$?
  echo "[exit ${RC[$tag]}] $tag"
}

for m in hard raw log1p; do
  run p2_train_${m} $PY -u train_soft_ablation.py --target-mode $m --calib-pairs 600 --seed 0 \
    --amp --max-tokens 16000 --save-model --out-root runs/p2 \
    --model-path answerdotai/ModernBERT-base
  run p2_eval_${m}  $PY -u train_soft_ablation.py --target-mode $m --calib-pairs 600 \
    --out-root runs/p2 --eval-only \
    --model-path answerdotai/ModernBERT-base
  run p2e2_train_${m} $PY -u train_soft_ablation.py --target-mode $m --epochs 2 --calib-pairs 600 --seed 0 \
    --amp --max-tokens 16000 --save-model --out-root runs/p2e2 \
    --model-path answerdotai/ModernBERT-base
  run p2e2_eval_${m}  $PY -u train_soft_ablation.py --target-mode $m --epochs 2 --calib-pairs 600 \
    --out-root runs/p2e2 --eval-only \
    --model-path answerdotai/ModernBERT-base
done

echo "=== ALL DONE $(date '+%F %T') ==="
for k in p2_train_hard p2_eval_hard p2e2_train_hard p2e2_eval_hard \
         p2_train_raw p2_eval_raw p2e2_train_raw p2e2_eval_raw \
         p2_train_log1p p2_eval_log1p p2e2_train_log1p p2e2_eval_log1p; do
  echo "RC[$k]=${RC[$k]}"
done
