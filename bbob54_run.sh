#!/bin/bash
# bbob54 重训：BBOB+BBOB-noisy 54函数 × 3instance × 100LHS点（D5 训练，rep3 D5/10/20 评测）
# 3 target-mode × epochs=2（p2e2 口径）；数据已随仓库提交（data_bbob54/，12MB，无需重新生成）
# 规模：106 record × 500 对 = 5.3 万对/epoch（旧 4 函数数据的 ~18 倍），4090 上每 mode 约 40-60 分钟
# 冒烟（~3 分钟）：先跑
#   $PY -u train_soft_ablation.py --target-mode raw --data-dir data_bbob54 \
#     --max-steps 20 --eval-max-records 1 --pairs-per-record 120 --test-pairs 100 --save-model
set -o pipefail
cd /root/modernbert-rlcd-ablation
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p runs/logs

# 数据兜底：仓库里 data_bbob54/ 缺失时现场生成（需 coco-experiment）
if [ ! -d data_bbob54 ]; then
  $PY -c "import cocoex" 2>/dev/null || $PY -m pip install coco-experiment || uv pip install coco-experiment
  $PY -u gen_bbob_data.py
fi

declare -A RC
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee runs/logs/${tag}.log
  RC[$tag]=$?
  echo "[exit ${RC[$tag]}] $tag"
}

for m in hard raw log1p; do
  run bbob54_train_${m} $PY -u train_soft_ablation.py --target-mode $m \
    --data-dir data_bbob54 --epochs 2 --calib-pairs 600 --max-tokens 16000 \
    --max-length 1600 --amp --save-model --tag bbob54 --seed 0 \
    --model-path answerdotai/ModernBERT-base
done

echo "=== ALL DONE $(date '+%F %T') ==="
for k in bbob54_train_hard bbob54_train_raw bbob54_train_log1p; do
  echo "RC[$k]=${RC[$k]}"
done
# 产物：runs/bbob54_{hard,raw,log1p}_tau1.0_e2_D5_seed0/{metrics.json, 同名.pt}
for m in hard raw log1p; do
  echo "--- $m ---"
  $PY -c "import json;d=json.load(open('runs/bbob54_${m}_tau1.0_e2_D5_seed0/metrics.json'));o=d['overall'];print({k:round(v,4) for k,v in o.items() if isinstance(v,float)})" 2>/dev/null
done
