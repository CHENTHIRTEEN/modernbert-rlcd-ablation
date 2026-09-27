#!/bin/bash
# ga54 重训：GA 轨迹数据集（54函数×3instance×5seed×10快照=8100 record，D5，pop100/100代/每10代取样）
# 3 target-mode × epochs=2；主线 mode=raw（2026-09-27 终裁），hard/log1p 为论文消融行
# 规模：6420 训练 record × 150 对 = 96 万对/epoch（bbob54 的 ~18 倍），4090 上每 mode 约 3.5-4h，
#       三 mode 合计 ~11h + 每 mode 评测 ~20min（test=seed3 全部 1620 record×500 对）
# 冒烟（~5 分钟，写独立 runs/smoke_*，不碰正式产物）：
#   $PY -u train_soft_ablation.py --target-mode raw --data-dir data_ga54 \
#     --max-steps 20 --eval-max-records 2 --pairs-per-record 120 --calib-pairs 60 --test-pairs 50 --tag smoke
set -o pipefail
cd /root/modernbert-rlcd-ablation
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p runs/logs

# 数据兜底：仓库里 data_ga54/ 缺失时现场生成（需 pymoo + coco-experiment，约 10-20 分钟）
if [ ! -d data_ga54 ]; then
  $PY -c "import cocoex, pymoo" 2>/dev/null || $PY -m pip install coco-experiment pymoo || uv pip install coco-experiment pymoo
  $PY -u gen_ga_data.py
fi

declare -A RC
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee runs/logs/${tag}.log
  RC[$tag]=$?
  echo "[exit ${RC[$tag]}] $tag"
}

for m in raw hard log1p; do
  run ga54_train_${m} $PY -u train_soft_ablation.py --target-mode $m \
    --data-dir data_ga54 --pairs-per-record 150 --epochs 2 --calib-pairs 600 \
    --max-tokens 16000 --amp --save-model --tag ga54 --seed 0 \
    --model-path answerdotai/ModernBERT-base
done

echo "=== ALL DONE $(date '+%F %T') ==="
for k in ga54_train_raw ga54_train_hard ga54_train_log1p; do
  echo "RC[$k]=${RC[$k]}"
done
# 产物：runs/ga54_{raw,hard,log1p}_tau1.0_e2_D5_seed0/{metrics.json, 同名.pt}
for m in raw hard log1p; do
  echo "--- $m ---"
  $PY -c "import json;d=json.load(open('runs/ga54_${m}_tau1.0_e2_D5_seed0/metrics.json'));o=d['overall'];print({k:round(v,4) for k,v in o.items() if isinstance(v,float)})" 2>/dev/null
done
