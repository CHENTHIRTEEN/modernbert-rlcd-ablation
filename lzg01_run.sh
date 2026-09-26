#!/bin/bash
# LZG01 D5 单次 run × 6 模型（p2/p2e2 × hard/raw/log1p）
# 计分：--score mean --n-anchor-cap 15（db20970 定稿的日常配置，450 对/代，非 R2SAEA 复刻口径）
# 预计：4090 上 ~10-20 分钟/run，共 ~1-2 小时；结果写 ea_opt/exp_bert_data/{summary.csv,runs_flat.csv,runs.jsonl}
# 冒烟（可选，~2 分钟）：把循环体里 --runs 1 换成 --n-evals 40 先验证管线
set -o pipefail
cd /root/modernbert-rlcd-ablation/ea_opt || exit 1
export HF_HUB_OFFLINE=1
PY=../.venv/bin/python
mkdir -p ../runs/logs

$PY -c "import pymoo" 2>/dev/null || {
  echo "[setup] installing pymoo"
  $PY -m pip install pymoo 2>/dev/null || (cd .. && uv pip install pymoo)
}
$PY -c "import pymoo; print('[setup] pymoo', pymoo.__version__)" || exit 1

for tag in p2e2/hard_tau1.0 p2e2/raw_tau1.0 p2e2/log1p_tau1.0 p2/hard_tau1.0 p2/raw_tau1.0 p2/log1p_tau1.0; do
  ckpt=../runs/$tag/model.pt
  [ -f "$ckpt" ] || { echo "[skip] $ckpt missing"; continue; }
  echo "=== $tag $(date '+%F %T') ==="
  $PY -u run_exp_bert.py --ckpt "$ckpt" --probs LZG01 --dims 5 --runs 1 \
    --score mean --n-anchor-cap 15 2>&1 | tee -a ../runs/logs/lzg01_bert.log
done
echo "=== ALL DONE $(date '+%F %T') ==="
cat exp_bert_data/summary.csv 2>/dev/null
