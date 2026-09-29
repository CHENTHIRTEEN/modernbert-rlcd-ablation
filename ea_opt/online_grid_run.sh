#!/bin/bash
# online_grid_run.sh — 大规模在线网格：Table I 全 15 问题（LZG01-04 已接入 + YLL 全量接入）
#   problem/YLL.py vendor 自 R2SAEA（与源码逐字一致），runner 的 build_problem/parse_probs 原生支持
# 主线 checkpoint：ga54 raw（本地离线 acc=0.776@81 万测试对，2026-09-28 回收）
# 网格：15 问题 × D{5,10,20} × 10 runs（seed0-9 模型间配对）= 450 runs/checkpoint
# 计分：--score mean --n-anchor-cap 15（db20970 定稿日常口径，450 对/代；严格对 Table I 复刻另跑 --score hard）
# 协议：--lhs-seed 20260926 固定初始 LHS 种群——同 (问题,维度) 的 10 runs（及所有模型）共享同一初始解，
#       只有繁殖算子+代理选择的随机性随 run seed(0-9) 变化，消除初始种群方差（2026-09-28 用户定稿）
# 维度实测（beta=5、12 证据句全对 token）：D5=440 / D10=795（1024 均够）；D20=1506 → 必须 --max-length 1600
# 顺序：random 下限基线（CPU 分钟级）→ ga54 D5 → D10 → D20；断点续跑自动（runs.jsonl 按 key 去重）
# 预计 4090 + --half：~1-2s/代 × 270 代 ≈ 5-9 min/run，450 runs ≈ 1.5-3 天；fp32 翻倍
# 冒烟（~5 分钟，写独立 exp_bert_data_smoke/，不污染正式 key）：SMOKE=1 bash online_grid_run.sh
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

CKPT=../runs/ga54_raw_tau1.0_e2_D5_seed0/ga54_raw_tau1.0_e2_D5_seed0.pt
[ -f "$CKPT" ] || { echo "[fatal] ckpt missing: $CKPT（服务器上由 ga54_run.sh --save-model 产出）"; exit 1; }

PROBS="LZG01-04,YLLF01-09,YLLF12,YLLF13"   # Table I 的 15 问题（YLLF10/11 与 LZG 重复排除）
SCORE_ARGS="--score mean --n-anchor-cap 15 --lhs-seed 20260926"
declare -A RC

run() {
  local tag=$1; shift
  echo "=== $tag $(date '+%F %T') ==="
  "$@" 2>&1 | tee -a ../runs/logs/online_grid_${tag}.log
  RC[$tag]=$?
  echo "[exit ${RC[$tag]}] $tag"
}

if [ "${SMOKE:-0}" = "1" ]; then
  run smoke_random $PY -u run_exp_bert.py --surrogate random --probs YLLF01,YLLF12 --dims 5 --runs 1 --n-evals 60 \
    --lhs-seed 20260926 --out-dir exp_bert_data_smoke
  run smoke_d5  $PY -u run_exp_bert.py --ckpt "$CKPT" $SCORE_ARGS --half --probs YLLF01 --dims 5 --runs 1 --n-evals 36 \
    --out-dir exp_bert_data_smoke
  run smoke_d20 $PY -u run_exp_bert.py --ckpt "$CKPT" $SCORE_ARGS --half --probs YLLF01 --dims 20 --runs 1 --n-evals 36 \
    --max-length 1600 --out-dir exp_bert_data_smoke
  echo "=== SMOKE DONE $(date '+%F %T') ==="; exit 0
fi

# 1) random 下限基线：同 EA 同种子同初始种群随机投票（CPU 秒级/run，先跑完立即可看）
run random $PY -u run_exp_bert.py --surrogate random --probs "$PROBS" --dims 5 10 20 --runs 10 --lhs-seed 20260926

# 2) ga54 raw 主线：D5 → D10 → D20（D20 必须 --max-length 1600：全对 1506 token，默认 1024 会截掉 ~1/3 证据段）
run ga54_d5  $PY -u run_exp_bert.py --ckpt "$CKPT" $SCORE_ARGS --half --probs "$PROBS" --dims 5  --runs 10
run ga54_d10 $PY -u run_exp_bert.py --ckpt "$CKPT" $SCORE_ARGS --half --probs "$PROBS" --dims 10 --runs 10
run ga54_d20 $PY -u run_exp_bert.py --ckpt "$CKPT" $SCORE_ARGS --half --probs "$PROBS" --dims 20 --runs 10 --max-length 1600

echo "=== ALL DONE $(date '+%F %T') ==="
for k in random ga54_d5 ga54_d10 ga54_d20; do echo "RC[$k]=${RC[$k]}"; done
echo "--- summary ---"
cat exp_bert_data/summary.csv 2>/dev/null
