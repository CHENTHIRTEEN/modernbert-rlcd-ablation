#!/bin/bash
# ga54_hard_decisive.sh —— D5 簇内塌缩问题：hard 标签决定性实验（2026-09-29 定稿）
# 背景：ga54 raw 在线 D5 被 random 全面超越（LZG01 差 13×、YLLF01 差 7.6×），D10/D20 反赢。
# 两个竞争假说（本地反证触发重审：raw 在 ga54 晚期快照与合成收缩簇上均不塌缩 ρ=0.3~0.8，
# 塌缩只在 raw 驱动的真实运行轨迹里出现）：
#   H1 标签侧：BT 软目标把簇内近平局对训成平局 → hard {0,1} 标签恢复簇内排序
#   H2 输入侧：archive 混龄锚点联合归一化把新簇压进窄带 / GA 数据无此分布 → hard 也救不回
# 裁决工具：--dump-dir 抓 raw 失败轨迹的真实每代场景 → scenario_replay.py 冻结回放 raw vs hard 同卷对比。
# 流程：
#   ① 训 ga54 hard（与 raw 同数据/种子/超参，只换 target-mode；~3.4h 4090，已有则跳过）
#   ② raw 驱动抓病灶场景：YLLF01 两个种子 + LZG01 一个种子（fp16 各 ~5-9 min，独立 out-dir 不碰网格 key）
#   ③ 冻结回放 raw vs hard（分钟级）。判据：hard 在 mid/late 段 spearman≥0.2 且显著高于 raw → H1；仍塌缩 → H2
#   ④ 在线验证：hard 4 问题 D5 ×10 runs（--lhs-seed 20260926 同 random/raw 配对）+ LZG02 D10/D20 不回归
#   ⑤ 顺手补 raw LZG01 D20 的断点欠账（主 out-dir resume 自动补 4 runs）
# 过拟合/校准哨兵（hard 的已知代价，看完 ① 后先看 metrics.json 再跑 ④）：
#   噪声函数 per-func acc 反常 >0.9 = 学到不可学方向；T_fit 爆掉/ECE 恶化 = 概率饱和毁校准（预期内，记录为代价）
set -o pipefail
cd /root/modernbert-rlcd-ablation || exit 1
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
mkdir -p runs/logs ea_opt/exp_bert_data_dumps
declare -A RC
run() {
  local tag=$1; shift
  echo "=== $tag $(date '+%F %T') ==="
  "$@" 2>&1 | tee -a runs/logs/ga54_hard_decisive_${tag}.log
  RC[$tag]=$?
  echo "[exit ${RC[$tag]}] $tag"
}

RAW=runs/ga54_raw_tau1.0_e2_D5_seed0/ga54_raw_tau1.0_e2_D5_seed0.pt
HARD=runs/ga54_hard_tau1.0_e2_D5_seed0/ga54_hard_tau1.0_e2_D5_seed0.pt

# ① 训 hard（与 ga54_run.sh 的 raw 逐参数一致，只换 mode）
if [ ! -f "$HARD" ]; then
  run train_hard $PY -u train_soft_ablation.py --target-mode hard --data-dir data_ga54 \
    --pairs-per-record 150 --epochs 2 --calib-pairs 600 --max-tokens 16000 \
    --amp --save-model --tag ga54 --seed 0 --model-path answerdotai/ModernBERT-base
fi
[ -f "$HARD" ] || { echo "[fatal] hard ckpt 缺失且训练失败"; exit 1; }

# ② raw 驱动抓病灶场景（seed 0/1 = 网格同款种子，轨迹与已完成 run 同分布）
cd ea_opt
for spec in "YLLF01 0" "YLLF01 1" "LZG01 0"; do
  set -- $spec
  run dump_${1}_s${2} $PY -u run_exp_bert.py --ckpt ../$RAW --score mean --n-anchor-cap 15 \
    --half --probs $1 --dims 5 --runs 1 --seed-base $2 --lhs-seed 20260926 \
    --out-dir ../exp_bert_dump_runs --dump-dir exp_bert_data_dumps
done

# ③ 冻结回放：raw vs hard 同卷
run replay $PY -u scenario_replay.py --dump exp_bert_data_dumps --half \
  --ckpt ../$RAW --ckpt ../$HARD

# ④ 在线验证（写主 exp_bert_data，tag 隔离、断点续跑）
run hard_d5 $PY -u run_exp_bert.py --ckpt ../$HARD --score mean --n-anchor-cap 15 --half \
  --probs LZG01,LZG02,YLLF01,YLLF02 --dims 5 --runs 10 --lhs-seed 20260926
run hard_hires $PY -u run_exp_bert.py --ckpt ../$HARD --score mean --n-anchor-cap 15 --half \
  --probs LZG02 --dims 10 20 --runs 10 --lhs-seed 20260926 --max-length 1600

# ⑤ raw 断点欠账：LZG01 D20 缺的 4 runs（resume 自动补）
run raw_debt $PY -u run_exp_bert.py --ckpt ../$RAW --score mean --n-anchor-cap 15 --half \
  --probs LZG01 --dims 20 --runs 10 --lhs-seed 20260926 --max-length 1600
cd ..

echo "=== ALL DONE $(date '+%F %T') ==="
for k in train_hard dump_YLLF01_s0 dump_YLLF01_s1 dump_LZG01_s0 replay hard_d5 hard_hires raw_debt; do
  echo "RC[$k]=${RC[$k]:-skipped}"
done
echo "--- hard 离线指标（哨兵） ---"
$PY -c "import json;d=json.load(open('runs/ga54_hard_tau1.0_e2_D5_seed0/metrics.json'));o=d['overall'];print({k:round(v,4) for k,v in o.items() if isinstance(v,float)});print('noise funcs acc:',{k:round(v['acc'],3) for k,v in d['per_func'].items() if k.startswith('IOH_n1') and v['acc']>0.9})" 2>/dev/null
echo "--- summary 尾部 ---"
tail -25 ea_opt/exp_bert_data/summary.csv 2>/dev/null
