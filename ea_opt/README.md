# ea_opt：R2SAEA 同框架在线优化实验

把微调 ModernBERT 关系代理插进 R2SAEA 的 LSEA 优化主循环（SOP），与论文 Table I 同参数对比优化结果。R2SAEA 论文/代码见其原仓库（`../R2SAEA`）。

## 与 R2SAEA 的关系

| 部分 | 来源 | 说明 |
|---|---|---|
| EA 主循环 | 照抄 `algorithm/lsea.py::UEDA/LSEA` | pop_size=30, n_evals=300, tao=50, LHS, VWH(M=15), Pb=0.2, Pc=0.2；每代只真实评估代理分最高的 1 解 |
| 繁殖算子 | 照抄 `algorithm/util/reproduction.py` + `edamodel.py` | vendor 在本目录（`edamodel.py` 只保留 VWH+local_search，去 matplotlib） |
| 基准函数 | 照抄 `problem/{LZG,YLL}.py` | vendor 在本目录；默认 15 问题 = 论文 Table I（YLLF10/11 与 LZG 重复排除） |
| 代理 | `bert_surrogate.py`（我们的 ModernBERT） | 接口语义对齐 `LLM_Relation_Fitness`：联合 min-max、每锚点 12 条证据句（与 `train_soft_ablation.py::build_pairs` 同构）、句对编码、投票打分逐行照抄 |

对原仓库的必要修复（他们的 LSEA-SOP 在 numpy 2.x 下跑不通）：

1. `lsea.py:254` float 切片 TypeError → 取得分最高的一半喂 EDA；
2. `local_search` 收 (NL,1) 列向量在 numpy 2.x 下 ValueError → flatten 后传入。

## 运行

依赖：`pip install pymoo`（torch/transformers 复用主环境；不需要 langchain）。

```bash
cd ea_opt
# 冒烟：单函数单次（ckpt 缺省 = 原生底座，cls 头随机，预期≈无代理基线）
python run_exp_bert.py --probs LZG01 --dims 5 --runs 1
# "我们的方法"轻量配置：mean 计分 + 15 锚点（450 对/代），不看 Table I 对表时的日常跑法
nohup python -u run_exp_bert.py --ckpt ../runs/p2/hard_tau1.0/model.pt \
  --score mean --n-anchor-cap 15 --runs 10 > exp_mean.log 2>&1 &
# R2SAEA 复刻配置：默认即 --score hard --n-anchor-cap 0（tao=50 全锚点 ε 权重投票），对表用
# 提速：--half（CUDA fp16）；下限对照：--surrogate random（同 EA 同种子，随机投票）
```

计分方式 `--score`：GA 只消费"分高者好"接口，聚合是代理内部实现——
`hard`（默认）= R2SAEA 硬票+按 f 的 ε 权重投票（复刻用）；`mean` = 直接平均
P(候选优于锚点)，无启发式权重（我们的计分）；`soft` = 同权重下 p 的线性扩展。

ckpt 为 `train_soft_ablation.py --save-model` 产出的 `runs/<mode>_tau1.0/model.pt`（`{"state_dict": ...}`）。

## 输出（`exp_bert_data/`，默认已 gitignore）

- `runs.jsonl`：逐运行全记录（含收敛曲线 `curve_fes/curve_best`、投票诊断 `mean_vote_better`，健康模型应 ≈0.5）
- `summary.csv`：论文 result.csv 同格式（算法,问题,维度,均值,标准差,运行数），直接对 Table I
- `runs_flat.csv`：逐运行扁平表
