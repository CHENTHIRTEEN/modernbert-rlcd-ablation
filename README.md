# ModernBERT × RLCD 软目标消融：解对关系判断的校准代理

用 Jev / [laya](https://github.com/NandhaKishorM/laya) 的 RLCD 思想（**R**einforcement **L**earning for **C**alibrated **D**ecisions，校准决策）微调 ModernBERT，判断昂贵优化中两个解的优劣关系（"A is better/worse than B"）。本仓库聚焦一个消融：**训练目标从 one-hot 硬标签换成 Bradley-Terry 式软目标后，精度与校准如何变化，以及 f 差要不要做 log1p 压缩**。

> 命名注意：RLCD 与 Apple 2023 年的 "Reinforcement Learning **from** Contrastive Distillation" (arXiv:2307.12950) 仅缩写相同，方法无关。

## 方法

对最小化问题，硬标签只表达方向、丢失关系强度（f(A)=1.0000 vs f(B)=1.0001 和 f(A)=1 vs f(B)=100 的硬标签同为 one-hot）。我们用 BT 链接函数把强度编码进目标分布：

```
y = σ( (f_B − f_A) / ((s_f + ε) · τ) )        # y = P(A better than B)，最小化
```

| target 模式 | 定义 | 说明 |
|---|---|---|
| `hard` | y ∈ {0,1} | 基线对照 |
| `raw` | s_f = 1.4826·MAD(f_train) | 锚点种群的稳健尺度，抗重尾 |
| `log1p` | f̃ = log1p(f − f_min)，s_f = 1.4826·MAD(f̃_train) | 对数压缩，跨函数同质化 |

关键性质（详见训练脚本 docstring）：反对称 y_ji = 1−y_ij 自动成立；Δ=0 → y=0.5（平局/模糊对自动落为均匀目标，即 RLCD 的 "unknowable → uniform"）；软 CE 最优点处 **logit gap = Δ/τ 精确成立**（概率承载校准、logit 差承载强度）；梯度 margin-weighted（近平局对贡献≈0）。

训练目标为对软目标的交叉熵（= log score，proper scoring rule 的特例）；训后按 laya 的流程在一批从未参与训练的 calib 记录上拟合温度 T（一维网格搜索），运行期 clamp [0.5, 5]。

## 数据

`gen_data.py` 用 [ioh](https://github.com/iohprofiler/IOHexperimenter) BBOB REAL 套件生成（已预生成在 `data/`，可直接用）：

- 函数：Sphere(1) / Ellipsoid(2) / Rastrigin(3) / Rosenbrock(8) × 维度 5, 10 × instance(rep) 1–3
- 每个 record：LHS 采样 30 锚点 + 30 候选，存原始 X 与真值 f
- **切分**：D5 的 rep1/rep2 训练（其中 Sphere/Ellipsoid 的 rep2 只做温度校准、绝不进训练），rep3 全部维度评测（D5 分布内 + D10 零样本迁移）

输入文本化与 [`test_bert.py`](test_bert.py)（R2SAEA 协议）逐字一致：train∪test 按维联合 min-max → 5 位小数 → 尖括号向量；`text_a = "A = <x_A> ; B = <x_B>"`，`text_b` = 12 条 "A is better/worse than \<x_j\>" 锚点证据句。f 值不进 prompt（模型须从 x 推断关系）。

## 快速开始（Linux/CUDA 服务器）

```bash
pip install -r requirements.txt
mkdir -p runs/logs
for m in hard raw log1p; do
  python -u train_soft_ablation.py --target-mode $m \
    --model-path answerdotai/ModernBERT-base \
    --amp --max-tokens 16000 --save-model \
    2>&1 | tee runs/logs/train_$m.log
done
```

- 设备自动选择 cuda > mps > cpu；CUDA 上建议 `--amp`（bf16 autocast）；显存 ≤16GB 时加 `--grad-ckpt`
- 单组 A100/4090 约 5–15 分钟；每组产出 `runs/<mode>_tau1.0/metrics.json`

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--target-mode` | 必填 | `hard` / `raw` / `log1p` |
| `--tau` | 1.0 | 目标锐度（与 s_f 相乘进入分母；改 s_f 语义优先） |
| `--train-dims` | 5 | 训练维度；评测始终覆盖数据中 rep3 的全部维度 |
| `--pairs-per-record` | 500 | 每 record 训练对数上限（D5 每 record 共 900 对） |
| `--epochs` / `--max-tokens` | 3 / 5000 | 轮数 / token 预算批（MPS 用默认，CUDA 可放大） |
| `--lr` | 2e-5 | AdamW，cosine + 5% warmup，clip 1.0 |
| `--stats-only` | 关 | 只打印目标分布诊断，不训练 |

## 输出：metrics.json

- `overall`：`acc`（硬方向准确率）、`nll_hard`/`brier_hard`/`ece`（max-prob 15-bin）、`nll_target`/`l1_target`（对软目标，proper 度量）、`spearman_conf_strength`（|p−0.5| 与 |Δ| 的秩相关，强度感知核心指标）、`T_fit`/`acc_T`…（温度校准后）
- `per_func`：按函数（× 维度）细分，Rosenbrock/Rastrigin 是区分 raw vs log1p 的关键
- `strata`：按 |f 差| 在 record 内四分位分层的 reliability 表（Q1 最平局、Q4 最悬殊）
- `diag`：训练前目标分布诊断（分位数、近平局/决断占比）

## 已知坑（Apple MPS）

ModernBERT + MPS 训练：bf16 autocast 会 NaN（用 fp32）；`use_reentrant=False` 梯度检查点会挂起；token 预算批 >~5000 会累积 OOM。**服务器 CUDA 无这些问题**，本地调试请用小批次 + 每 50 步 `torch.mps.empty_cache()`（脚本已内置）。

## 参考

- [laya](https://github.com/NandhaKishorM/laya)：Jev 的开源复现，本消融的 RLCD 训练配方（proper scoring rule + GRPO 式采样 + soft CE、训后温度校准）参考其 `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- TypeSafe Jev（System One 决策模型）与社区分析；R2SAEA（Lu et al. 2026）：关系代理的锚点式 prompt 协议来源

私有研究仓库，未做许可授权前请勿外传。
