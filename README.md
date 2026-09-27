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

### ga54（GA 轨迹数据集，当前主线）

`gen_ga_data.py` 按 R2SAEA 论文 III-B2 的语料协议生成**真实优化轨迹**（pymoo 默认 GA + cocoex 真值；已预生成在 `data_ga54/`，gzip jsonl，可直接用）：

- 轨迹：54 函数（bbob + bbob-noisy）× instance 1–3 × **seed 1–5** = 810 条 GA 轨迹；每条 pop_size=100、100 代、pymoo 默认算子（SBX 0.9/η15 + PM η20 + 锦标赛）
- 快照：**每 10 代取当代种群**（gens 10..100，每轨迹 10 个快照）→ 每 (函数,instance,seed,gen) 一条 record，共 **8100 条**
- record 内 100 个体随机置换后 50 作锚点/证据池、50 作候选；`fes = 100×(gen+1)`（gen10 → 1100，与 R2SAEA test_data 口径一致）
- **切分**：rep=seed；seed {1,2,4,5} 训练、**seed 3 为 held-out 测试**（同函数同 instance 的未见轨迹）、Sphere/Ellipsoid 的 seed2 只做温度校准
- 输出 `.jsonl.gz`（控 git 体积；`load_records` 已兼容）；GA 由 pymoo seed 决定，noisy 噪声随求值流推进（同版本重放逐位一致）
- 注意：易函数 ~gen50 后种群高度收敛（近平局对为主），软目标的 margin 加权与 y→0.5 机制正为此设计；训练侧以 `--pairs-per-record 150` 控制单 epoch 规模（96 万对）

### bbob54（LHS 数据集，2026-09-27 前主线）

`gen_bbob_data.py` 用 [coco-experiment](https://github.com/numbbo/coco)（COCO 官方 Python 接口，import 名 `cocoex`）生成 **BBOB + BBOB-noisy 全套件**（已预生成在 `data_bbob54/`，12MB，可直接用；ioh 0.3.x 不含 noisy 套件故改用 cocoex）：

- 函数：BBOB f1–f24（无噪）+ BBOB-noisy f101–f130（gauss/uniform/cauchy）共 54 个 × instance(rep) 1–3 × 维度 5, 10, 20 = 486 条 record
- 每个 record：LHS 采样 100 点（[-5,5]^D，随机置换后前 50 作锚点/证据池、后 50 作候选，置换避免 LHS 分层偏向半值域）
- **切分约定与旧数据完全一致**：rep=instance；D5 的 rep1/rep2 训练（Sphere/Ellipsoid 的 rep2 只做温度校准），rep3 全部维度评测（D5 分布内 + D10/D20 零样本迁移）
- `prob_name`：无噪 = `IOH_{ioh名}`（与旧数据对齐，`IOH_Sphere`/`IOH_Ellipsoid` 仍作 calib），noisy = `IOH_n{fid}_{gauss|unif|cauchy}`；另附 `suite/fid/instance/noise/coco_id` 溯源字段
- **noisy 语义**：噪声随求值流推进（每个 y 是该求值位置的一次抽样，cocoex 禁止重复求值）；固定种子 + 固定求值顺序 ⇒ 同种子重放逐位一致（已验证）

训练对规模：D5 rep1/2 共 106 record × 500 对 = **5.3 万对/epoch**（旧 4 函数数据集约 3000 对）。

### 旧数据集（ioh 4 函数，保留作对照）

`gen_data.py` 用 [ioh](https://github.com/iohprofiler/IOHexperimenter) BBOB REAL 套件生成（已预生成在 `data/`）：

- 函数：Sphere(1) / Ellipsoid(2) / Rastrigin(3) / Rosenbrock(8) × 维度 5, 10 × instance(rep) 1–3
- 每个 record：LHS 采样 30 锚点 + 30 候选，存原始 X 与真值 f
- **切分**：D5 的 rep1/rep2 训练（其中 Sphere/Ellipsoid 的 rep2 只做温度校准、绝不进训练），rep3 全部维度评测（D5 分布内 + D10 零样本迁移）

输入文本化与 [`test_bert.py`](test_bert.py)（R2SAEA 协议）逐字一致：train∪test 按维联合 min-max → 5 位小数 → 尖括号向量；`text_a = "A = <x_A> ; B = <x_B>"`，`text_b` = 12 条 "A is better/worse than \<x_j\>" 锚点证据句。f 值不进 prompt（模型须从 x 推断关系）。

## 快速开始（Linux/CUDA 服务器）

```bash
# ga54 GA 轨迹数据集（推荐，数据已随仓库提交，无需重新生成）：
bash ga54_run.sh 2>&1 | tee runs/logs/ga54_all.log
# 或 bbob54 LHS 数据集：
bash bbob54_run.sh 2>&1 | tee runs/logs/bbob54_all.log
# 等价手写（ga54）：
for m in raw hard log1p; do
  python -u train_soft_ablation.py --target-mode $m --data-dir data_ga54 \
    --pairs-per-record 150 --epochs 2 --calib-pairs 600 --max-tokens 16000 \
    --model-path answerdotai/ModernBERT-base --amp --save-model --tag ga54 \
    2>&1 | tee runs/logs/ga54_train_$m.log
done
```

- 设备自动选择 cuda > mps > cpu；CUDA 上建议 `--amp`（bf16 autocast）；显存 ≤16GB 时加 `--grad-ckpt`
- `--max-length 1600` 为 D20 评测所需（D5 训练对实际远短于此）
- 每组产出 `runs/<tag>_<mode>_tau<tau>_e<epochs>_D<dims>_seed<seed>/`：`metrics.json` + 同名 `.pt` 权重；`--tag`/seed/epochs 变更都会产生新目录，不再互相覆盖

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--target-mode` | 必填 | `hard` / `raw` / `log1p` |
| `--tau` | 1.0 | 目标锐度（与 s_f 相乘进入分母；改 s_f 语义优先） |
| `--data-dir` | `data/` | 数据目录；bbob54 数据集传 `data_bbob54` |
| `--tag` | `bbob54` | 运行标识，进输出目录/权重文件名 |
| `--train-dims` | 5 | 训练维度；评测始终覆盖数据中 rep3 的全部维度 |
| `--pairs-per-record` / `--test-pairs` | 500 / 500 | 每 record 训练/评测对数上限（0=全量；固定种子，跨 mode 可比） |
| `--epochs` / `--max-tokens` | 2 / 5000 | 轮数（p2e2 消融：2ep 全面优于 3ep）/ token 预算批（MPS 用默认，CUDA 可放大） |
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
