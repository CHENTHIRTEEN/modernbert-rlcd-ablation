"""
ModernBERT 关系代理：顶替 R2SAEA 的 LLM_Relation_Fitness，插入同一套 LSEA 优化框架。

与 R2SAEA src/llm4surrogate/model.py::LLM_Relation_Fitness 的接口语义逐项对齐：
  fit(Xs, ys)             只缓存 anchor 集（tao 个历史最优解），零样本、每代不重训
  predict(Us) -> scores   返回每个候选解的投票得分，越大越好（LSEA 取 argmax 做真实评估）
  1) train∪test 按维联合 min-max 归一化（= LLM_Base.normalize，论文 'local' 消融方案）
  2) 每个锚点抽 n_evidence=12 条 "A is better/worse than <x_j>" 证据句：每锚点选一次、
     对该锚点的所有候选复用（与 train_soft_ablation.py::build_pairs 同构）
  3) 句对编码 text_a = "A = <...> ; B = <...>"，text_b = 证据块 -> ModernBERT 二分类头
  4) 投票打分逐行照抄 Fitness_Anchor_Voting_Scoring：
       w_i = 0.1 + 0.8 * (ymax - y_i) / (ymax - ymin)    （f 越小的锚点权重越大）
       score_j = sum_i( -1 * element[j,i] * w_i )        （element=+1 表示锚点 i 优于候选 j）

文本化协议与 test_bert.py / train_soft_ablation.py 逐字一致：beta=5 位小数、
逗号+空格分隔、尖括号包裹、最小化语义（label 1 = A better than B）。
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def fmt_vec(v: np.ndarray, beta: int) -> str:
    return ", ".join(f"{x:.{beta}f}" for x in v)


def joint_normalize(X_train: np.ndarray, X_test: np.ndarray):
    """对应 R2SAEA LLM_Base.normalize：train∪test 联合逐维 min-max"""
    comb = np.vstack([X_train, X_test])
    lo, hi = comb.min(axis=0), comb.max(axis=0)
    rng = hi - lo
    return (X_train - lo) / rng, (X_test - lo) / rng


def voting_weights(ys: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """Fitness_Anchor_Voting_Scoring.calculate_weights 的逐行照抄"""
    ys_max = np.max(ys)
    ys_min = np.min(ys)
    if ys_max == ys_min:
        return np.ones_like(ys) / len(ys)
    normalized = (ys_max - ys) / (ys_max - ys_min)
    return epsilon + (1 - 2 * epsilon) * normalized


class ModernBERTRelationSurrogate:
    def __init__(self, ckpt=None, base_model="answerdotai/ModernBERT-base",
                 device="auto", batch_size=256, n_evidence=12, beta=5,
                 half=False, max_length=1024, soft_vote=False):
        """
        ckpt: train_soft_ablation.py --save-model 产出的 runs/<mode>_tau1.0/model.pt
              （内存为 {"state_dict": ...}）。为 None 时直接用原生底座（= cls 基线）。
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.n_evidence = n_evidence
        self.beta = beta
        self.max_length = max_length
        self.soft_vote = soft_vote

        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=2)
        if ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
            model.load_state_dict(sd)
        model.to(self.device).eval()
        if half and self.device.type == "cuda":
            model.half()
        self.model = model

        self.Train_Xs = None
        self.Train_ys = None
        self._calls = 0
        # 最近一次 predict 的诊断量（供上层记录投票是否退化）
        self.last_stats = {}

    # ---- 与 LLM_Base.fit 相同：只存数据，不训练（零样本代理） ----
    def fit(self, Xs, ys):
        Xs = np.asarray(Xs)
        ys = np.asarray(ys)
        if Xs.ndim <= 1:
            Xs = Xs.reshape(-1, 1)
        self.Train_Xs = Xs
        self.Train_ys = ys.flatten()

    @torch.inference_mode()
    def _pair_probs(self, texts_a, texts_b):
        """返回 p(A better than B)，shape (N,)"""
        ps = []
        for s in range(0, len(texts_a), self.batch_size):
            enc = self.tok(texts_a[s:s + self.batch_size],
                           texts_b[s:s + self.batch_size],
                           padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1]
            ps.append(p.cpu().numpy())
        return np.concatenate(ps) if ps else np.zeros(0)

    def predict(self, Us, seed=0):
        """
        Us: (n_cand, D) 候选解（未经归一化的原空间）
        返回 (n_cand,) 投票得分，越大越好。
        """
        Xs, ys = self.Train_Xs, self.Train_ys
        Ns, Nu = joint_normalize(Xs, Us)          # Ns: (n_anchor, D), Nu: (n_cand, D)
        n_anchor, n_cand = len(Ns), len(Nu)

        # 证据句：每锚点选一次，对该锚点的所有候选复用（与训练侧 build_pairs 同构）
        ev_rng = np.random.default_rng([int(seed), self._calls])
        self._calls += 1
        texts_a, texts_b = [], []
        for i in range(n_anchor):
            others = np.delete(np.arange(n_anchor), i)
            chosen = ev_rng.choice(others, min(self.n_evidence, len(others)),
                                   replace=False)
            lines = []
            for j in chosen:
                word = "better" if ys[i] < ys[j] else "worse"
                lines.append(f"A is {word} than <{fmt_vec(Ns[j], self.beta)}>")
            ev = "\n".join(lines)
            a_str = fmt_vec(Ns[i], self.beta)
            for k in range(n_cand):
                texts_a.append(f"A = <{a_str}> ; B = <{fmt_vec(Nu[k], self.beta)}>")
                texts_b.append(ev)

        p = self._pair_probs(texts_a, texts_b).reshape(n_anchor, n_cand)
        # element[j, i] = +1 表示锚点 i 优于候选 j（与 R2SAEA 的 element 矩阵同约定）
        element = np.where(p.T > 0.5, 1, -1)

        w = voting_weights(ys)
        if self.soft_vote:
            # 硬投票的线性扩展：p=1 -> -w，p=0 -> +w
            scores = (1.0 - 2.0 * p.T) @ w
        else:
            scores = (-1.0 * element * w[None, :]).sum(axis=1)

        self.last_stats = {
            "n_pairs": int(n_anchor * n_cand),
            "mean_p_better": float(p.mean()),
            "frac_better_votes": float((p > 0.5).mean()),
        }
        return scores


class RandomVotingSurrogate:
    """随机投票基线：同 EA、同投票打分，票面随机 —— 代理无效时的下限对照。"""

    def __init__(self):
        self.Train_Xs = None
        self.Train_ys = None
        self._calls = 0
        self.last_stats = {}

    def fit(self, Xs, ys):
        self.Train_Xs = np.asarray(Xs)
        self.Train_ys = np.asarray(ys).flatten()

    def predict(self, Us, seed=0):
        rng = np.random.default_rng([int(seed), self._calls])
        self._calls += 1
        n_anchor, n_cand = len(self.Train_Xs), len(Us)
        element = rng.choice([1, -1], size=(n_cand, n_anchor))
        w = voting_weights(self.Train_ys)
        self.last_stats = {"n_pairs": int(n_anchor * n_cand),
                           "mean_p_better": 0.5, "frac_better_votes": 0.5}
        return (-1.0 * element * w[None, :]).sum(axis=1)
