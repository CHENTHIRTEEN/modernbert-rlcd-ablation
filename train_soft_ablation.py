#!/usr/bin/env python3
"""
RLCD 式软目标消融：hard / raw / log1p 三种 target 构造微调 ModernBERT-base（cls 头，ctx 格式）。

软目标（Bradley-Terry 链接，最小化问题）：
    y = sigmoid((f_B - f_A) / ((s_f + eps) * tau))        # y = P(A better than B)
    A better <=> f_A < f_B <=> f_B - f_A > 0 <=> y > 0.5
    raw   : s_f = 1.4826 * MAD(y_train)                   # 逐 record，锚点种群的稳健尺度
    log1p : f~ = log1p(f - f_min(train∪test)), s_f = 1.4826 * MAD(f~_train)
    hard  : y ∈ {0,1}（基线对照）

输入格式与 test_bert.py ctx 模式逐字一致（直接 import 其构造函数，杜绝格式漂移）：
    text_a = "A = <x_A> ; B = <x_B>"（决定性信息在序列头部，不会被截断）
    text_b = n_evidence 条 "A is better/worse than <x_j>" 证据句

数据切分（按 record）：rep1/rep2 -> train + calib，rep3 -> test。
calib 固定 4 条（每函数 1 条），只用于温度拟合，不进训练。

用法：
    python train_soft_ablation.py --target-mode raw   --stats-only
    python train_soft_ablation.py --target-mode raw   --max-steps 20 --eval-max-records 1 --pairs-per-record 120 --calib-pairs 50   # smoke
    python train_soft_ablation.py --target-mode raw                                  # 完整训练+评测
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from test_bert import evidence_block, fmt_vec, joint_normalize, load_records  # noqa: E402

FUNCS = ["Sphere", "Ellipsoid", "Rastrigin", "Rosenbrock"]
# calib 函数：取前两个函数的 rep2 记录（dim 跟随 --train-dims），温度拟合用，绝不进训练
CALIB_FUNCS = {"IOH_Sphere", "IOH_Ellipsoid"}


# ---------------- 软目标构造 ----------------

def mad_scale(v: np.ndarray) -> float:
    med = np.median(v)
    return 1.4826 * float(np.median(np.abs(v - med)))


def record_scale(rec, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 (f~_train, f~_test, s_f)：mode 决定 f 是否做 log1p 压缩以及尺度来源。"""
    ys, uys = rec["y_train"], rec["y_test"]
    if mode == "log1p":
        c = float(min(ys.min(), uys.min()))
        ft, fu = np.log1p(ys - c), np.log1p(uys - c)
        s = mad_scale(ft)
    elif mode in ("raw", "hard"):             # hard: 目标取 0/1，delta 仍用 raw 归一化做强度度量
        ft, fu = ys.astype(float), uys.astype(float)
        s = mad_scale(ft)                     # 锚点种群的稳健尺度（抗 Rosenbrock 重尾）
    else:
        raise ValueError(mode)
    return ft, fu, max(s, 1e-12)


def soft_target(f_a: float, f_b: float, s: float, eps_rel: float, tau: float) -> float:
    z = (f_b - f_a) / ((s * (1.0 + eps_rel)) * tau)
    return float(1.0 / (1.0 + math.exp(-z)))     # y = P(A better than B)


# ---------------- 数据构建 ----------------

def build_pairs(records, args, mode, subset: str):
    """subset: 'train' | 'calib' | 'test'。证据句/子采样对 mode 不敏感，跨 config 可比。"""
    items = []
    for rec in records:
        run_id = rec["run_id"]
        rep = run_id % 10
        key = (rec["prob_name"], rec["prob_D"])
        if subset == "test":
            if rep != 3:
                continue                      # test 吃 rep3 的所有维度（含未训练维度，测迁移）
        elif subset == "calib":
            if rep != 2 or rec["prob_name"] not in CALIB_FUNCS or rec["prob_D"] not in args.train_dims:
                continue
        else:  # train
            if rep == 3 or rec["prob_D"] not in args.train_dims:
                continue
            if rep == 2 and rec["prob_name"] in CALIB_FUNCS:
                continue

        Xs, ys = rec["X_train"], rec["y_train"]
        Us, uys = rec["X_test"], rec["y_test"]
        Ns, Nu = joint_normalize(Xs, Us)
        ft, fu, s = record_scale(rec, mode)

        rng_ev = np.random.default_rng(run_id)          # 证据句：逐 record 确定性
        metas = []
        for i in range(len(Ns)):
            others = np.delete(np.arange(len(Ns)), i)
            chosen = rng_ev.choice(others, min(args.n_evidence, len(others)), replace=False)
            ev = evidence_block(Ns, ys, i, chosen, args.beta)
            for k in range(len(Nu)):
                if mode == "hard":
                    y = float(ys[i] < uys[k])
                else:
                    y = soft_target(float(ft[i]), float(fu[k]), s, args.eps_rel, args.tau)
                metas.append({
                    "ids": None,
                    "text_a": f"A = <{fmt_vec(Ns[i], args.beta)}> ; B = <{fmt_vec(Nu[k], args.beta)}>",
                    "text_b": ev,
                    "y": y,
                    "label": int(ys[i] < uys[k]),
                    "delta": (float(fu[k]) - float(ft[i])) / ((s * (1.0 + args.eps_rel)) * args.tau),
                    "strength": abs(float(uys[k]) - float(ys[i])),   # 原始 f 差，跨 config 不变
                    "rec": f"{rec['prob_name']}_D{rec['prob_D']}_r{rep}",
                    "func": rec["prob_name"], "dim": rec["prob_D"],
                })

        # 子采样（train/calib），全局固定种子 -> 三种 mode 看到同一批对
        if subset in ("train", "calib"):
            cap = args.pairs_per_record if subset == "train" else args.calib_pairs
            if cap and len(metas) > cap:
                idx = np.random.default_rng(args.seed + 7).choice(len(metas), cap, replace=False)
                metas = [metas[j] for j in sorted(idx)]

        enc = _tok([m["text_a"] for m in metas], [m["text_b"] for m in metas], args.max_length)
        for m, ids in zip(metas, enc):
            m["ids"] = ids
        items.extend(metas)
        print(f"  [{subset}] {rec['prob_name']}_D{rec['prob_D']}_r{rep}: {len(metas)} pairs")
    return items


_TOK = None


def _tok(texts_a, texts_b, max_length):
    enc = _TOK(texts_a, texts_b, truncation=True, max_length=max_length)
    return [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]


# ---------------- 批处理 ----------------

def make_batches(items, max_tokens: int, rng: np.random.Generator):
    idx = np.arange(len(items)); rng.shuffle(idx)
    idx = sorted(idx.tolist(), key=lambda i: len(items[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in idx:
        L = len(items[i]["ids"])
        if cur and (len(cur) + 1) * max(cur_max, L) > max_tokens:
            batches.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(items, idxs, pad_id, device):
    sel = [items[i] for i in idxs]
    n = len(sel); L = max(len(s["ids"]) for s in sel)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    for r, s in enumerate(sel):
        k = len(s["ids"])
        ids[r, :k] = torch.from_numpy(s["ids"].astype(np.int64))
        att[r, :k] = 1
    y = torch.tensor([s["y"] for s in sel], dtype=torch.float32)
    return ids.to(device), att.to(device), y.to(device)


# ---------------- 指标 ----------------

def ece(conf: np.ndarray, corr: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
    return float(e)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ent(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 1e-9, 1 - 1e-9)
    return -(y * np.log(y) + (1 - y) * np.log(1 - y))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """一维网格 + 局部细化搜 T（LBFGS 在 sigmoid 饱和区会数值爆炸）。"""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    yy = np.asarray(y, float)

    def nll(T: float) -> float:
        x = z / T
        q = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700))),
                     np.exp(np.clip(x, -700, 700)) / (1.0 + np.exp(np.clip(x, -700, 700))))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean())

    grid = np.exp(np.linspace(math.log(0.1), math.log(10.0), 121))
    vals = [nll(T) for T in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    cand = np.linspace(a, b, 33)
    for _ in range(3):
        vals = [nll(T) for T in cand]
        j = int(np.argmin(vals))
        a, b = cand[max(0, j - 1)], cand[min(len(cand) - 1, j + 1)]
        cand = np.linspace(a, b, 33)
    return float(cand[int(np.argmin([nll(T) for T in cand]))])


def evaluate(p: np.ndarray, items, T: float = 1.0) -> dict:
    pT = torch.sigmoid(torch.logit(torch.tensor(p, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)) / T).numpy()
    label = np.array([it["label"] for it in items], float)
    y = np.array([it["y"] for it in items], float)
    delta = np.array([abs(it["delta"]) for it in items], float)
    conf, corr = np.maximum(p, 1 - p), (p >= 0.5).astype(float) == label
    return {
        "n": int(len(p)),
        "acc": float(((p >= 0.5).astype(float) == label).mean()),
        "nll_hard": bce(p, label),
        "brier_hard": float(((p - label) ** 2).mean()),
        "ece": ece(conf, corr.astype(float)),
        "l1_target": float(np.abs(p - y).mean()),
        "nll_target": bce(p, y),
        "target_entropy": float(ent(y).mean()),
        "spearman_conf_strength": spearman(np.abs(p - 0.5), delta),
        "acc_T": float(((pT >= 0.5).astype(float) == label).mean()),
        "nll_hard_T": bce(pT, label),
        "ece_T": ece(np.maximum(pT, 1 - pT), ((pT >= 0.5).astype(float) == label).astype(float)),
        "l1_target_T": float(np.abs(pT - y).mean()),
    }


def strata(p: np.ndarray, items, n_bins: int = 4) -> list:
    strength = np.array([it["strength"] for it in items], float)
    rec_keys = [it["rec"] for it in items]
    pct = np.zeros(len(items))
    for key in set(rec_keys):
        m = np.array([k == key for k in rec_keys])
        order = strength[m].argsort().argsort()
        pct[m] = order / max(1, len(order) - 1)
    y = np.array([it["y"] for it in items], float)
    label = np.array([it["label"] for it in items], float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        sel = (pct >= edges[b]) & (pct <= edges[b + 1] if b == n_bins - 1 else pct < edges[b + 1])
        rows.append({
            "bucket": f"Q{b + 1}", "n": int(sel.sum()),
            "acc": float(((p[sel] >= 0.5).astype(float) == label[sel]).mean()),
            "mean_y": float(y[sel].mean()),
            "mean_conf": float(np.maximum(p[sel], 1 - p[sel]).mean()),
            "mean_abs_p05": float(np.abs(p[sel] - 0.5).mean()),
        })
    return rows


# ---------------- 主流程 ----------------

def probe_amp(model, device) -> bool:
    try:
        ids = torch.randint(0, 50000, (2, 16), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        return True
    except Exception:
        return False


@torch.inference_mode()
def predict(model, items, pad_id, device, max_tokens, use_amp) -> np.ndarray:
    model.eval()
    if device.type == "mps":
        torch.mps.empty_cache()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = np.zeros(len(items))
    cur, cur_max = [], 0
    chunks = []
    for i in order:
        L = len(items[i]["ids"])
        if cur and (len(cur) + 1) * max(cur_max, L) > max_tokens:
            chunks.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur:
        chunks.append(cur)
    for idxs in chunks:
        ids, att, _ = collate(items, idxs, pad_id, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=att).logits
        p = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        for j, i in enumerate(idxs):
            out[i] = p[j]
    model.train()
    return out


def main():
    global _TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mode", required=True, choices=["hard", "raw", "log1p"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--eps-rel", type=float, default=0.01)
    ap.add_argument("--data-dir", default=Path(__file__).parent / "data")
    ap.add_argument("--model-path", default="/Users/bingchen/Desktop/20260925/ModernBert/ModernBERT-base")
    ap.add_argument("--out-root", default=Path(__file__).parent / "runs")
    ap.add_argument("--beta", type=int, default=5)
    ap.add_argument("--n-evidence", type=int, default=12)
    ap.add_argument("--train-dims", type=int, nargs="+", default=[5],
                    help="训练用维度；test 始终评测数据里 rep3 的全部维度")
    ap.add_argument("--pairs-per-record", type=int, default=500)
    ap.add_argument("--calib-pairs", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast（MPS + ModernBERT 已知会 NaN，默认关）")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="激活检查点（MPS 上 use_reentrant=False 有挂起前科，默认关）")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时只训这么多步（smoke）")
    ap.add_argument("--eval-max-records", type=int, default=0, help=">0 时评测只取前 N 条 test record（smoke）")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，从 runs/<mode>_tau1.0/model.pt 加载权重直接评测（要求此前用 --save-model 训过）")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    records = load_records(Path(args.data_dir))
    pool = [r for r in records if r["prob_D"] in args.train_dims and r["run_id"] % 10 != 3]
    calib_recs = [r for r in pool if r["run_id"] % 10 == 2 and r["prob_name"] in CALIB_FUNCS]
    calib_ids = {r["run_id"] for r in calib_recs}
    train_recs = [r for r in pool if r["run_id"] not in calib_ids]
    test_recs = [r for r in records if r["run_id"] % 10 == 3]
    print(f"device={device} | train={len(train_recs)} calib={len(calib_recs)} test={len(test_recs)} records "
          f"(train dims={args.train_dims}, test dims={sorted({r['prob_D'] for r in test_recs})})")

    _TOK = AutoTokenizer.from_pretrained(args.model_path)
    if _TOK.pad_token_id is None:
        _TOK.pad_token = _TOK.eos_token

    print(f"building pairs (mode={args.target_mode}) ...")
    if args.eval_only:
        train_items = []
        calib_items = build_pairs(calib_recs, args, args.target_mode, "calib")
        test_items = build_pairs(test_recs, args, args.target_mode, "test")
    else:
        train_items = build_pairs(train_recs, args, args.target_mode, "train")
        calib_items = build_pairs(calib_recs, args, args.target_mode, "calib")
        test_items = build_pairs(test_recs, args, args.target_mode, "test")

    # ---- 目标分布诊断 ----
    y_all = np.array([it["y"] for it in train_items + test_items])
    diag = {
        "y_quantiles": {q: float(np.quantile(y_all, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "frac_near_tie_[.45,.55]": float(((y_all >= 0.45) & (y_all <= 0.55)).mean()),
        "frac_decisive_>_.95_or_<_.05": float(((y_all >= 0.95) | (y_all <= 0.05)).mean()),
        "mean_abs_delta": float(np.abs([it["delta"] for it in train_items + test_items]).mean()),
    }
    print("target diag:", json.dumps(diag, indent=2))
    if args.stats_only:
        return

    out_dir = Path(args.out_root) / f"{args.target_mode}_tau{args.tau}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2).to(device)
    # MPS：fp32 + 小批次控显存；bf16 autocast 对 ModernBERT 会 NaN；grad ckpt 会挂起，均默认关
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    use_amp = False
    pad_id = _TOK.pad_token_id
    if args.eval_only:
        ckpt_path = out_dir / "model.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"])
        train_minutes, step = 0.0, 0
        print(f"eval-only: loaded {ckpt_path}", flush=True)
    else:
        use_amp = args.amp and probe_amp(model, device)
        print(f"bf16 autocast: {use_amp} | grad ckpt: {args.grad_ckpt}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rng = np.random.default_rng(args.seed)
        ep_batches = make_batches(train_items, args.max_tokens, rng)
        steps_per_epoch = len(ep_batches)
        total_steps = args.epochs * steps_per_epoch if not args.max_steps else args.max_steps
        warmup = max(1, int(args.warmup_frac * total_steps))
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        print(f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup}", flush=True)

        step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
        done = False
        for ep in range(args.epochs):
            batches = ep_batches if ep == 0 else make_batches(train_items, args.max_tokens, rng)
            for idxs in batches:
                ids, att, y = collate(train_items, idxs, pad_id, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids=ids, attention_mask=att).logits
                logits = logits.float()
                logp = F.log_softmax(logits, -1)
                t = torch.stack([1 - y, y], -1)
                loss = -(t * logp).sum(-1).mean()
                if not math.isfinite(loss.item()):
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(f"loss {loss.item()} at step {step}; aborting before wasting GPU-hours")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if step % 50 == 0 and device.type == "mps":
                    torch.mps.empty_cache()
                run_loss += loss.item() * len(idxs); run_n += len(idxs)
                step += 1
                if step % 50 == 0:
                    with torch.no_grad():
                        p̄ = torch.softmax(logits, -1)[:, 1].mean().item()
                    print(f"  step {step:5d}/{total_steps} ep{ep + 1} loss={run_loss / run_n:.4f} "
                          f"p̄={p̄:.3f} bs={len(idxs)} lr={sched.get_last_lr()[0]:.2e} "
                          f"{step / (time.time() - t0):.2f} it/s", flush=True)
                    run_loss, run_n = 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    done = True; break
            if done:
                break
        train_minutes = (time.time() - t0) / 60
        print(f"training done in {train_minutes:.1f} min")

    # ---- 校准温度（calib 不曾进训练）----
    # 推理无激活存储，批次可以放大很多
    eval_budget = max(args.max_tokens, 12000)
    p_cal = predict(model, calib_items, pad_id, device, eval_budget, use_amp)
    y_cal = np.array([it["y"] for it in calib_items])
    T_fit = fit_temperature(p_cal, y_cal)
    T_use = min(5.0, max(0.5, T_fit))
    print(f"fitted temperature T={T_fit:.3f} (runtime clamp -> {T_use:.3f})")

    # ---- 评测 ----
    eval_recs = test_recs
    if args.eval_max_records:
        eval_recs = test_recs[:args.eval_max_records]
    keep = set()
    for r in eval_recs:
        keep.add(f"{r['prob_name']}_D{r['prob_D']}_r3")
    test_used = [it for it in test_items if it["rec"] in keep]
    p_test = predict(model, test_used, pad_id, device, eval_budget, use_amp)

    res = {
        "mode": args.target_mode, "tau": args.tau, "T_fit": T_fit, "T_use": T_use,
        "train_minutes": train_minutes, "steps": step, "diag": diag,
        "overall": evaluate(p_test, test_used, T_use),
        "per_func": {},
        "strata": strata(p_test, test_used),
    }
    res["per_func"] = {}
    for key in sorted({f"{it['func']}_D{it['dim']}" for it in test_used}):
        m = [i for i, it in enumerate(test_used) if f"{it['func']}_D{it['dim']}" == key]
        res["per_func"][key] = evaluate(p_test[np.array(m)], [test_used[i] for i in m], T_use)
    res["per_dim"] = {}
    for d in sorted({it["dim"] for it in test_used}):
        m = [i for i, it in enumerate(test_used) if it["dim"] == d]
        res["per_dim"][f"D{d}"] = evaluate(p_test[np.array(m)], [test_used[i] for i in m], T_use)

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if args.save_model:
        torch.save({"state_dict": model.state_dict(), "mode": args.target_mode, "tau": args.tau},
                   out_dir / "model.pt")
    print(json.dumps({k: v for k, v in res["overall"].items()}, indent=2))
    print("per-function acc:",
          {k: round(v["acc"], 4) for k, v in res["per_func"].items()})
    print("strata:", json.dumps(res["strata"], indent=2))
    print(f"saved -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
