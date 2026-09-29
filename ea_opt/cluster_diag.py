"""
簇内排序分辨率诊断（2026-09-29 D5 在线塌缩问题的决定性实验工具）。

背景：ga54 raw 在线测评 D5 被 random 投票全面超越。已确诊机制=BT 软目标
y=σ(Δf/(s_f·τ)) 的 s_f 锚定种群尺度，收敛簇内 Δf≪s_f ⇒ 训练标签 y≈0.5 ⇒
模型在簇内对上输出纯平局（mean|p−0.5|≈0.003，排序 ρ≈0），且带抱团偏好
（分数与"距最优锚点距离"负相关），比无偏随机票更伤探索。

本脚本有两种场景模式（2026-09-29 校准发现：ga54 晚期快照虽 f 全体近平局但 x 仍散开，
raw 模型在其上不塌缩 ρ=0.30 —— 塌缩只发生在 LSEA 运行时自造的 x 收缩簇，GA 训练数据
从未覆盖该分布）：

  A. synthetic（主判据）：在 LZG/YLL 问题最优邻域按半径 r∈{0.3,0.1,0.03,0.01}（占界宽比例）
     高斯采样造"运行时簇"，画分辨率-r 衰减曲线 —— hard 若把曲线整体右移/抬高即恢复分辨率
  B. records（对照）：ga54 seed3 快照 early/late —— 验证非塌缩场景不退化

每场景报告三个量（复刻 bert_surrogate.predict 的 mean 计分路径）：
  1) spearman(score, −f)               排序质量
  2) mean|p−0.5|                       概率分辨率
  3) spearman(score, dist_best_anchor) 抱团偏好（负=偏好贴近最优锚点）

用法（服务器或本地）：
  python cluster_diag.py --ckpt ../runs/ga54_raw_tau1.0_e2_D5_seed0/xxx.pt \
      [--ckpt ../runs/ga54_hard_tau1.0_e2_D5_seed0/xxx.pt ...] \
      [--data ../data_ga54] [--skip-records]
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_surrogate import (ModernBERTRelationSurrogate, joint_normalize,
                            fmt_vec)


def load_test_records(data_dir: str, rep: int = 3):
    recs = []
    for p in sorted(Path(data_dir).glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["run_id"] % 10 == rep:
                    recs.append(r)
    return recs


def diag_record(surrogate, rec, n_anchor, n_cand, n_evidence, beta):
    """复刻 bert_surrogate.predict 的 mean 计分路径，返回逐候选诊断量。"""
    X_tr = np.asarray(rec["X_train"]); y_tr = np.asarray(rec["y_train"])
    X_te = np.asarray(rec["X_test"]);  y_te = np.asarray(rec["y_test"])
    order = np.argsort(y_tr)[:n_anchor]           # 与 anchor_cap 同语义：取 f 最优
    Xs, ys = X_tr[order], y_tr[order]
    rng = np.random.default_rng([rec["run_id"], 7])
    sel = rng.choice(len(X_te), min(n_cand, len(X_te)), replace=False)
    Us, uys = X_te[sel], y_te[sel]

    Ns, Nu = joint_normalize(Xs, Us)
    ev_rng = np.random.default_rng([rec["run_id"], 0])
    texts_a, texts_b = [], []
    for i in range(len(Ns)):
        others = np.delete(np.arange(len(Ns)), i)
        chosen = ev_rng.choice(others, min(n_evidence, len(others)), replace=False)
        lines = [f"A is {'better' if ys[i] < ys[j] else 'worse'} than <{fmt_vec(Ns[j], beta)}>"
                 for j in chosen]
        ev = "\n".join(lines)
        a_str = fmt_vec(Ns[i], beta)
        for k in range(len(Us)):
            texts_a.append(f"A = <{a_str}> ; B = <{fmt_vec(Nu[k], beta)}>")
            texts_b.append(ev)

    p = surrogate._pair_probs(texts_a, texts_b).reshape(len(Ns), len(Us))
    scores = (1.0 - p).mean(axis=0)               # mean 计分，与在线口径一致

    # 抱团偏好：候选到最优锚点的归一化空间欧氏距离（近=抱团）
    best = int(np.argmin(ys))
    dist_best = np.linalg.norm(Nu - Ns[best], axis=1)

    # 种群紧致度：全体个体 y 的绝对 IQR（synthetic 模式用它展示 Δf 随半径的收缩）
    all_y = np.concatenate([rec["y_train"], rec["y_test"]])
    q75, q25 = np.percentile(all_y, [75, 25])

    return {
        "rho_rank": float(spearmanr(scores, -uys)[0]),
        "res": float(np.abs(p - 0.5).mean()),
        "rho_club": float(spearmanr(scores, dist_best)[0]),
        "spread": float(q75 - q25),
    }


def make_synthetic_recs(probs, radius_list, n_pts, seed0=1000):
    """在问题最优邻域按半径 r 造 LSEA 运行时风格的收缩簇（f 用问题真值求值）。
    x_best 来自 200 点 LHS 预搜索的最优解（问题无关，不需知道理论最优）。"""
    from run_exp_bert import build_problem
    from pymoo.operators.sampling.lhs import LHS
    recs = []
    for pn in probs:
        problem = build_problem(pn, 5)
        rng = np.random.default_rng([seed0, hash(pn) % (2**31)])
        np.random.seed(seed0 + len(pn))          # LHS 吃全局流
        X_seed = LHS().do(problem, 200).get("X")
        y_seed = problem.evaluate(X_seed).flatten()
        x_best = X_seed[int(np.argmin(y_seed))]
        lo, hi = np.asarray(problem.xl), np.asarray(problem.xu)
        for r in radius_list:
            Xc = x_best + r * (hi - lo) * rng.standard_normal((n_pts, 5))
            Xc = np.clip(Xc, lo, hi)
            yc = problem.evaluate(Xc).flatten()
            perm = rng.permutation(n_pts)
            half = n_pts // 2
            recs.append({
                "prob_name": pn, "gen": int(round(-np.log10(r) * 10)),  # 编码半径
                "radius": r, "run_id": seed0 + int(round(r * 10000)) + len(pn),
                "X_train": Xc[perm[:half]].tolist(), "y_train": yc[perm[:half]].tolist(),
                "X_test": Xc[perm[half:]].tolist(), "y_test": yc[perm[half:]].tolist(),
            })
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", action="append", required=True,
                    help="可传多个，逐个对照（如 raw 与 hard）")
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--data", default="../data_ga54")
    ap.add_argument("--probs", default="LZG01,LZG02,YLLF01,YLLF02",
                    help="synthetic 模式的问题（=在线塌缩的 4 个 D5 问题）")
    ap.add_argument("--radius-list", default="0.3,0.1,0.03,0.01",
                    help="簇半径（占界宽比例），衰减曲线横轴")
    ap.add_argument("--n-pts", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3,
                    help="每 (问题,半径) 重复次数（不同随机器）")
    ap.add_argument("--n-anchor-cap", type=int, default=15)
    ap.add_argument("--n-cand", type=int, default=15)
    ap.add_argument("--n-evidence", type=int, default=12)
    ap.add_argument("--beta", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-records", action="store_true",
                    help="跳过 ga54 快照对照模式")
    ap.add_argument("--n-records", type=int, default=12)
    args = ap.parse_args()

    syn_recs = []
    radii = [float(x) for x in args.radius_list.split(",")]
    for rep in range(args.repeats):
        for pn in args.probs.split(","):
            for r in radii:
                syn_recs.extend(make_synthetic_recs([pn], [r], args.n_pts,
                                                    seed0=1000 + rep * 100))
    print(f"[synthetic] {len(args.probs.split(','))} 问题 × 半径 {radii} × "
          f"{args.repeats} 重复 = {len(syn_recs)} 条收缩簇记录")

    ctrl_recs = []
    if not args.skip_records:
        recs = load_test_records(args.data)
        early = [r for r in recs if r["gen"] <= 20]
        late = [r for r in recs if r["gen"] >= 80]
        funcs = sorted({r["prob_name"] for r in recs})
        fr = np.random.default_rng(0)
        picked = fr.choice(funcs, args.n_records // 2, replace=False)
        for fn in picked:
            e = [r for r in early if r["prob_name"] == fn]
            l = [r for r in late if r["prob_name"] == fn]
            if e: ctrl_recs.append(e[int(fr.integers(len(e)))])
            if l: ctrl_recs.append(l[int(fr.integers(len(l)))])
        print(f"[records] seed3 快照对照 {len(ctrl_recs)} 条")

    for ckpt in args.ckpt:
        surrogate = ModernBERTRelationSurrogate(
            ckpt=ckpt, base_model=args.base_model, device=args.device,
            batch_size=args.batch_size, n_evidence=args.n_evidence,
            beta=args.beta, half=args.half, score_mode="mean")

        # --- A. synthetic：分辨率-r 衰减曲线（主判据）---
        rows = []
        for rec in syn_recs:
            d = diag_record(surrogate, rec, args.n_anchor_cap, args.n_cand,
                            args.n_evidence, args.beta)
            d.update(radius=rec["radius"], func=rec["prob_name"])
            rows.append(d)
        tag = Path(ckpt).parent.name
        print(f"\n===== {tag} · synthetic 收缩簇（主判据）=====")
        print(f"{'半径r':>8}{'spearman(排序)':>14}{'mean|p-0.5|':>12}{'抱团ρ':>9}{'簇内Δf/初始Δf':>14}{'n':>4}")
        base_spread = {}
        for r in radii:
            rs = [x for x in rows if x["radius"] == r]
            sp = [x["spread"] for x in rs]
            base_spread[r] = np.mean(sp)
        r0 = max(radii)
        for r in sorted(radii, reverse=True):
            rs = [x for x in rows if x["radius"] == r]
            rel = np.mean([x["spread"] for x in rs]) / (base_spread[r0] + 1e-30)
            print(f"{r:>8g}{np.nanmean([x['rho_rank'] for x in rs]):>14.3f}"
                  f"{np.mean([x['res'] for x in rs]):>12.4f}"
                  f"{np.nanmean([x['rho_club'] for x in rs]):>9.3f}"
                  f"{rel:>14.4g}{len(rs):>4}")

        # --- B. records 对照 ---
        if ctrl_recs:
            crows = []
            for rec in ctrl_recs:
                d = diag_record(surrogate, rec, args.n_anchor_cap, args.n_cand,
                                args.n_evidence, args.beta)
                d.update(group="late" if rec["gen"] >= 80 else "early",
                         clean=not rec["prob_name"].startswith("IOH_n"))
                crows.append(d)
            print(f"----- {tag} · ga54 快照对照（预期不塌缩）-----")
            for group in ("early", "late"):
                for clean in (True, False):
                    rs = [r for r in crows if r["group"] == group and r["clean"] == clean]
                    if not rs:
                        continue
                    print(f"{group}/{'clean' if clean else 'noisy':<7}"
                          f"{np.nanmean([r['rho_rank'] for r in rs]):>10.3f}"
                          f"{np.mean([r['res'] for r in rs]):>10.4f}"
                          f"{np.nanmean([r['rho_club'] for r in rs]):>9.3f}"
                          f"{len(rs):>4}")


if __name__ == "__main__":
    main()
