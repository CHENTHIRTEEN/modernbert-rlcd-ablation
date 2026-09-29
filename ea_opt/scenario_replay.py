"""
冻结场景回放：把 run_exp_bert.py --dump-dir 抓下来的运行时场景
（每代：代理实际看到的锚点集合 + 候选 + 候选真值）当作冻结考卷，
用任意 ckpt 重放打分，对比排序质量随代数的演化。

用途（2026-09-29 D5 塌缩问题的决定性实验）：
  1. 用 raw 跑出的失败轨迹当考卷（场景=真实病灶分布，比合成簇保真），
     raw 自己考 + hard 考同一张卷 —— hard 若在 mid/late 段恢复 spearman 即机制确认；
  2. 顺带量化"归一化坐标带宽"：锚点∪候选联合 min-max 后，候选坐标占据的
     归一化带宽（宽带=正常，窄带=archive 老解把新簇压进角落的输入侧几何假说）。

用法：
  python scenario_replay.py --dump exp_bert_data_dumps/ --ckpt ../runs/ga54_raw_tau1.0_e2_D5_seed0/xxx.pt \
      [--ckpt ../runs/ga54_hard_tau1.0_e2_D5_seed0/xxx.pt ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_surrogate import (ModernBERTRelationSurrogate, joint_normalize)
from cluster_diag import diag_record


def load_dumps(dump_spec: str):
    recs = []
    p = Path(dump_spec)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    for fp in files:
        run_key = fp.stem
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["run_key"] = run_key
            recs.append(r)
    return recs


def coord_band(rec, n_anchor, n_cand):
    """锚点∪候选联合归一化后，候选坐标的平均带宽（每维 max-min 的均值）"""
    Xs = np.asarray(rec["X_train"])[:n_anchor]
    Us = np.asarray(rec["X_test"])[:n_cand]
    _, Nu = joint_normalize(Xs, Us)
    return float((Nu.max(axis=0) - Nu.min(axis=0)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dump", required=True, help="dump 目录或单个 jsonl")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--n-anchor-cap", type=int, default=15)
    ap.add_argument("--n-cand", type=int, default=15)
    ap.add_argument("--n-evidence", type=int, default=12)
    ap.add_argument("--beta", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-gens", type=int, default=0,
                    help=">0 时每条轨迹只回放前 N 代（省时间）")
    args = ap.parse_args()

    recs = load_dumps(args.dump)
    runs = sorted({r["run_key"] for r in recs})
    print(f"[dump] {len(runs)} 条轨迹共 {len(recs)} 代场景")
    # 场景本身的画像（与 ckpt 无关）：坐标带宽 + Δf 尺度
    for run in runs:
        rr = [r for r in recs if r["run_key"] == run]
        rr.sort(key=lambda r: r["gen"])
        if args.max_gens:
            rr = rr[:args.max_gens]
        n_third = max(1, len(rr) // 3)
        bands = [coord_band(r, args.n_anchor_cap, args.n_cand) for r in rr]
        df = [np.percentile(np.asarray(r["y_test"]), 75) -
              np.percentile(np.asarray(r["y_test"]), 25) for r in rr]
        seg = lambda sl: (f"带宽{np.mean(bands[sl]):.3f}/Δf-IQR{np.mean(df[sl]):.3g}")
        print(f"  {run}: {len(rr)} 代 | early {seg(slice(0, n_third))} | "
              f"mid {seg(slice(n_third, 2 * n_third))} | late {seg(slice(2 * n_third, None))}")

    for ckpt in args.ckpt:
        surrogate = ModernBERTRelationSurrogate(
            ckpt=ckpt, base_model=args.base_model, device=args.device,
            batch_size=args.batch_size, n_evidence=args.n_evidence,
            beta=args.beta, half=args.half, score_mode="mean")
        tag = Path(ckpt).parent.name
        acc = {"early": [], "mid": [], "late": []}
        res_acc = {"early": [], "mid": [], "late": []}
        for run in runs:
            rr = sorted((r for r in recs if r["run_key"] == run),
                        key=lambda r: r["gen"])
            if args.max_gens:
                rr = rr[:args.max_gens]
            n_third = max(1, len(rr) // 3)
            for i, r in enumerate(rr):
                d = diag_record(surrogate, r, args.n_anchor_cap, args.n_cand,
                                args.n_evidence, args.beta)
                phase = "early" if i < n_third else ("mid" if i < 2 * n_third else "late")
                acc[phase].append(d["rho_rank"])
                res_acc[phase].append(d["res"])
        print(f"\n===== {tag} =====")
        print(f"{'阶段':<8}{'spearman(排序)':>14}{'mean|p-0.5|':>12}{'n':>5}")
        for ph in ("early", "mid", "late"):
            if acc[ph]:
                print(f"{ph:<8}{np.nanmean(acc[ph]):>14.3f}"
                      f"{np.mean(res_acc[ph]):>12.4f}{len(acc[ph]):>5}")


if __name__ == "__main__":
    main()
