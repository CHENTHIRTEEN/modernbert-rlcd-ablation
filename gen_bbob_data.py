#!/usr/bin/env python3
"""
BBOB + BBOB-noisy 全套件的后训练数据生成器（54 函数 × 3 instance × 100 LHS 点）。

套件与协议：
- bbob (f1-f24, 无噪) + bbob-noisy (f101-f130, gauss/uniform/cauchy 噪声)，经 coco-experiment
  （COCO 官方 Python 接口，import 名 cocoex）。ioh 0.3.x 不含 noisy 套件，故改用 cocoex。
- 每个 (函数, instance, 维度) 用 LHS 采 100 个点（[-5,5]^D，与 R2SAEA 初始种群采样一致），
  随机置换后前 50 个作 context/锚点种群（X_train），后 50 个作候选（X_test）。
  先置换是为了避免 LHS 分层导致前 50 点系统性偏向半值域。
- noisy 函数的噪声随求值流推进（每个 y 是该求值位置的一次抽样，cocoex 拒绝对同 x 重复求值）：
  固定 LHS 种子 + 固定求值顺序 ⇒ 数据集完全可复现（已验证同序重放逐位一致）。

与旧协议（gen_data.py）的兼容约定（train_soft_ablation.py 依赖）：
- prob_name 无噪 = "IOH_{ioh名}"（f1=IOH_Sphere、f2=IOH_Ellipsoid ⇒ CALIB_FUNCS 不变），
  noisy = "IOH_n{fid}_{gauss|unif|cauchy}"
- rep = instance ∈ {1,2,3} ⇒ rep1/rep2 进 train+calib、rep3 进 test 的既有切分不变
- run_id = int(f"{seed}{fid:03d}{dim:02d}{rep}")，rep = run_id % 10
- schema 同 R2SAEA test_data（run_id/prob_name/prob_D/.../X_train/y_train/X_test/y_test），
  另附 suite/fid/instance/noise/coco_id 溯源字段

用法：
    python gen_bbob_data.py                                   # 全量 486 条 record（54×3×3维）
    python gen_bbob_data.py --dims 5                          # 只生成 D5
    python gen_bbob_data.py --fids 1 101 --out-dir data_smoke # 冒烟
"""
import json
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import cocoex

# ioh 对 f1-f24 的命名（硬编码，服务器免装 ioh；与旧 data/ 的 prob_name 对齐）
IOH_NAMES = {
    1: "Sphere", 2: "Ellipsoid", 3: "Rastrigin", 4: "BuecheRastrigin",
    5: "LinearSlope", 6: "AttractiveSector", 7: "StepEllipsoid", 8: "Rosenbrock",
    9: "RosenbrockRotated", 10: "EllipsoidRotated", 11: "Discus", 12: "BentCigar",
    13: "SharpRidge", 14: "DifferentPowers", 15: "RastriginRotated", 16: "Weierstrass",
    17: "Schaffers10", 18: "Schaffers1000", 19: "GriewankRosenbrock", 20: "Schwefel",
    21: "Gallagher101", 22: "Gallagher21", 23: "Katsuura", 24: "LunacekBiRastrigin",
}

NOISE_SLUG = {"gaussian": "gauss", "uniform": "unif", "cauchy": "cauchy"}


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """拉丁超立方采样，返回 [0,1)^(n,d)（与 gen_data.py 同实现）"""
    u = rng.random((n, d))
    perms = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    return (perms + u) / n


def make_record(problem, n_train: int, rng: np.random.Generator, dim: int,
                fid: int, instance: int, suite: str, prob_name: str, noise, seed: int):
    lb, ub = float(problem.lower_bounds[0]), float(problem.upper_bounds[0])
    X = lb + lhs(100, dim, rng) * (ub - lb)
    X = X[rng.permutation(len(X))]                 # 打乱 LHS 分层，再切 50/50
    y = np.array([float(problem(x)) for x in X])   # 单遍求值；noisy 每个值是一次噪声抽样
    return {
        "run_id": int(f"{seed}{fid:03d}{dim:02d}{instance}"),
        "prob_name": prob_name,
        "prob_D": dim,
        "generation": 0,
        "fes": 100,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "suite": suite, "fid": fid, "instance": instance, "noise": noise,
        "coco_id": problem.id,
        "X_train": X[:n_train].tolist(),
        "y_train": y[:n_train].tolist(),
        "X_test": X[n_train:].tolist(),
        "y_test": y[n_train:].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=Path(__file__).parent / "data_bbob54")
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--fids", type=int, nargs="+", default=None,
                    help="只生成这些 fid（1-24 无噪、101-130 noisy）；默认全部 54 个")
    ap.add_argument("--n-train", type=int, default=50, help="context/锚点数（100 点的一半）")
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    assert args.n_train < 100, "协议：每个 (函数,instance) 共 100 个 LHS 点，其余作候选"
    args.out_dir = Path(args.out_dir)
    fids = args.fids or list(range(1, 25)) + list(range(101, 131))
    groups = (("bbob", sorted(f for f in fids if 1 <= f <= 24)),
              ("bbob-noisy", sorted(f for f in fids if 101 <= f <= 130)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_rec = 0
    for suite_name, group in groups:
        if not group:
            continue
        for dim in args.dims:
            flist = ",".join(str(group.index(f) + 1) for f in group)  # suite 内 1-based 序号
            suite = cocoex.Suite(
                suite_name, "",
                f"dimensions: {dim} function_indices: {flist} instance_indices: 1-3")
            for problem in suite:          # cocoex 只允许评估当前问题：顺序单遍
                m = re.match(r"bbob(?:_noisy)?_f(\d+)_i(\d+)_d(\d+)", problem.id)
                fid, instance = int(m.group(1)), int(m.group(2))
                if fid not in group:
                    continue
                if suite_name == "bbob":
                    prob_name, noise, tag = f"IOH_{IOH_NAMES[fid]}", None, IOH_NAMES[fid]
                else:
                    nm = problem.name.split("_noise_model")[0]
                    noise = NOISE_SLUG.get(nm, nm)
                    prob_name, tag = f"IOH_n{fid}_{noise}", f"n{fid}_{noise}"
                rng = np.random.default_rng(args.seed + fid * 1000 + dim * 10 + instance)
                rec = make_record(problem, args.n_train, rng, dim,
                                  fid, instance, suite_name, prob_name, noise, args.seed)
                # instance 1..3 依次写进同一个 (函数,维度) 文件，与旧 data/ 布局一致
                out = args.out_dir / f"IOH_{tag}_D{dim}_{args.seed}_lhs.jsonl"
                with open(out, "w" if instance == 1 else "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                n_rec += 1
            print(f"[{suite_name}] D{dim} done ({n_rec} records so far)", flush=True)

    print(f"wrote {n_rec} records -> {args.out_dir}")


if __name__ == "__main__":
    main()
