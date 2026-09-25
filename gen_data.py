#!/usr/bin/env python3
"""
按 R2SAEA 的数据格式构造关系判断测试数据。

- 测试函数取自 ioh (BBOB 真值套件): Sphere(1), Ellipsoid(2), Rastrigin(3), Rosenbrock(8)
- 维度: 5D / 10D；每个 (函数, 维度) 用不同 instance 生成 n_reps 份
- 采样: 拉丁超立方 (LHS)，与 R2SAEA 初始种群采样方式一致
- 输出: 与 R2SAEA test_data 相同 schema 的 JSONL
    run_id / prob_name / prob_D / generation / fes / timestamp /
    X_train(30,D) / y_train(30,) / X_test(30,D) / y_test(30,)
  X 存原始空间数值（归一化留给测试脚本做，与 LLM_Relation_Fitness.normalize 相同）
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import ioh

FUNCS = {
    "Sphere": 1,
    "Ellipsoid": 2,
    "Rastrigin": 3,
    "Rosenbrock": 8,
}


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """拉丁超立方采样，返回 [0,1)^(n,d)"""
    u = rng.random((n, d))
    perms = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    return (perms + u) / n


def make_record(problem, n_train: int, n_test: int, rng: np.random.Generator, dim: int):
    lb, ub = problem.bounds.lb[0], problem.bounds.ub[0]
    X = lb + lhs(n_train + n_test, dim, rng) * (ub - lb)
    y = np.array([problem(x) for x in X])
    return {
        "run_id": None,  # 由调用方填充
        "prob_name": f"IOH_{problem.meta_data.name}",
        "prob_D": dim,
        "generation": 0,
        "fes": n_train + n_test,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "X_train": X[:n_train].tolist(),
        "y_train": y[:n_train].tolist(),
        "X_test": X[n_train:].tolist(),
        "y_test": y[n_train:].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=Path(__file__).parent / "data")
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--funcs", nargs="+", default=list(FUNCS.keys()),
                    choices=list(FUNCS.keys()))
    ap.add_argument("--n-train", type=int, default=30, help="训练(锚点)点数, 对齐 R2SAEA pop_size")
    ap.add_argument("--n-test", type=int, default=30, help="候选测试点数")
    ap.add_argument("--n-reps", type=int, default=3, help="每个(函数,维度)的重复次数, 用不同 instance")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fname in args.funcs:
        fid = FUNCS[fname]
        for dim in args.dims:
            records = []
            for rep in range(1, args.n_reps + 1):
                problem = ioh.get_problem(
                    fid, instance=rep, dimension=dim,
                    problem_class=ioh.ProblemClass.REAL,
                )
                rng = np.random.default_rng(args.seed + fid * 1000 + dim * 10 + rep)
                rec = make_record(problem, args.n_train, args.n_test, rng, dim)
                rec["run_id"] = int(f"{args.seed}{fid:02d}{dim:02d}{rep}")
                records.append(rec)

            out = args.out_dir / f"IOH_{fname}_D{dim}_{args.seed}_lhs.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
