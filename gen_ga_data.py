#!/usr/bin/env python3
"""
GA 轨迹后训练数据集（R2SAEA 论文 III-B2 同款协议，bbob54 函数集 + 真实优化轨迹）。

协议（用户定稿 2026-09-27）：
- 54 函数（BBOB f1-24 无噪 + BBOB-noisy f101-130）× instance 1-3 × seed 1-5，D5
- 每条轨迹：pymoo 默认 GA（SBX 0.9/η15 + 多项式变异 η20 + 锦标赛，pop_size=100）跑 100 代
- 每 10 代抽取当代种群（100 个体，gens 10..100 共 10 个快照）作为一条 record：
  随机置换后前 50 作 context/锚点（X_train）、后 50 作候选（X_test），y 为 cocoex 真值
- 与旧协议的兼容约定（train_soft_ablation.py 依赖）：rep=seed∈{1..5} ⇒ rep=run_id%10，
  切分=seed {1,2,4,5} 训练、seed 3 测试（held-out 轨迹）、f1/f2(Sphere/Ellipsoid) 的
  seed2 只做温度校准（CALIB_FUNCS 不变）；run_id=int(f"{seed_base}{fid:03d}{inst:02d}{gen:03d}{seed}")
- 输出：data_ga54/ 下每 (函数,instance,seed) 一个 .jsonl.gz（10 条 record，共 810 文件/8100 条）。
  gz 为控 git 体积；test_bert.load_records 已兼容 .gz

确定性：GA 由 pymoo seed 决定；noisy 函数的噪声随求值流推进，同版本 pymoo/cocoex + 同 seed
重放逐位一致（求解顺序固定）。50/50 切分 rng 独立于 GA seed。

用法：
    python gen_ga_data.py                          # 全量 8100 条（~20-40 分钟）
    python gen_ga_data.py --fids 1 101 --seeds 1   # 冒烟
"""
import json
import argparse
import re
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import cocoex
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from gen_bbob_data import IOH_NAMES, NOISE_SLUG


class CocoexProblem(Problem):
    """pymoo 包装：逐行调 cocoex 求值（cocoex 只允许评估当前问题，须单 problem 生命周期）。

    注意：cocoex Problem 不可 deepcopy/pickle，而 pymoo save_history 每代 deepcopy 算法
    （含 problem）。因此 cocoex 对象必须经闭包引用（函数 deepcopy 按引用传递），不能挂实例属性。
    """

    def __init__(self, coco_problem, n_var):
        cp = coco_problem

        class _Eval(Problem):
            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.array([float(cp(x)) for x in X])

        self._impl = _Eval(n_var=n_var, xl=coco_problem.lower_bounds[0],
                           xu=coco_problem.upper_bounds[0])
        # 透传 pymoo 需要的属性
        self.n_var = self._impl.n_var
        self.xl = self._impl.xl
        self.xu = self._impl.xu

    def minimize_with_history(self, seed, n_gen, pop_size):
        return minimize(self._impl, GA(pop_size=pop_size), ("n_gen", n_gen),
                        seed=seed, save_history=True, verbose=False)


def make_record(X, y, dim, fid, instance, seed, gen, fes, suite, prob_name, noise, coco_id, seed_base):
    idx = np.random.default_rng(zlib.crc32(f"{prob_name}|{instance}|{gen}|{seed}".encode())).permutation(len(X))
    X, y = X[idx], y[idx]
    n = len(X) // 2
    return {
        "run_id": int(f"{seed_base}{fid:03d}{instance:02d}{gen:03d}{seed}"),
        "prob_name": prob_name,
        "prob_D": dim,
        "generation": gen,
        "fes": fes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "suite": suite, "fid": fid, "instance": instance, "seed": seed, "gen": gen,
        "noise": noise, "coco_id": coco_id, "algo": "pymoo-GA(pop100)", "pop_size": len(X),
        "X_train": X[:n].tolist(),
        "y_train": y[:n].tolist(),
        "X_test": X[n:].tolist(),
        "y_test": y[n:].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=Path(__file__).parent / "data_ga54")
    ap.add_argument("--dims", type=int, nargs="+", default=[5])
    ap.add_argument("--fids", type=int, nargs="+", default=None,
                    help="只跑这些 fid（1-24 无噪、101-130 noisy）；默认全部 54 个")
    ap.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=20260927, help="run_id 前缀（数据集版本标识）")
    args = ap.parse_args()

    args.out_dir = Path(args.out_dir)
    fids = args.fids or list(range(1, 25)) + list(range(101, 131))
    groups = (("bbob", sorted(f for f in fids if 1 <= f <= 24)),
              ("bbob-noisy", sorted(f for f in fids if 101 <= f <= 130)))
    gens = list(range(args.snapshot_every, args.n_gen + 1, args.snapshot_every))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import gzip

    n_rec, n_run = 0, 0
    for suite_name, group in groups:
        if not group:
            continue
        for dim in args.dims:
            flist = ",".join(str(group.index(f) + 1) for f in group)
            inst_str = ",".join(map(str, args.instances))
            suite = cocoex.Suite(
                suite_name, "",
                f"dimensions: {dim} function_indices: {flist} instance_indices: {inst_str}")
            for problem in suite:          # cocoex 只允许评估当前问题：单 problem 生命周期内跑完其全部 seed
                m = re.match(r"bbob(?:_noisy)?_f(\d+)_i(\d+)_d(\d+)", problem.id)
                fid, instance = int(m.group(1)), int(m.group(2))
                if fid not in group or instance not in args.instances:
                    continue
                if suite_name == "bbob":
                    prob_name, noise, tag = f"IOH_{IOH_NAMES[fid]}", None, IOH_NAMES[fid]
                else:
                    nm = problem.name.split("_noise_model")[0]
                    noise = NOISE_SLUG.get(nm, nm)
                    prob_name, tag = f"IOH_n{fid}_{noise}", f"n{fid}_{noise}"

                for seed in args.seeds:
                    prob = CocoexProblem(problem, dim)
                    res = prob.minimize_with_history(seed, args.n_gen, args.pop_size)
                    fp = args.out_dir / f"IOH_{tag}_I{instance}_S{seed}_ga.jsonl.gz"
                    with gzip.open(fp, "wt", encoding="utf-8") as f:
                        for gen in gens:
                            pop = res.history[gen - 1].pop      # history[i] = 第 i+1 代
                            X, F = pop.get("X"), pop.get("F")[:, 0]
                            rec = make_record(X, np.asarray(F, float), dim, fid, instance,
                                              seed, gen, args.pop_size * (gen + 1), suite_name,
                                              prob_name, noise, problem.id, args.seed_base)
                            f.write(json.dumps(rec) + "\n")
                            n_rec += 1
                    n_run += 1
                    if n_run % 27 == 0:
                        print(f"[{suite_name} D{dim}] {n_run} runs / {n_rec} records", flush=True)

    print(f"wrote {n_rec} records ({n_run} GA runs) -> {args.out_dir}")


if __name__ == "__main__":
    main()
