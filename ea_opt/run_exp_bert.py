"""
用我们的 ModernBERT 关系代理跑 R2SAEA 同款 LSEA 优化实验（SOP：LZG + YLL）。

与 R2SAEA run_exp.py + LSEA 默认参数逐项相同的部分（论文 Table I 的 SOP 设定）：
  pop_size=30, n_offsprings=30, n_evals=300, tao=50, sampling=LHS,
  survival=FitnessSurvival, reproduction=VWH_Local_Reproduction_unevaluate(VWH(M=15), Pb=0.2, Pc=0.2)
  —— 每代代理给全部 offspring 打分，只真实评估得分最高的 1 个解，其余进 unevaluated_pop 喂 EDA。

有意不同的三处（其余全部照抄）：
  1) 代理 = 微调 ModernBERT 关系模型（零样本、每代不重训），替代 vLLM/Qwen 关系模型
  2) 修复 R2SAEA lsea.py:254 的 float 切片 bug（原版 `sorted_ind[self.pop_size/2:-1]` 必
     TypeError，按 UEDA 基类对称语义改为取"得分最高的一半"喂 EDA）
  3) 每次运行显式设种子，可复现、可断点续跑
另有两处 numpy 2.x 兼容修复（见 VWH_Local_Reproduction_unevaluate_fixed 与 edamodel.py 头注）。

本目录自包含：problem/{LZG,YLL}.py 与 edamodel.py（VWH+local_search）vendor 自 R2SAEA
仓库，协议出处见各文件头注。代理实现在 bert_surrogate.py。

用法（服务器，ckpt = train_soft_ablation.py --save-model 产出的 model.pt）：
  # 冒烟：单函数单次
  python run_exp_bert.py --ckpt /path/to/runs/hard_tau1.0/model.pt --probs LZG01 --dims 5 --runs 1
  # 论文同款网格：15 问题 x D{5,10,20} x 10 次（预计数十 GPU 小时，建议 nohup + 断点续跑）
  nohup python -u run_exp_bert.py --ckpt /path/to/model.pt --runs 10 > exp_bert.log 2>&1 &
  # 提速：fp16 + 锚点数上限 15（代理上下文变小，偏离 tao=50，属额外消融）
  python run_exp_bert.py --ckpt ... --half --n-anchor-cap 15 ...

依赖：torch / transformers / pymoo(0.6.x) / numpy / scipy（pymoo 需要）。不需要 langchain。
"""
import sys
import os
import copy
import json
import math
import random
import time
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.population import Population
from pymoo.core.callback import Callback
from pymoo.core.infill import InfillCriterion
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.optimize import minimize

from edamodel import VWH, local_search
from bert_surrogate import ModernBERTRelationSurrogate, RandomVotingSurrogate


class VWH_Local_Reproduction_unevaluate_fixed(InfillCriterion):
    """
    照抄 R2SAEA algorithm/util/reproduction.py::VWH_Local_Reproduction_unevaluate，唯一改动：
    local_search 传入的 ys 先 flatten。原版把 (NL,1) 的 F 列向量直接传入，numpy 2.x 下
    `Xs_new[r0, d] = -b / (2.0 * a)`（把 (1,) 数组赋给标量槽）抛 ValueError（numpy 1.x 容忍）。
    """

    def __init__(self, eda=None, Pb=0.2, Pc=0.2, **kwargs):
        super().__init__(**kwargs)
        self.eda = eda or VWH(M=15)
        self.Pb = Pb
        self.Pc = Pc

    def do(self, problem, pop, n_offsprings, **kwargs):
        algorithm = kwargs['algorithm']
        unevaluated_pop = kwargs['unevaluated_pop']
        xs, ys = pop.get('X'), pop.get('F')
        I = np.argsort(ys.flatten())
        xs = xs[I, :]
        ys = ys[I]

        self.eda.update(np.concatenate(
            [xs, unevaluated_pop[:int(algorithm.pop_size / 2), :]], axis=0))

        xs_eda = self.eda.sample(n_offsprings)

        NL = int(np.floor(algorithm.pop_size * self.Pb))
        xs_ls = local_search(xs[:NL, :], np.asarray(ys[:NL]).flatten())  # <-- 唯一改动

        I = np.floor(np.random.random((algorithm.pop_size, 1))
                     * (xs_ls.shape[0] - 2)).astype(int).flatten()
        xtmp = xs_ls[I, :]
        mask = np.random.random((algorithm.pop_size, problem.n_var)) < self.Pc
        xs_eda[mask] = xtmp[mask]

        lb_matrix = problem.xl * np.ones(shape=xs_eda.shape)
        ub_matrix = problem.xu * np.ones(shape=xs_eda.shape)
        pos = xs_eda < problem.xl
        xs_eda[pos] = 0.5 * (xs[pos] + lb_matrix[pos])
        pos = xs_eda > problem.xu
        xs_eda[pos] = 0.5 * (xs[pos] + ub_matrix[pos])

        return Population.new(X=xs_eda)


# ---------------- 算法：照抄 R2SAEA algorithm/lsea.py 的 UEDA/LSEA ----------------

class UEDA(GeneticAlgorithm):
    """与 R2SAEA lsea.py::UEDA 相同，仅去掉 LLM 相关导入。"""

    def __init__(self, pop_size=50, tao=100, sampling=LHS(),
                 reproduction=None, output=SingleObjectiveOutput(),
                 surrogate=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, output=output,
                         survival=FitnessSurvival(), **kwargs)
        self.reproduction = reproduction or VWH_Local_Reproduction_unevaluate_fixed()
        self.archive_eva = Population()
        self.unevaluated_pop = None
        self.tao = tao
        self.surrogate = surrogate

    def _initialize_advance(self, infills=None, **kwargs):
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var))
        self.archive_eva = Population.merge(self.pop, self.archive_eva)
        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))
        if self.surrogate is None:
            raise Exception("surrogate model is None")

    def _infill(self):
        t_xs, t_ys = self.get_raw_training_data()
        self.training_surrogete_model(t_xs, t_ys)

        infills = self.reproduction.do(
            self.problem, self.pop, self.n_offsprings, algorithm=self,
            unevaluated_pop=self.unevaluated_pop)

        x_best, unevaluated_pop = self.surrogate_assisted_selection(infills)
        self.unevaluated_pop = unevaluated_pop
        return Population.new(X=x_best)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.archive_eva = Population.merge(self.archive_eva, infills)
        self.pop = self.survival.do(self.problem, self.archive_eva,
                                    n_survive=self.pop_size, algorithm=self, **kwargs)

    def get_raw_training_data(self):
        t_xs, t_ys = self.archive_eva.get("X"), self.archive_eva.get("F")
        if len(self.archive_eva) <= self.tao:
            return t_xs, t_ys.flatten()
        t = copy.deepcopy(t_ys).flatten()
        index = t.argsort()
        return t_xs[index[:self.tao], :], t_ys[index[:self.tao], :].flatten()

    def training_surrogete_model(self, Xs, ys):
        self.surrogate.fit(Xs, ys)


class LSEA_BERT(UEDA):
    """pairwise 关系投票版（对应 R2SAEA lsea.py::LSEA），surrogate 为 ModernBERT/随机投票代理。

    run_seed 会传给 surrogate.predict 做证据句选择的确定性随机流。
    """

    def __init__(self, pop_size=50, tao=50, sampling=LHS(),
                 output=SingleObjectiveOutput(), surrogate=None,
                 reproduction=None, run_seed=0, anchor_cap=0, **kwargs):
        self.run_seed = run_seed
        self.anchor_cap = anchor_cap
        super().__init__(pop_size=pop_size, tao=tao, sampling=sampling,
                         output=output, surrogate=surrogate,
                         reproduction=reproduction, **kwargs)

    def get_raw_training_data(self):
        t_xs, t_ys = super().get_raw_training_data()
        # 提速消融：锚点数截断（取 F 最优的前 anchor_cap 个；0 = 不截，保持 tao 语义）
        if self.anchor_cap and len(t_xs) > self.anchor_cap:
            order = np.argsort(t_ys.flatten())[:self.anchor_cap]
            return t_xs[order], t_ys.flatten()[order]
        return t_xs, t_ys

    def __deepcopy__(self, memo):
        # pymoo minimize 默认会 deepcopy 算法；torch 模型不可复制，代理整体共享引用
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('surrogate', 'reproduction'):
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def surrogate_assisted_selection(self, pop):
        Xs = pop.get('X')
        scores = self.surrogate.predict(Xs, seed=self.run_seed)

        # 得分越大越好：得分最高者做真实评估
        sorted_ind = np.argsort(scores.flatten())
        X_best = copy.deepcopy(Xs[sorted_ind[-1], :]).reshape(1, -1)

        # 修复原版 lsea.py:254 的 float 切片 bug：
        # 原文 `Xs[sorted_ind[self.pop_size/2:-1], :]` 必 TypeError；
        # 按 UEDA 基类对称语义（基类取 F 最优的一半喂 EDA），这里取得分最高的一半
        selected_decs = copy.deepcopy(Xs[sorted_ind[int(self.pop_size / 2):], :])
        return X_best, selected_decs


class ObjCallback(Callback):
    """照抄 R2SAEA savedata/call_back.py::Base_Callback（save_decs=False 分支）"""

    def __init__(self):
        super().__init__()
        self.data["objs"] = []
        self.data["fes"] = []

    def notify(self, algorithm):
        self.data["objs"].append(algorithm.pop.get("F"))
        self.data["fes"].append(algorithm.evaluator.n_eval)


# ---------------- 问题构造 ----------------

def build_problem(prob_name: str, n_var: int):
    if prob_name.startswith("LZG"):
        module = importlib.import_module("problem.LZG")
        cls = getattr(module, prob_name)
    elif prob_name.startswith("YLL"):
        module = importlib.import_module("problem.YLL")
        cls = getattr(module, f"YLLF{int(prob_name[4:]):02d}")
    else:
        raise ValueError(f"unknown problem {prob_name}")
    return cls(n_var=n_var)


def parse_probs(spec: str):
    """支持 'LZG01-04' / 'YLLF01-13' 区间与逗号混写，展开成显式列表"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            head, tail = tok.split("-")
            prefix = head[:4] if head.startswith("YLL") else head[:3]
            lo, hi = int(head[len(prefix):]), int(tail)
            for i in range(lo, hi + 1):
                out.append(f"{prefix}{i:02d}")
        else:
            out.append(tok)
    return out


# ---------------- 主流程 ----------------

def seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_surrogate(args):
    if args.surrogate == "random":
        return RandomVotingSurrogate()
    return ModernBERTRelationSurrogate(
        ckpt=args.ckpt, base_model=args.base_model, device=args.device,
        batch_size=args.batch_size, n_evidence=args.n_evidence, beta=args.beta,
        half=args.half, max_length=args.max_length, score_mode=args.score)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", default=None,
                    help="model.pt（state_dict）；缺省 = 原生 ModernBERT 底座（cls 基线）")
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--surrogate", choices=["bert", "random"], default="bert")
    ap.add_argument("--probs", type=str,
                    default="LZG01-04,YLLF01-09,YLLF12,YLLF13",
                    help="论文 Table I 的 15 个问题（YLLF10/11 与 LZG 重复被排除）")
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--n-evals", type=int, default=300)
    ap.add_argument("--tao", type=int, default=50)
    # 代理侧
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-evidence", type=int, default=12)
    ap.add_argument("--beta", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--half", action="store_true", help="CUDA 上 fp16 推理提速")
    ap.add_argument("--score", choices=["hard", "soft", "mean"], default="hard",
                    help="计分方式：hard=R2SAEA 硬票+ε权重投票（复刻）；"
                         "soft=同权重、p 线性扩展；mean=平均 P(候选优于锚点)（我们的计分，"
                         "推荐搭配 --n-anchor-cap 15）")
    ap.add_argument("--n-anchor-cap", type=int, default=0,
                    help=">0 时锚点数截到该值（提速用；0=全部 tao 个，偏离记录在案）")
    # 输出
    ap.add_argument("--out-dir", default="./exp_bert_data")
    ap.add_argument("--alg-name", default="MBERT-LSEA")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    flat_path = out_dir / "runs_flat.csv"
    summary_path = out_dir / "summary.csv"

    probs = parse_probs(args.probs)
    # 模型标识：取 ckpt 路径的 <probe>/<mode>_tau*，p1/p2/p2e2 天然区分；
    # key/summary/flat 全部按 tag 隔离，多个 checkpoint 可写同一个 out-dir 不串数据
    if args.surrogate == "random":
        surrogate_tag = "random"
    elif args.ckpt:
        p = Path(args.ckpt)
        surrogate_tag = (f"{p.parent.parent.name}/{p.parent.name}"
                         if str(p.parent.parent) not in ("", ".", "/") else p.parent.name)
    else:
        surrogate_tag = "raw-base"

    # 断点续跑：读已完成 key
    done = set()
    records = []
    if runs_path.exists():
        for line in open(runs_path, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                done.add(rec["key"])
    print(f"[config] probs={len(probs)} dims={args.dims} runs={args.runs} "
          f"pop={args.pop_size} evals={args.n_evals} tao={args.tao} "
          f"surrogate={args.surrogate}({surrogate_tag}) out={out_dir}")
    print(f"[resume] {len(done)} runs already done")

    est_pairs = min(args.tao, args.pop_size) * args.pop_size
    print(f"[est] ~{est_pairs} pairs/generation, {args.n_evals - args.pop_size} generations/run")

    # 代理整个实验只建一次（模型加载一次；fit 每代覆盖缓存数据）
    surrogate = make_surrogate(args)

    flat_header = "prob,surrogate_tag,dim,run,seed,best_f,wall_s,n_pairs_total,mean_vote_better"
    if not flat_path.exists():
        flat_path.write_text(flat_header + "\n")

    t_all = time.time()
    n_done_this_session = 0
    for prob_name in probs:
        for Dim in args.dims:
            for r in range(1, args.runs + 1):
                seed = args.seed_base + r - 1
                key = f"{surrogate_tag}/{prob_name}_D{Dim}_r{r}"
                if key in done:
                    continue

                problem = build_problem(prob_name, Dim)
                seed_all(seed)
                # 每次运行给一份全新的 EDA/reproduction（避免默认参数共享实例的状态残留）
                repro = VWH_Local_Reproduction_unevaluate_fixed(eda=VWH(M=15))
                algorithm = LSEA_BERT(pop_size=args.pop_size, tao=args.tao,
                                      surrogate=surrogate, reproduction=repro,
                                      callback=ObjCallback(), run_seed=seed,
                                      anchor_cap=args.n_anchor_cap)

                t0 = time.time()
                res = minimize(problem, algorithm, ("n_evals", args.n_evals),
                               verbose=args.verbose, copy_algorithm=False)
                wall = time.time() - t0

                # 收敛曲线：每代种群最优的累计最优
                fes_list = algorithm.callback.data["fes"]
                objs_list = algorithm.callback.data["objs"]
                curve_fes, curve_best, best = [], [], math.inf
                for fes, objs in zip(fes_list, objs_list):
                    best = min(best, float(np.min(objs)))
                    curve_fes.append(int(fes))
                    curve_best.append(best)

                stats = surrogate.last_stats
                rec = {
                    "key": key,
                    "alg": args.alg_name,
                    "prob": prob_name, "dim": Dim, "run": r, "seed": seed,
                    "surrogate": args.surrogate, "surrogate_tag": surrogate_tag,
                    "ckpt": str(args.ckpt) if args.ckpt else None,
                    "pop_size": args.pop_size, "n_evals": args.n_evals, "tao": args.tao,
                    "best_f": curve_best[-1] if curve_best else None,
                    "wall_s": round(wall, 1),
                    "n_pairs_total": stats.get("n_pairs", 0),
                    "mean_vote_better": stats.get("frac_better_votes", None),
                    "curve_fes": curve_fes, "curve_best": curve_best,
                    "config": {"n_evidence": args.n_evidence, "beta": args.beta,
                               "max_length": args.max_length, "half": args.half,
                               "score": args.score,
                               "n_anchor_cap": args.n_anchor_cap},
                }

                with open(runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                records.append(rec)
                done.add(key)
                n_done_this_session += 1

                with open(flat_path, "a") as f:
                    f.write(f"{prob_name},{surrogate_tag},{Dim},{r},{seed},{rec['best_f']:.6g},"
                            f"{rec['wall_s']},{rec['n_pairs_total']},"
                            f"{rec['mean_vote_better']:.4f}\n")

                print(f"[{key}] best_f={rec['best_f']:.6g} wall={wall:.0f}s "
                      f"vote_better={rec['mean_vote_better']:.3f} "
                      f"(session done={n_done_this_session}, "
                      f"elapsed={(time.time() - t_all) / 60:.0f}min)", flush=True)

    # ---- 汇总（论文 result.csv 风格：均值/标准差/运行数；按模型 tag 分组） ----
    agg = {}
    alg_by_tag = {}
    for rec in records:
        tag = rec.get("surrogate_tag", "unknown")
        agg.setdefault((rec["prob"], rec["dim"], tag), []).append(rec["best_f"])
        alg_by_tag.setdefault(tag, rec.get("alg", args.alg_name))
    lines = ["算法,问题,维度,均值,标准差,运行数"]
    for (prob, dim, tag) in sorted(agg, key=lambda k: (k[2], k[0], k[1])):
        vals = np.array(agg[(prob, dim, tag)], dtype=float)
        lines.append(f"{alg_by_tag[tag]}[{tag}],{prob},{dim},{vals.mean():.6f},"
                     f"{vals.std(ddof=0):.6f},{len(vals)}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {n_done_this_session} new runs, total {len(records)}; "
          f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
