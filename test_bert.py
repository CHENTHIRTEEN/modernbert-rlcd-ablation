#!/usr/bin/env python3
"""
用原生(未后训练) ModernBERT 测试其对解对关系 (A better/worse than B) 的判断能力。

x 的处理与 R2SAEA 完全一致 (src/llm4surrogate/model.py):
  1. train∪test 按维联合 min-max 归一化到 [0,1]
  2. 固定 beta=5 位小数, 逗号+空格分隔, 尖括号包裹
  3. 证据句式 "A is better/worse than <...>" (锚点对其余训练点的真实序关系)

三种模式:
  cls : [CLS] A = <...> ; B = <...>            -> AutoModelForSequenceClassification
        (分类头随机初始化, 是微调前的 pipeline 基线, 预期 ~50%)
  ctx : [CLS] A = <...> ; B = <...> [SEP] 证据句 -> 同上, 但带上锚点关系上下文
  mlm : "A = <...> ; B = <...> . A is [MASK] than B ." -> AutoModelForMaskedLM
        (用预训练 MLM 头比较 'better'/'worse' 两个词元的 logits,
         这是"原生能力"更有意义的零样本读数)

标签: 最小化问题, y_A < y_B -> better(1), 否则 worse(0)
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

# ---------------- R2SAEA 式文本化 ----------------

def fmt_vec(v: np.ndarray, beta: int) -> str:
    return ", ".join(f"{x:.{beta}f}" for x in v)


def joint_normalize(X_train: np.ndarray, X_test: np.ndarray):
    """对应 LLM_Base.normalize: train∪test 联合 min-max"""
    comb = np.vstack([X_train, X_test])
    lo, hi = comb.min(axis=0), comb.max(axis=0)
    rng = np.maximum(hi - lo, 1e-12)
    return (X_train - lo) / rng, (X_test - lo) / rng


def evidence_block(Ns: np.ndarray, ys: np.ndarray, anchor: int,
                   chosen: np.ndarray, beta: int) -> str:
    """锚点与其余训练点的已知关系句 (R2SAEA 'A is better than <...>')"""
    lines = []
    for j in chosen:
        word = "better" if ys[anchor] < ys[j] else "worse"
        lines.append(f"A is {word} than <{fmt_vec(Ns[j], beta)}>")
    return "\n".join(lines)


def load_records(data_dir: Path):
    records = []
    for fp in sorted(data_dir.glob("*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            d = json.loads(line)
            d["X_train"] = np.array(d["X_train"])
            d["y_train"] = np.array(d["y_train"])
            d["X_test"] = np.array(d["X_test"])
            d["y_test"] = np.array(d["y_test"])
            records.append(d)
    return records


# ---------------- 评测 ----------------

@torch.inference_mode()
def run_cls(model, tok, texts_a, texts_b, labels, device, batch_size, mode):
    """cls / ctx 模式: 返回预测标签"""
    preds = []
    for i in range(0, len(texts_a), batch_size):
        a = texts_a[i:i + batch_size]
        b = texts_b[i:i + batch_size] if texts_b is not None else None
        enc = tok(a, b, padding=True, truncation=True, max_length=8192,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds).numpy()


@torch.inference_mode()
def run_mlm(model, tok, texts, labels, device, batch_size,
            id_better: int, id_worse: int):
    """mlm 模式: 比较 [MASK] 位置 'better' vs 'worse' 的 logits"""
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=8192, return_tensors="pt").to(device)
        out = model(**enc).logits                      # (B, L, V)
        mask_pos = (enc["input_ids"] == tok.mask_token_id).nonzero()
        row_logits = out[mask_pos[:, 0], mask_pos[:, 1]]   # (B, V)
        pred = (row_logits[:, id_better] > row_logits[:, id_worse]).long()
        preds.append(pred.cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=Path(__file__).parent / "data")
    ap.add_argument("--model-path",
                    default="/Users/bingchen/Desktop/20260925/ModernBert/ModernBERT-base")
    ap.add_argument("--mode", choices=["cls", "ctx", "mlm"], default="cls")
    ap.add_argument("--beta", type=int, default=5)
    ap.add_argument("--n-evidence", type=int, default=12,
                    help="ctx 模式每个锚点抽取的证据关系条数")
    ap.add_argument("--max-pairs", type=int, default=400,
                    help="每条 record 最多评测的 (锚点,候选) 对数 (ctx 模式防超时)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    records = load_records(args.data_dir)
    print(f"[{args.mode}] device={device}, records={len(records)}, "
          f"model={Path(args.model_path).name}")

    if args.mode == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device).eval()
        # ModernBERT 是 BPE 词表, mask 后接的词带前导空格: " better"/" worse" 才是单 token
        bet = tok.encode(" better", add_special_tokens=False)
        wor = tok.encode(" worse", add_special_tokens=False)
        assert len(bet) == 1 and len(wor) == 1, "' better'/' worse' 必须是单 token"
        id_better, id_worse = bet[0], wor[0]
        bs = args.batch_size or 128
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=2).to(device).eval()
        bs = args.batch_size or (32 if args.mode == "ctx" else 128)

    per_record, per_group = [], {}
    for rec in records:
        Xs, ys = rec["X_train"], rec["y_train"]
        Us, uys = rec["X_test"], rec["y_test"]
        Ns, Nu = joint_normalize(Xs, Us)               # R2SAEA 同款归一化

        texts_a, texts_b, labels = [], [], []
        for i in range(len(Ns)):
            if args.mode == "ctx":
                others = np.delete(np.arange(len(Ns)), i)
                chosen = rng.choice(others, min(args.n_evidence, len(others)),
                                    replace=False)
                ev = evidence_block(Ns, ys, i, chosen, args.beta)
            for k in range(len(Nu)):
                query = f"A = <{fmt_vec(Ns[i], args.beta)}> ; B = <{fmt_vec(Nu[k], args.beta)}>"
                if args.mode == "mlm":
                    texts_a.append(f"{query} . A is [MASK] than B .")
                else:
                    texts_a.append(query)
                    if args.mode == "ctx":
                        texts_b.append(ev)
                labels.append(int(ys[i] < uys[k]))     # 1=better (最小化)

        if args.max_pairs and args.mode == "ctx" and len(texts_a) > args.max_pairs:
            idx = rng.choice(len(texts_a), args.max_pairs, replace=False)
            texts_a = [texts_a[j] for j in idx]
            texts_b = [texts_b[j] for j in idx]
            labels = [labels[j] for j in idx]

        if args.mode == "mlm":
            preds = run_mlm(model, tok, texts_a, labels, device, bs,
                            id_better, id_worse)
        else:
            preds = run_cls(model, tok, texts_a, texts_b or None, labels,
                            device, bs, args.mode)

        acc = float(np.mean(preds == np.array(labels)))
        per_record.append(acc)
        key = f"{rec['prob_name']}_D{rec['prob_D']}"
        per_group.setdefault(key, []).append(acc)
        print(f"  {key:24s} inst-rep n_pairs={len(labels):4d}  acc={acc:.4f}")

    print("-" * 60)
    print(f"[{args.mode}] 总体: mean={np.mean(per_record):.4f}  "
          f"std={np.std(per_record):.4f}  n_records={len(per_record)}")
    for key, accs in sorted(per_group.items()):
        print(f"  {key:24s} mean={np.mean(accs):.4f}  (n={len(accs)})")


if __name__ == "__main__":
    main()
