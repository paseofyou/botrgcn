#!/usr/bin/env python3
"""
实验结果聚合脚本
================
把 experiments/results.csv 里的多种子运行按 (dataset, model, variant) 分组,
输出 mean±std 的表格, 可直接粘进论文。

用法:
    python experiments/aggregate.py                        # 全部结果, markdown
    python experiments/aggregate.py --dataset TwiBot-22    # 只看一个数据集
    python experiments/aggregate.py --format latex         # 输出 LaTeX 表格
    python experiments/aggregate.py --min-seeds 5          # 只显示跑满 5 个种子的组
    python experiments/aggregate.py --ttest pure v2        # 对两组做配对 t 检验

注意:
  - commit 带 -dirty 的行代表运行时有未提交改动, 默认会被标红提示并排除,
    用 --allow-dirty 可以强行包含 (但不要写进论文)。
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "results.csv")

METRICS = ["test_acc", "precision", "recall", "f1", "mcc"]
METRIC_LABELS = {
    "test_acc": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}


def load_rows(csv_path, allow_dirty=False):
    if not os.path.exists(csv_path):
        raise SystemExit(f"结果表不存在: {csv_path}\n先跑几次训练, 训练脚本会自动建表。")

    rows, dirty = [], 0
    # utf-8-sig: 容忍用 Excel 打开再另存导致的 BOM
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get("commit", "")).endswith("-dirty"):
                dirty += 1
                if not allow_dirty:
                    continue
            rows.append(row)

    if dirty:
        note = "已包含" if allow_dirty else "已排除"
        print(f"[!] {dirty} 行来自 dirty 工作区 ({note})。这些结果不可复现, 不要写进论文。\n")
    return rows


def group(rows, keys=("dataset", "model", "variant")):
    """按 (dataset, model, variant) 分组, 组内每个 seed 只保留最新一次运行。"""
    buckets = defaultdict(dict)  # key -> {seed: row}
    for row in rows:
        k = tuple(row.get(c, "") for c in keys)
        seed = row.get("seed", "")
        prev = buckets[k].get(seed)
        # time 是 ISO 格式, 字符串比较即时间先后
        if prev is None or row.get("time", "") >= prev.get("time", ""):
            buckets[k][seed] = row
    return buckets


def summarize(buckets, min_seeds=1):
    out = []
    for k in sorted(buckets):
        runs = list(buckets[k].values())
        if len(runs) < min_seeds:
            continue
        entry = {"key": k, "n": len(runs), "seeds": sorted(buckets[k])}
        for m in METRICS:
            vals = [float(r[m]) for r in runs if r.get(m) not in (None, "")]
            entry[m] = (np.mean(vals) * 100, np.std(vals, ddof=0) * 100) if vals else (float("nan"),) * 2
        out.append(entry)
    return out


def fmt_markdown(entries):
    header = ["Dataset", "Model", "Variant", "n"] + [METRIC_LABELS[m] for m in METRICS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for e in entries:
        cells = list(e["key"]) + [str(e["n"])]
        cells += [f"{e[m][0]:.2f}±{e[m][1]:.2f}" for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_latex(entries):
    cols = "lll" + "c" * (len(METRICS) + 1)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "Dataset & Model & Variant & $n$ & " +
             " & ".join(METRIC_LABELS[m] for m in METRICS) + r" \\", r"\midrule"]
    for e in entries:
        cells = [str(c).replace("_", r"\_") for c in e["key"]] + [str(e["n"])]
        cells += [f"${e[m][0]:.2f}_{{\\pm{e[m][1]:.2f}}}$" for m in METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def paired_ttest(buckets, a_variant, b_variant, dataset, metric):
    """对两个变体在相同种子上做配对 t 检验 (审稿人会问显著性)。"""
    try:
        from scipy import stats
    except ImportError:
        print("[!] 未安装 scipy, 跳过 t 检验 (pip install scipy)")
        return

    def pick(v):
        for k, seeds in buckets.items():
            if k[0] == dataset and (k[2] == v or k[1] == v):
                return seeds
        return None

    A, B = pick(a_variant), pick(b_variant)
    if not A or not B:
        print(f"[!] 在 {dataset} 上找不到 '{a_variant}' 或 '{b_variant}' 的结果")
        return

    common = sorted(set(A) & set(B))
    if len(common) < 2:
        print(f"[!] 共同种子只有 {len(common)} 个, 无法做 t 检验")
        return

    a = np.array([float(A[s][metric]) for s in common])
    b = np.array([float(B[s][metric]) for s in common])
    t, p = stats.ttest_rel(b, a)
    print(f"\n=== 配对 t 检验 ({dataset}, {metric}) ===")
    print(f"  种子: {common}")
    print(f"  {a_variant}: {a.mean()*100:.2f}  |  {b_variant}: {b.mean()*100:.2f}")
    print(f"  Δ = {(b.mean()-a.mean())*100:+.2f}  t = {t:.3f}  p = {p:.4f}"
          f"  {'(显著, p<0.05)' if p < 0.05 else '(不显著)'}")


def main():
    ap = argparse.ArgumentParser(description="聚合 results.csv 为论文表格")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--dataset", default=None, help="只看某个数据集 (TwiBot-20 / TwiBot-22)")
    ap.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    ap.add_argument("--min-seeds", type=int, default=1, help="至少跑了多少个种子才显示")
    ap.add_argument("--allow-dirty", action="store_true", help="包含 dirty 工作区的运行")
    ap.add_argument("--ttest", nargs=2, metavar=("A", "B"),
                    help="对两个 model/variant 做配对 t 检验, 如 --ttest pure v2")
    ap.add_argument("--ttest-metric", default="f1", choices=METRICS)
    args = ap.parse_args()

    rows = load_rows(args.csv, args.allow_dirty)
    if args.dataset:
        rows = [r for r in rows if r.get("dataset") == args.dataset]
    if not rows:
        raise SystemExit("没有可用结果。")

    buckets = group(rows)
    entries = summarize(buckets, args.min_seeds)
    if not entries:
        raise SystemExit(f"没有分组满足 --min-seeds {args.min_seeds}。")

    print(fmt_latex(entries) if args.format == "latex" else fmt_markdown(entries))
    print(f"\n(数值为百分比, 格式 mean±std, 共 {len(rows)} 条运行记录)")

    # 提示种子数不齐的组
    incomplete = [e for e in entries if e["n"] < 5]
    if incomplete:
        print("\n[!] 以下分组种子数 < 5, 论文里报这些数字会被质疑:")
        for e in incomplete:
            print(f"    {' / '.join(e['key'])}: n={e['n']} seeds={e['seeds']}")

    if args.ttest:
        ds = args.dataset or entries[0]["key"][0]
        paired_ttest(buckets, args.ttest[0], args.ttest[1], ds, args.ttest_metric)


if __name__ == "__main__":
    main()
