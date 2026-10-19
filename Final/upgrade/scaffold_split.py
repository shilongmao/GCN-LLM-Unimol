# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
scaffold_split.py — 用 Murcko 骨架划分替换随机划分，消除类似物泄漏
==================================================================

为什么
------
原工程的数据集构建脚本（已移除）用的是随机划分::

    train_idx, temp = train_test_split(idx, test_size=0.2, random_state=42)
    val_idx,  test  = train_test_split(temp, test_size=0.5, random_state=42)

630 个分子**全部来自同一个靶点**，且是 ChEMBL 同系物系列
（``raw_data.csv`` 里大量 ``CHEMBL3127xxx`` 只差一个取代基）。
随机划分会让近乎同构的类似物同时出现在训练和测试集 —— 这是
``results_complete`` 里 test AUC 0.827 的最可能来源。

做法
----
按 **Bemis-Murcko 骨架**（去掉侧链、只保留环系与连接子）分组，
同骨架分子整体进同一个 split，保证测试集骨架从未在训练集出现。

用法::

    python scaffold_split.py \
        --input ../Data/output/processed_data.csv \
        --smiles-col smiles --label-col label \
        --ratios 0.8 0.1 0.1 --seed 42 \
        --out-dir ../Data/output/splits_scaffold

输出::

    splits_scaffold/train.csv, val.csv, test.csv
    splits_scaffold/report.txt      # 骨架统计 + 泄漏自检
    splits_scaffold/scaffolds.csv   # 每个分子的骨架

依赖: rdkit + pandas
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """返回 Bemis-Murcko 骨架 SMILES；解析失败返回空串。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_groups(
    smiles: Sequence[str],
    include_chirality: bool = False,
) -> Tuple[Dict[str, List[int]], List[str]]:
    """smiles -> {scaffold: [分子索引]}，以及每个分子的骨架列表。"""
    groups: Dict[str, List[int]] = defaultdict(list)
    per_mol: List[str] = []
    for i, smi in enumerate(smiles):
        scaf = murcko_scaffold(smi, include_chirality)
        # 空骨架（无环小分子 / 解析失败）按自身 SMILES 单独成组，避免全部落一个桶
        key = scaf if scaf else f"__nolib__::{smi}"
        groups[key].append(i)
        per_mol.append(key)
    return dict(groups), per_mol


def greedy_scaffold_split(
    groups: Dict[str, List[int]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    n_total: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    """按骨架组从大到小贪心分配，尽量逼近目标比例。

    这是 MoleculeNet / ChemProp 常用的做法，优点是可复现、无随机性
    （除同尺寸骨架的排序用 seed 打乱）。
    """
    import random

    rng = random.Random(seed)
    sizes = {k: len(v) for k, v in groups.items()}
    # 大骨架优先；同尺寸用 seed 决定顺序，保证可复现
    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: sizes[k], reverse=True)

    total = n_total if n_total is not None else sum(sizes.values())
    targets = [total * r for r in ratios]

    buckets: List[List[str]] = [[], [], []]
    counts = [0, 0, 0]
    for k in keys:
        # 放到"距离目标比例缺口最大"的那个 split
        deficits = [targets[j] - counts[j] for j in range(3)]
        j = max(range(3), key=lambda t: deficits[t])
        buckets[j].append(k)
        counts[j] += sizes[k]

    idx = [[i for k in buckets[j] for i in groups[k]] for j in range(3)]
    return idx[0], idx[1], idx[2]


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Murcko 骨架划分")
    ap.add_argument("--input", required=True, help="CSV 文件")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-chirality", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"读取 {len(df)} 行, 列 = {list(df.columns)[:8]}...")

    # --- 1. 去重（同一 canonical SMILES 只保留一次）-------------------------
    df = df.copy()
    df["_canon"] = df[args.smiles_col].map(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )
    n_before = len(df)
    df = df[df["_canon"].notna()].drop_duplicates("_canon").reset_index(drop=True)
    print(f"去重/去非法: {n_before} -> {len(df)}")

    # --- 2. 骨架分组 --------------------------------------------------------
    groups, per_mol = scaffold_groups(df["_canon"].tolist(), args.include_chirality)
    df["scaffold"] = per_mol
    print(f"骨架数: {len(groups)}  （平均每组 {len(df)/len(groups):.2f} 个分子）")

    # --- 3. 贪心划分 --------------------------------------------------------
    tr, va, te = greedy_scaffold_split(
        groups, tuple(args.ratios), seed=args.seed, n_total=len(df)
    )
    print(f"split 大小: train={len(tr)} val={len(va)} test={len(te)}")

    # --- 4. 泄漏自检 --------------------------------------------------------
    scaf_tr = set(df.loc[tr, "scaffold"])
    scaf_va = set(df.loc[va, "scaffold"])
    scaf_te = set(df.loc[te, "scaffold"])
    smi_tr = set(df.loc[tr, "_canon"])
    smi_te = set(df.loc[te, "_canon"])

    overlap_scaf_tr_te = scaf_tr & scaf_te
    overlap_smi_tr_te = smi_tr & smi_te

    # 阳性率
    def pos_rate(idx):
        if len(idx) == 0:
            return float("nan")
        return float(df.loc[idx, args.label_col].mean())

    # --- 5. 落盘 ------------------------------------------------------------
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        split_df = df.loc[idx].drop(columns=["_canon", "scaffold"])
        split_df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    df[["scaffold"]].assign(**{args.smiles_col: df["_canon"]}).to_csv(
        os.path.join(args.out_dir, "scaffolds.csv"), index=False
    )

    # --- 6. 报告 ------------------------------------------------------------
    lines = [
        "Murcko 骨架划分报告",
        "=" * 50,
        f"输入文件        : {args.input}",
        f"分子数(去重后)  : {len(df)}",
        f"骨架数          : {len(groups)}",
        f"ratio 目标      : {args.ratios}",
        f"实际大小        : train={len(tr)} val={len(va)} test={len(te)}",
        "",
        "阳性率：",
        f"  train = {pos_rate(tr):.3f}",
        f"  val   = {pos_rate(va):.3f}",
        f"  test  = {pos_rate(te):.3f}",
        "",
        "泄漏自检（应为 0）：",
        f"  train ∩ test 骨架重叠 = {len(overlap_scaf_tr_te)}",
        f"  train ∩ test SMILES 重叠 = {len(overlap_smi_tr_te)}",
        f"  val   ∩ test 骨架重叠 = {len(scaf_va & scaf_te)}",
        "",
        "最大骨架组（前 10）：",
    ]
    top = Counter({k: len(v) for k, v in groups.items()}).most_common(10)
    for scaf, cnt in top:
        lines.append(f"  {cnt:4d}  {scaf[:80]}")

    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print()
    print(report)
    print()
    print(f"写出至 {args.out_dir}/")

    if overlap_scaf_tr_te or overlap_smi_tr_te:
        print("[!] 警告：仍存在重叠，检查是否有空骨架组被合并")
    else:
        print("[OK] 无骨架泄漏、无 SMILES 重复")


if __name__ == "__main__":
    main()
